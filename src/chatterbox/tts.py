from dataclasses import dataclass
from pathlib import Path
import time
from typing import Generator, Tuple, Optional

import librosa
import numpy as np
import torch
import perth
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"


def punc_norm(text: str) -> str:
    """
    Quick cleanup func for punctuation from LLMs or
    containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", '"'),
        ("”", '"'),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """

    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(t3=self.t3.__dict__, gen=self.gen)
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs["t3"]), kwargs["gen"])


@dataclass
class StreamingMetrics:
    """Metrics for streaming TTS generation"""

    latency_to_first_chunk: Optional[float] = None
    rtf: Optional[float] = None
    total_generation_time: Optional[float] = None
    total_audio_duration: Optional[float] = None
    chunk_count: int = 0


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()
        self._warmed = False
        self._hift_cache = None

    @classmethod
    def from_local(cls, ckpt_dir, device) -> "ChatterboxTTS":
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device("cpu")
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(torch.load(ckpt_dir / "ve.pt", map_location=map_location))
        ve.to(device).eval()

        t3 = T3()
        t3_state = torch.load(ckpt_dir / "t3_cfg.pt", map_location=map_location)
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(torch.load(ckpt_dir / "s3gen.pt", map_location=map_location))
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device, warmup: bool = True) -> "ChatterboxTTS":
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine."
                )
            device = "cpu"

        for fpath in ["ve.pt", "t3_cfg.pt", "s3gen.pt", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        instance = cls.from_local(Path(local_path).parent, device)
        if warmup:
            try:
                instance.warmup()
            except Exception as e:
                # don't hard-fail on warmup errors — print for visibility
                print(f"Model warmup failed (continuing): {e}")
        return instance

    def warmup(self, *, text: str = "Hello.", max_new_tokens: int = 8, chunk_size: int = 4, print_progress: bool = True):
        """
        Run a short forward pass to "warm" compiled kernels, caches and allocate memory on the device.
        Requires conditionals (self.conds) because the S3Gen pipeline needs a ref_dict.
        - text: small text to tokenize for a short T3 inference pass
        - max_new_tokens: how many tokens to step to trigger compilation
        - chunk_size: chunk size used for inference_stream
        """
        if self._warmed:
            if print_progress:
                print("Model already warmed — skipping warmup.")
            return

        if self.conds is None:
            # Warmup depends on conditionals (ref dict) to run a representative S3Gen pass.
            print(
                "No conditionals present (self.conds is None). Skipping warmup. "
                "Call prepare_conditionals(...) or pass audio_prompt_path to generate to enable warmup."
            )
            return

        if print_progress:
            print(f"Warming up models on device={self.device} — this will run a tiny forward pass...")

        start_time = time.time()
        with torch.inference_mode():
            # Build a pair of text tokens for CFG path similar to generate/generate_stream
            text = punc_norm(text)
            text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # CFG double-batch
            sot = self.t3.hp.start_text_token
            eot = self.t3.hp.stop_text_token
            text_tokens = F.pad(text_tokens, (1, 0), value=sot)
            text_tokens = F.pad(text_tokens, (0, 1), value=eot)

            # Use inference_stream to trigger compilation of patched_model and an initial generation step.
            # We only consume the first yielded token chunk, then pass it through _process_token_buffer to warm S3Gen.
            try:
                gen = self.inference_stream(
                    t3_cond=self.conds.t3,
                    text_tokens=text_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    cfg_weight=0.0,
                    chunk_size=chunk_size,
                )

                # get the first token chunk (or until EOS) and run S3Gen inference on it
                for token_chunk in gen:
                    # Each yielded chunk is a tensor (batch, tokens). We extract the conditional batch.
                    token_chunk = token_chunk[0]  # conditional batch
                    # call internal processor: pass minimal args to force S3Gen forward pass
                    metrics = StreamingMetrics()
                    audio_tensor, audio_duration, success = self._process_token_buffer(
                        [token_chunk],
                        all_tokens_so_far=[],
                        context_window=0,
                        start_time=start_time,
                        metrics=metrics,
                        print_metrics=False,
                        fade_duration=0.0,
                    )
                    # We don't use the output, just force the forward call
                    break
            except Exception as e:
                # If anything goes wrong, don't fail hard; report and continue.
                print(f"Warmup encountered an error (non-fatal): {e}")

        self._warmed = True
        if print_progress:
            print(f"Warmup complete (took {time.time() - start_time:.3f}s).")

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[: self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    ):
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    def inference_stream(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.Tensor,
        max_new_tokens=1000,
        temperature=0.8,
        cfg_weight=0.5,
        chunk_size=25,  # Number of tokens per chunk
    ) -> Generator[torch.Tensor, None, None]:
        """
        Streaming version of T3 inference that yields speech tokens in chunks
        """
        from tqdm import tqdm
        import torch.nn.functional as F
        from transformers.generation.logits_process import TopPLogitsWarper, RepetitionPenaltyLogitsProcessor

        # Validate inputs
        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)

        # Default initial speech to a single start-of-speech token
        initial_speech_tokens = self.t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])

        # Prepare custom input embeds
        embeds, len_cond = self.t3.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
        )

        # Setup model if not compiled
        if not self.t3.compiled:
            from .models.t3.inference.alignment_stream_analyzer import AlignmentStreamAnalyzer
            from .models.t3.inference.t3_hf_backend import T3HuggingfaceBackend

            alignment_stream_analyzer = AlignmentStreamAnalyzer(
                self.t3.tfmr,
                None,
                text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
                alignment_layer_idx=9,
                eos_idx=self.t3.hp.stop_speech_token,
            )
            patched_model = T3HuggingfaceBackend(
                config=self.t3.cfg,
                llama=self.t3.tfmr,
                speech_enc=self.t3.speech_emb,
                speech_head=self.t3.speech_head,
                alignment_stream_analyzer=alignment_stream_analyzer,
            )
            self.t3.patched_model = patched_model
            self.t3.compiled = True

        device = embeds.device

        bos_token = torch.tensor([[self.t3.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = self.t3.speech_emb(bos_token)
        bos_embed = bos_embed + self.t3.speech_pos_emb.get_fixed_embedding(0)

        # batch_size=2 for CFG
        bos_embed = torch.cat([bos_embed, bos_embed])

        # Combine condition and BOS token for the initial input
        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)

        # Track generated token ids
        generated_ids = bos_token.clone()
        predicted = []
        chunk_buffer = []

        # Instantiate logits processors
        top_p_warper = TopPLogitsWarper(top_p=0.8)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=2.0)

        # Initial forward pass
        output = self.t3.patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = output.past_key_values

        # Generation loop
        for i in range(max_new_tokens):
            logits = output.logits[:, -1, :]

            # CFG
            logits_cond = logits[0:1]
            logits_uncond = logits[1:2]
            logits = logits_cond + cfg_weight * (logits_cond - logits_uncond)
            logits = logits.squeeze(1)

            # Apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Apply repetition penalty and top‑p filtering
            logits = repetition_penalty_processor(generated_ids, logits)
            logits = top_p_warper(None, logits)

            # Convert logits to probabilities and sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            predicted.append(next_token)
            chunk_buffer.append(next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Check for EOS token
            if next_token.view(-1) == self.t3.hp.stop_speech_token:
                # Yield final chunk if buffer has tokens
                if chunk_buffer:
                    yield torch.cat(chunk_buffer, dim=1)
                break

            # Yield chunk when buffer is full
            if len(chunk_buffer) >= chunk_size:
                yield torch.cat(chunk_buffer, dim=1)
                chunk_buffer = []

            # Get embedding for the new token
            next_token_embed = self.t3.speech_emb(next_token)
            next_token_embed = next_token_embed + self.t3.speech_pos_emb.get_fixed_embedding(i + 1)

            # For CFG
            next_token_embed = torch.cat([next_token_embed, next_token_embed])

            # Forward pass with cached past
            output = self.t3.patched_model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = output.past_key_values

    def _process_token_buffer(
        self,
        token_buffer,
        all_tokens_so_far,
        context_window,
        start_time,
        metrics,
        print_metrics,
        fade_duration=0.02,  # seconds to apply linear fade-in on each chunk
        cache_hot_path=True,
    ):
        # Combine buffered chunks of tokens
        new_tokens = torch.cat(token_buffer, dim=-1)

        # Build tokens_to_process by including a context window
        if len(all_tokens_so_far) > 0:
            context_tokens = all_tokens_so_far[-context_window:] if len(all_tokens_so_far) > context_window else all_tokens_so_far
            tokens_to_process = torch.cat([context_tokens, new_tokens], dim=-1)
            context_length = len(context_tokens)
        else:
            tokens_to_process = new_tokens
            context_length = 0

        # Drop any invalid tokens and move to the correct device
        clean_tokens = drop_invalid_tokens(tokens_to_process).to(self.device)
        if len(clean_tokens) == 0:
            return None, 0.0, False

        # Run S3Gen inference to get a waveform (1 × T)
        if cache_hot_path:
            mels = self.s3gen.flow_inference(speech_tokens=clean_tokens, ref_dict=self.conds.gen, finalize=False)
            wav, new_cache = self.s3gen.hift_inference(speech_feat=mels, cache_source=self._hift_cache)
            self._hift_cache = new_cache
        else:
            mels_final = self.s3gen.flow_inference(speech_tokens=clean_tokens, ref_dict=self.conds.gen, finalize=True)
            wav, _ = self.s3gen.hift_inference(speech_feat=mels_final, cache_source=self._hift_cache)

        wav = wav.squeeze(0).detach().cpu().numpy()

        # If we have context tokens, crop out the samples corresponding to them
        if context_length > 0:
            samples_per_token = len(wav) / len(clean_tokens)
            skip_samples = int(context_length * samples_per_token)
            audio_chunk = wav[skip_samples:]
        else:
            audio_chunk = wav

        if len(audio_chunk) == 0:
            return None, 0.0, False

        # Apply a short linear fade-in on the new chunk to smooth boundaries
        fade_samples = int(fade_duration * self.sr)
        if fade_samples > 0:
            if fade_samples > len(audio_chunk):
                fade_samples = len(audio_chunk)
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=audio_chunk.dtype)
            audio_chunk[:fade_samples] *= fade_in

        # Compute audio duration and watermark
        audio_duration = len(audio_chunk) / self.sr
        watermarked_chunk = self.watermarker.apply_watermark(audio_chunk, sample_rate=self.sr)
        audio_tensor = torch.from_numpy(watermarked_chunk).unsqueeze(0)

        # Update first‐chunk latency metric
        if metrics.chunk_count == 0:
            metrics.latency_to_first_chunk = time.time() - start_time
            if print_metrics:
                print(f"Latency to first chunk: {metrics.latency_to_first_chunk:.3f}s")

        metrics.chunk_count += 1
        return audio_tensor, audio_duration, True

    def generate_stream(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        chunk_size: int = 25,  # Tokens per chunk
        context_window=50,
        fade_duration=0.02,  # seconds to apply linear fade-in on each chunk
        print_metrics: bool = True,
    ) -> Generator[Tuple[torch.Tensor, StreamingMetrics], None, None]:
        """
        Streaming version of generate that yields audio chunks as they are generated.

        Args:
            text: Input text to synthesize
            audio_prompt_path: Optional path to reference audio for voice cloning
            exaggeration: Emotion exaggeration factor
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature
            chunk_size: Number of speech tokens per chunk
            context_window: The context passed for each chunk
            fade_duration: Seconds to apply linear fade-in on each chunk
            print_metrics: Whether to print RTF and latency metrics

        Yields:
            Tuple of (audio_chunk, metrics) where audio_chunk is a torch.Tensor
            and metrics contains timing information
        """
        start_time = time.time()
        metrics = StreamingMetrics()

        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        total_audio_length = 0.0
        all_tokens_processed = []  # Keep track of all tokens processed so far

        with torch.inference_mode():
            self._hift_cache = torch.zeros(1, 1, 0, device=self.device)

            # Stream speech tokens
            for token_chunk in self.inference_stream(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,
                temperature=temperature,
                cfg_weight=cfg_weight,
                chunk_size=chunk_size,
            ):
                # Extract only the conditional batch
                token_chunk = token_chunk[0]

                # Process each chunk immediately
                audio_tensor, audio_duration, success = self._process_token_buffer(
                    [token_chunk],
                    all_tokens_processed,
                    context_window,
                    start_time,
                    metrics,
                    print_metrics,
                    fade_duration,
                )

                if success:
                    total_audio_length += audio_duration
                    yield audio_tensor, metrics

                # Update all_tokens_processed with the new tokens
                if len(all_tokens_processed) == 0:
                    all_tokens_processed = token_chunk
                else:
                    all_tokens_processed = torch.cat([all_tokens_processed, token_chunk], dim=-1)

            if self._hift_cache is not None and len(all_tokens_processed) > 0:
                audio_tensor, audio_duration, success = self._process_token_buffer(
                    [token_chunk],
                    all_tokens_processed,
                    context_window,
                    start_time,
                    metrics,
                    print_metrics,
                    fade_duration,
                    cache_hot_path=False,
                )

                if success:
                    total_audio_length += audio_duration
                    yield audio_tensor, metrics
                else:
                    print("Error on last tokens")

        # Final metrics calculation
        metrics.total_generation_time = time.time() - start_time
        metrics.total_audio_duration = total_audio_length
        if total_audio_length > 0:
            metrics.rtf = metrics.total_generation_time / total_audio_length
            if print_metrics:
                print(f"Total generation time: {metrics.total_generation_time:.3f}s")
                print(f"Total audio duration: {metrics.total_audio_duration:.3f}s")
                print(f"RTF (Real-Time Factor): {metrics.rtf:.3f}")
                print(f"Total chunks yielded: {metrics.chunk_count}")
