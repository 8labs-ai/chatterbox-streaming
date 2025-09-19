import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")
text = "Welcome to the world of streaming text-to-speech! This audio will be generated and played in real-time chunks."

# Basic streaming
audio_chunks = []
for audio_chunk, metrics in model.generate_stream(text):
    audio_chunks.append(audio_chunk)
    # You can play audio_chunk immediately here for real-time playback
    print(f"Generated chunk {metrics.chunk_count}, RTF: {metrics.rtf:.3f}" if metrics.rtf else f"Chunk {metrics.chunk_count}")

# Combine all chunks into final audio
final_audio = torch.cat(audio_chunks, dim=-1)
ta.save("streaming_output.wav", final_audio, model.sr)
