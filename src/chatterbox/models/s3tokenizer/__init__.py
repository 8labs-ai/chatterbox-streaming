from .s3tokenizer import (
    S3_SR,
    S3_HOP,
    S3_TOKEN_HOP,
    S3_TOKEN_RATE,
    SPEECH_VOCAB_SIZE,
    S3Tokenizer,
)


SOS = SPEECH_VOCAB_SIZE
EOS = SPEECH_VOCAB_SIZE + 1



def drop_invalid_tokens(x):
    """Drop SoS and EoS"""
    assert len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[0] == 1), "only batch size of one allowed for now"
    if len(x.shape) == 2:
        x = x.squeeze(0)

    start_idxs = (x == SOS).nonzero(as_tuple=True)[0]
    if start_idxs.numel() > 0:
        s = int(start_idxs[0].item()) + 1
    else:
        s = 0

    end_idxs = (x == EOS).nonzero(as_tuple=True)[0]
    if end_idxs.numel() > 0:
        e = int(end_idxs[0].item())
    else:
        e = None

    return x[s:e]
