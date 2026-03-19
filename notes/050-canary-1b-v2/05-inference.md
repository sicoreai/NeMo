# Canary 1B v2 — Inference

## Transcription API

**File**: `nemo/collections/asr/models/aed_multitask_models.py:491-610`

### Basic Usage

```python
from nemo.collections.asr.models import EncDecMultiTaskModel

model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b-v2')

# Single file
result = model.transcribe("audio.wav", source_lang="el", target_lang="el", pnc="yes")

# Batch of files
results = model.transcribe(["a.wav", "b.wav"], batch_size=4, source_lang="el", target_lang="el")

# Translation
result = model.transcribe("greek.wav", source_lang="el", target_lang="en")

# NumPy array (16kHz)
import numpy as np
result = model.transcribe(audio_array, source_lang="el", target_lang="el")
```

### Accepted Input Formats

| Input | Type | Notes |
|-------|------|-------|
| File path | `str` | Single audio file |
| File list | `List[str]` | Multiple audio files |
| NumPy array | `np.ndarray` | Raw waveform (16kHz) |
| Manifest | `str` (.json/.jsonl) | NeMo manifest file |
| DataLoader | `DataLoader` | Pre-built dataloader |

### Prompt Specification (3 formats)

```python
# 1. Legacy keyword args (simplest)
model.transcribe("a.wav", source_lang="el", target_lang="el", pnc="yes")

# 2. Single-turn explicit
model.transcribe("a.wav", role="user", slots={
    "source_lang": "el", "target_lang": "el", "pnc": "yes"
})

# 3. Multi-turn (for context biasing)
model.transcribe("a.wav", turns=[
    {"role": "user", "slots": {"source_lang": "el", "target_lang": "el"}},
    {"role": "assistant", "slots": {"message": "previous context..."}},
    {"role": "user", "slots": {"source_lang": "el", "target_lang": "el"}},
])
```

### Key Parameters

```python
model.transcribe(
    audio,
    batch_size=4,               # inference batch size
    return_hypotheses=False,    # True to get Hypothesis objects with scores
    num_workers=0,              # DataLoader workers
    timestamps=False,           # word/segment-level timestamps
    override_config=None,       # MultiTaskTranscriptionConfig
    **prompt,                   # source_lang, target_lang, pnc, etc.
)
```

## Decoding Strategies

**File**: `nemo/collections/asr/parts/submodules/multitask_decoding.py`

### Configuration

```yaml
decoding:
  strategy: beam              # or "greedy", "greedy_batch"
  return_best_hypothesis: true
  beam:
    beam_size: 1              # beam_size=1 ≈ greedy but with caching
    len_pen: 0.0              # length penalty (0 = no penalty)
    max_generation_delta: 50  # max tokens beyond prompt length
```

### Available Strategies

| Strategy | Description | Speed | Quality |
|----------|-------------|-------|---------|
| `greedy` | Single best path, no caching | Slowest | Baseline |
| `greedy_batch` | Batched greedy with KV caching | Fast | Same as greedy |
| `beam` (size=1) | Beam search with caching | Fast | Slightly better |
| `beam` (size>1) | Full beam search | Slower | Best |

### Changing Decoding Strategy at Runtime

```python
from nemo.collections.asr.parts.submodules.multitask_decoding import MultiTaskDecodingConfig

decoding_cfg = MultiTaskDecodingConfig()
decoding_cfg.strategy = "beam"
decoding_cfg.beam.beam_size = 4
decoding_cfg.beam.len_pen = 0.6
model.change_decoding_strategy(decoding_cfg)
```

### Greedy Decoding

**File**: `nemo/collections/asr/parts/submodules/multitask_greedy_decoding.py`

- `TransformerAEDGreedyInfer`: Autoregressive token-by-token generation
- Supports temperature-based sampling (`temperature` parameter)
- Optional confidence score tracking per token

### Beam Search Decoding

**File**: `nemo/collections/asr/parts/submodules/multitask_beam_decoding.py`

- `TransformerAEDBeamInfer`: Standard beam search
- Configurable beam size, length penalty
- Optional N-gram LM integration for rescoring
- Returns n-best hypotheses if `return_best_hypothesis=False`

## Streaming / Chunked Inference

### Chunked Inference (Long Audio)

**File**: `examples/asr/asr_chunked_inference/aed/speech_to_text_aed_chunked_infer.py`

For audio files longer than ~40 seconds:
- Splits into non-overlapping chunks (default: 40s)
- Processes chunks in parallel batches
- Merges hypotheses from all chunks

```bash
python speech_to_text_aed_chunked_infer.py \
    model_path=canary-1b-v2.nemo \
    dataset_manifest=manifest.json \
    batch_size=8 \
    chunk_len_in_secs=40
```

### Streaming Inference (Real-time)

**File**: `examples/asr/asr_chunked_inference/aed/speech_to_text_aed_streaming_infer.py`

Two streaming policies:
- **Wait-k**: Predicts one token per audio chunk (higher latency, simpler)
- **AlignAtt**: Cross-attention driven — predicts when attention peaks (lower latency ~1.5s)

Configurable context windows: left_context, chunk_size, right_context.

## Hypothesis Object

When `return_hypotheses=True`:

```python
hypothesis = model.transcribe("a.wav", return_hypotheses=True, ...)[0]

hypothesis.text             # decoded text string
hypothesis.score            # decoding score/log-probability
hypothesis.y_sequence       # token ID sequence (torch.Tensor)
hypothesis.token_confidence # per-token confidence scores
hypothesis.word_confidence  # per-word confidence
hypothesis.timestep         # {'word': [...], 'segment': [...]}
```

## Timestamps

```python
results = model.transcribe("audio.wav", timestamps=True, source_lang="el", target_lang="el")
```

Timestamp computation:
- If external CTC model available: forced alignment (more accurate)
- Otherwise: CTC-free estimation from decoder attention

## ONNX / TensorRT Export

`EncDecMultiTaskModel` inherits `ExportableEncDecModel`, enabling ONNX export:

```python
model.export("model.onnx")
```

**Note**: AED models have limited export support due to the autoregressive decoder requiring iterative token generation. CTC/RNNT models export more straightforwardly.

## Key Source Files

| File | Contents |
|------|----------|
| `nemo/collections/asr/models/aed_multitask_models.py:491-610` | transcribe() method |
| `nemo/collections/asr/models/aed_multitask_models.py:958-1120` | _transcribe_forward/output |
| `nemo/collections/asr/parts/submodules/multitask_decoding.py` | Decoding strategy dispatcher |
| `nemo/collections/asr/parts/submodules/multitask_greedy_decoding.py` | Greedy decoder |
| `nemo/collections/asr/parts/submodules/multitask_beam_decoding.py` | Beam search decoder |
| `nemo/collections/asr/parts/mixins/transcription.py` | TranscriptionMixin base |
| `examples/asr/asr_chunked_inference/aed/` | Chunked & streaming inference scripts |
| `examples/asr/transcribe_speech_multitask.py` | Transcription config script |
