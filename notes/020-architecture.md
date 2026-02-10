# Architecture Notes

## ASR Model Families

### CTC Models (Connectionist Temporal Classification)
- Architecture: `Audio → Encoder → Linear Decoder → logits [B, T, V+1] → CTC Loss`
- Classes: `EncDecCTCModel` (char), `EncDecCTCBPEModel` (BPE subword)
- Simpler, no autoregressive decoding
- Single-task (ASR only)

### RNNT Models (RNN-Transducer)
- Architecture: `Audio → Encoder → ┐`
                                    `├→ Joint → logits [B, T, U, V+1] → RNNT Loss`
                `Previous tokens → Decoder → ┘`
- Classes: `EncDecRNNTModel` (char), `EncDecRNNTBPEModel` (BPE)
- Three components: encoder, decoder (prediction network), **joint network**
- Joint combines encoder + decoder at every (time, token) position
- Streaming-capable

### AED Models (Attention Encoder-Decoder) — Canary Family
- Architecture: `Audio → FastConformer Encoder → [Optional Transf Encoder] → Transformer Decoder → Head → logits`
- Class: `EncDecMultiTaskModel`
- Multitask: ASR, translation, timestamps, diarization, PnC, ITN
- Decoder is autoregressive (generates tokens one by one)
- Uses prompt tokens to control behavior

## Canary Model Variants

| Model | Params | Enc Layers | Dec Layers | Enc Hidden | Dec Hidden | Tokenizer |
|-------|--------|------------|------------|------------|------------|-----------|
| canary-1b | 1B | 24 | 24 | 1024 | 1024 | aggregate |
| canary-1b-flash | 883M | 32 | 4 | 1024 | 1024 | aggregate |
| canary-180m-flash | 182M | 17 | 4 | 512 | 1024 | aggregate |
| canary-1b-v2 | ~1B | - | - | - | - | unified BPE |

Flash models: more encoder layers, fewer decoder layers (speed/accuracy tradeoff).

## Encoder: FastConformer
- Location: `nemo/collections/asr/modules/ConformerEncoder`
- Conformer blocks: self-attention + convolution + feed-forward
- 8x subsampling via depthwise striding (10ms stride → 80ms frames)
- Relative positional encoding (`rel_pos`)

## Decoder: Transformer
- Location: `nemo/collections/asr/modules/transformer/`
- Standard Transformer decoder with cross-attention to encoder states
- `max_sequence_length` limits output length (positional embedding table size)
- `vocab_size` set at runtime from tokenizer

## Configuration Approaches
- **NeMo 1.0** (ASR, TTS): YAML-based via Hydra + `@hydra_runner` decorator
- **NeMo 2.0** (LLM, VLM): Python-based configuration (deprecated collections)

## Key Files
- `nemo/collections/asr/models/aed_multitask_models.py` — EncDecMultiTaskModel (Canary)
- `nemo/collections/asr/models/rnnt_bpe_models.py` — RNNT BPE models
- `nemo/collections/asr/models/ctc_bpe_models.py` — CTC BPE models
- `examples/asr/speech_multitask/speech_to_text_aed.py` — Canary training script
- `examples/asr/speech_to_text_finetune.py` — Generic fine-tuning script
- `examples/asr/conf/speech_multitask/fast-conformer_aed.yaml` — Canary config

## Prompt System (Canary)

### Decoder Input Structure
```
<|startofcontext|>{context}<|startoftranscript|><|emo:...|><|source_lang|><|target_lang|><|pnc|><|itn|><|timestamp|><|diarize|>
```

### Key Files
- `nemo/collections/common/prompts/canary2.py` — Canary2 prompt template
- `nemo/collections/common/prompts/formatter.py` — Base PromptFormatter class
- `nemo/collections/common/data/prompt_fn.py` — Registry for prompt format functions

### Prompt Slots
| Slot | Values | Purpose |
|------|--------|---------|
| source_lang | `<\|en\|>`, `<\|es\|>`, ... | Input audio language |
| target_lang | `<\|en\|>`, `<\|es\|>`, ... | Output text language |
| pnc | `<\|pnc\|>` / `<\|nopnc\|>` | Punctuation & capitalization |
| itn | `<\|itn\|>` / `<\|noitn\|>` | Inverse text normalization |
| timestamp | `<\|timestamp\|>` / `<\|notimestamp\|>` | Word timestamps |
| diarize | `<\|diarize\|>` / `<\|nodiarize\|>` | Speaker diarization |
| emotion | `<\|emo:undefined\|>`, etc. | Speaker emotion |

### Custom Behaviors via Placeholder Tokens
Default tokenizer includes `<|spltoken0|>` through `<|spltoken29|>` — unassigned tokens
for custom behaviors. Use pairs (on/off) and teach meaning through training data.

## Tokenizer Architecture

### Hierarchy
```
TokenizerSpec (abstract base)
├── SentencePieceTokenizer     — BPE/Unigram (.model files)
│   └── CanaryBPETokenizer     — Canary-specific wrapper
├── AggregateTokenizer         — Combines per-language tokenizers
│   └── CanaryTokenizer        — Multi-language + special tokens
└── AutoTokenizer              — HuggingFace wrapper
```

### Two Tokenizer Strategies
- **Aggregate** (canary-1b, flash models): Per-language SentencePiece + shared special tokens
- **Unified BPE** (canary-1b-v2): Single SentencePiece for all languages

### Key Files
- `nemo/collections/common/tokenizers/tokenizer_spec.py` — Abstract interface
- `nemo/collections/common/tokenizers/sentencepiece_tokenizer.py` — SPE wrapper
- `nemo/collections/common/tokenizers/canary_tokenizer.py` — Canary aggregate tokenizer

### SentencePiece Key Concepts
- `user_defined_symbols`: Always tokenized as single tokens, present in decoded output
- `control_symbols`: Must be inserted programmatically, stripped during decoding
- `▁` (U+2581): Word boundary marker prepended to word-starting tokens
- `sample_alpha`: Enables stochastic subword sampling for data augmentation

## Dataset Classes
- `AudioToCharDataset` — Character-level tokenization (small vocab ~28-100)
- `AudioToBPEDataset` — Subword tokenization via TokenizerSpec (vocab 128-32000)
- Both inherit from `_AudioTextDataset`, differ only in text tokenization
- Lhotse datasets: Dynamic batching by duration, bucketing support

## Data Pipeline

### Manifest Format
```json
{"audio_filepath": "/path/to/audio.wav", "text": "transcription", "duration": 1.5,
 "source_lang": "en", "target_lang": "en", "pnc": "yes"}
```

### Tarred Datasets
- Convert with: `scripts/speech_recognition/convert_to_tarred_audio_dataset.py`
- Produces: tar shards + tarred manifest + metadata.yaml
- Config: `is_tarred: true`, `tarred_audio_filepaths: audio_{0..N}.tar`
- Benefits: Sequential I/O, cloud storage friendly, natural multi-GPU sharding

### Lhotse Data Loading
- `use_lhotse: true` in config
- Dynamic batching: `batch_duration` (total seconds per batch)
- `quadratic_duration`: Penalty for long sequences (memory)
- `use_bucketing: true` + `num_buckets`: Group similar-length samples
- Lhotse handles distributed sampling (`use_distributed_sampler: false`)
