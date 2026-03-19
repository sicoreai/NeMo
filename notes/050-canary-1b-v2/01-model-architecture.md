# Canary 1B v2 — Model Architecture

## Class Hierarchy

```
EncDecMultiTaskModel                         # aed_multitask_models.py:133
  ├── ASRModel                               # Base ASR model
  ├── ExportableEncDecModel                  # ONNX/TRT export support
  ├── ASRBPEMixin                            # Tokenizer setup (mixins.py)
  ├── ASRModuleMixin                         # Module utilities
  └── ASRTranscriptionMixin                  # transcribe() API
```

## Component Diagram

```
Audio (16kHz waveform)
  │
  ▼
┌───────────────────────────┐
│  Preprocessor             │  AudioToMelSpectrogramPreprocessor
│  128 mel bins, 25ms win   │  audio_preprocessing.py
│  10ms stride, Hann, n_fft=512
└───────────┬───────────────┘
            │  (B, 128, T)
  ▼
┌───────────────────────────┐
│  SpecAugment              │  2 freq masks (27 bins), 10 time masks (5%)
└───────────┬───────────────┘
            │
  ▼
┌───────────────────────────┐
│  FastConformer Encoder    │  conformer_encoder.py
│  32 layers, d=1024, 8 heads│
│  8x subsampling (dw_stride)│
│  conv_kernel=9, rel_pos    │
└───────────┬───────────────┘
            │  encoded: (B, T/8, 1024)
            │
  ▼
┌───────────────────────────┐
│  Encoder-Decoder Proj     │  Identity (1024→1024)
│  (Linear if sizes differ) │  aed_multitask_models.py:159-165
└───────────┬───────────────┘
            │
            ├─────────────────────────── cross-attention ──┐
            │                                              │
            │                                    ┌─────────┴──────────┐
            │                                    │  Transformer       │
  Prompt tokens ──────────────────────────────►  │  Decoder           │
  (autoregressive)                               │  8 layers, d=1024  │
                                                 │  8 heads, FFN=4096 │
                                                 │  max_seq_len=512   │
                                                 └─────────┬──────────┘
                                                           │  (B, S, 1024)
                                                   ▼
                                                 ┌─────────────────────┐
                                                 │  Token Classifier   │  (head)
                                                 │  1024 → vocab_size  │
                                                 │  + log_softmax      │
                                                 └─────────┬───────────┘
                                                           │  (B, S, V)
                                                   ▼
                                                 logits / predictions
```

## Encoder: FastConformer

**File**: `nemo/collections/asr/modules/conformer_encoder.py`

| Parameter | Value |
|-----------|-------|
| n_layers | **32** |
| d_model (hidden) | 1024 |
| n_heads | 8 |
| ff_expansion_factor | 4 (FFN inner = 4096) |
| conv_kernel_size | 9 |
| subsampling | `dw_striding` (depthwise striding) |
| subsampling_factor | 8 (10ms → 80ms frames) |
| subsampling_conv_channels | 256 |
| self_attention_model | `rel_pos` (relative positional encoding) |
| att_context_size | [-1, -1] (unlimited context) |
| dropout | 0.1 |

Each Conformer block consists of:
1. Feed-forward module (half-step)
2. Multi-head self-attention with relative positional encoding
3. Convolution module (depthwise separable, kernel=9)
4. Feed-forward module (half-step)
5. Layer normalization

## Decoder: Transformer

**File**: `nemo/collections/asr/modules/transformer/transformer.py`

| Parameter | Value |
|-----------|-------|
| num_layers | **8** |
| hidden_size | 1024 |
| inner_size | 4096 (4x hidden) |
| num_attention_heads | 8 |
| max_sequence_length | 1024 tokens |
| hidden_act | relu |
| pre_ln | true (Pre-LayerNorm) |
| attn_score_dropout | 0.1 |
| attn_layer_dropout | 0.1 |
| ffn_dropout | 0.1 |

Each decoder layer:
1. Masked self-attention (causal — can only attend to previous tokens)
2. Cross-attention (attends to encoder output)
3. Feed-forward network (1024 → 4096 → 1024)

## Head: Token Classifier

**File**: `nemo/collections/asr/parts/submodules/token_classifier.py`

```yaml
head:
  _target_: nemo.collections.asr.parts.submodules.token_classifier.TokenClassifier
  num_layers: 1
  activation: relu
  log_softmax: true
  hidden_size: 1024
  num_classes: <vocab_size>  # set at runtime, rounded up to multiple of 8
  dropout: 0.0
  use_transformer_init: true
```

**Weight tying** (aed_multitask_models.py:198-200): The token classifier's weight matrix is tied to the decoder's token embedding. This means the output projection and input embedding share the same parameters.

## Parameter Distribution (978M total)

| Component | Estimated Params | % |
|-----------|-----------------|---|
| FastConformer Encoder (32 layers, d=1024) | ~870M | ~89% |
| Transformer Decoder (8 layers, d=1024) | ~100M | ~10% |
| Embeddings + Head (weight-tied) | ~8M | ~1% |
| **Total** | **978M** | 100% |

The encoder heavily dominates because:
1. 32 layers vs 8 — 4x more layers
2. Each Conformer block has both attention AND convolution modules, making it larger than a standard Transformer layer

This is a "flash"-style architecture: heavy encoder for strong acoustic modeling, light decoder for fast autoregressive generation.

## Tokenizer: CanaryBPETokenizer

**File**: `nemo/collections/common/tokenizers/canary_tokenizer.py`

Canary 1B v2 uses a **unified BPE** tokenizer (single SentencePiece model for all languages), wrapped by `CanaryBPETokenizer` which overrides BOS/EOS/PAD with Canary-specific special tokens.

| Property | Value |
|----------|-------|
| Type | `bpe` with `CanaryBPETokenizer` wrapper |
| Vocab size | 16384 |
| Special tokens | ~1163 (language tags, task tags, emotion, etc.) |
| Normal tokens | ~15221 (multilingual subword pieces) |

Key special tokens:
- `<|startoftranscript|>` — BOS
- `<|endoftext|>` — EOS
- `<pad>` — padding
- `<|nospeech|>` — no speech detected
- `<|pnc|>` / `<|nopnc|>` — punctuation control
- `<|startofcontext|>` — v2 context start
- `<|spltoken0|>` ... `<|spltoken29|>` — reserved placeholder tokens

## Key Source Files

| File | Contents |
|------|----------|
| `nemo/collections/asr/models/aed_multitask_models.py` | EncDecMultiTaskModel main class |
| `nemo/collections/asr/modules/conformer_encoder.py` | FastConformer encoder |
| `nemo/collections/asr/modules/transformer/transformer.py` | Transformer decoder |
| `nemo/collections/asr/parts/submodules/token_classifier.py` | Output head |
| `nemo/collections/common/tokenizers/canary_tokenizer.py` | CanaryBPETokenizer |
| `nemo/collections/asr/parts/mixins/mixins.py` | ASRBPEMixin (tokenizer setup) |
| `examples/asr/conf/speech_multitask/fast-conformer_aed.yaml` | Reference config |
