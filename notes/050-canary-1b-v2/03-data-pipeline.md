# Canary 1B v2 — Data Pipeline

## Pipeline Overview

```
Manifest JSON / Tarred audio
  │
  ▼
LazyNeMoIterator / LazyNeMoTarredIterator    (nemo_adapters.py)
  │  → Lhotse CutSet
  ▼
DynamicBucketingSampler                       (Lhotse)
  │  → groups by duration, respects batch_duration
  ▼
PromptedAudioToTextLhotseDataset              (audio_to_text_lhotse_prompted.py)
  │  → loads audio, tokenizes prompts, collates
  ▼
PromptedAudioToTextMiniBatch                  (dataclass)
  │  → audio, audio_lens, prompt, transcript, prompted_transcript
  ▼
Model forward pass
```

## Manifest Format

### Required Fields

```json
{"audio_filepath": "/path/to/audio.wav", "duration": 5.234, "text": "transcription"}
```

### Canary-Specific Fields

```json
{
  "audio_filepath": "/path/to/audio.wav",
  "duration": 5.234,
  "text": "Γεια σου κόσμε",
  "source_lang": "el",
  "target_lang": "el",
  "pnc": "yes",
  "taskname": "asr"
}
```

Optional v2 fields: `decodercontext`, `emotion`, `itn`, `timestamp`, `diarize`

All unrecognized fields are attached to `cut.custom` and available for filtering/routing.

## Tarred Dataset Format

For large-scale training, audio is packed into tar shards:

```yaml
# input_cfg.yaml — root-level list (no wrapping key!)
- corpus: greek_podcasts
  language: el
  manifest_filepath: "/path/tarred_audio_manifest.json"
  tarred_audio_filepaths: "/path/audio_{0..15}.tar"
  type: nemo_tarred
  weight: 2050
```

Create tarred datasets with:
```bash
python scripts/speech_recognition/convert_to_tarred_audio_dataset.py \
    --manifest_path manifest.json \
    --target_dir ./tarred_output \
    --num_shards 16
```

## Batch Assembly: PromptedAudioToTextMiniBatch

**File**: `nemo/collections/asr/data/audio_to_text_lhotse_prompted.py:29-48`

| Field | Shape | Description |
|-------|-------|-------------|
| audio | (B, max_audio_len) | Raw waveforms |
| audio_lens | (B,) | Actual audio lengths |
| prompt | (B, max_prompt_len) | Prompt tokens only |
| prompt_lens | (B,) | Prompt token lengths |
| transcript | (B, max_text_len) | Answer text tokens only |
| transcript_lens | (B,) | Answer token lengths |
| prompted_transcript | (B, max_seq_len) | Full: prompt + answer |
| prompted_transcript_lens | (B,) | Full sequence lengths |

`get_decoder_inputs_outputs()` returns:
- **input**: `prompted_transcript[:, :-1]` (everything except last token)
- **output**: `prompted_transcript[:, 1:]` (everything except first token — shifted right)

## Dynamic Batching (Lhotse)

### Key Parameters

```yaml
train_ds:
  batch_duration: 2200       # max total audio seconds per batch
  quadratic_duration: 30     # penalty for long sequences
  use_bucketing: true        # group similar-length audio
  num_buckets: 20            # number of duration bins
  bucket_buffer_size: 20000  # cuts held in buffer for bucketing
  shuffle_buffer_size: 10000 # shuffle buffer after bucketing
  max_duration: 40.0         # filter: discard audio > 40s
  min_duration: 0.01         # filter: discard audio < 10ms
```

### How batch_duration Works

Each batch is filled until total audio duration reaches the limit:
- Short utterances → large batch count (e.g., 200 x 10s = 2000s)
- Long utterances → small batch count (e.g., 50 x 40s = 2000s)

### quadratic_duration

Addresses quadratic memory in attention: penalizes long sequences so the sampler avoids packing too many long utterances together. Prevents OOM from batches with many long audio files.

### Bucketing

1. Pre-sort cuts by duration into `num_buckets` bins
2. Sample batches from same bucket → minimal padding waste
3. Shuffle across buckets via `shuffle_buffer_size`
4. **Disabled for validation** (`use_bucketing: false`)

## Audio Preprocessing

**File**: `nemo/collections/asr/modules/audio_preprocessing.py`

```yaml
preprocessor:
  _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
  sample_rate: 16000
  normalize: per_feature       # zero-mean, unit-variance per mel bin
  window_size: 0.025           # 25 ms
  window_stride: 0.01          # 10 ms → 100 frames/sec
  window: hann
  features: 128                # mel bins
  n_fft: 512
  log: true                    # log-mel
  frame_splicing: 1            # no splicing
  dither: 1e-05                # small noise for numerical stability
  pad_to: 0
```

Output: `(B, 128, T)` where `T = audio_duration / 0.01`

After 8x subsampling in the encoder: `T_enc = T / 8`

## Data Augmentation: SpecAugment

**File**: `nemo/collections/asr/parts/submodules/spectr_augment.py`

```yaml
spec_augment:
  _target_: nemo.collections.asr.modules.SpectrogramAugmentation
  freq_masks: 2          # number of frequency masks
  time_masks: 10         # number of time masks
  freq_width: 27         # max frequency mask width (of 128 bins)
  time_width: 0.05       # max time mask width (5% of duration)
```

Applied **only during training** (not validation/inference). Zeros out random rectangular regions of the mel spectrogram to improve generalization.

## Multi-Source Training (input_cfg.yaml)

For multi-language or multi-corpus training:

```yaml
- corpus: greek_podcasts
  language: el
  manifest_filepath: "/path/el_manifest.json"
  tarred_audio_filepaths: "/path/el_audio_{0..15}.tar"
  type: nemo_tarred
  weight: 2000

- corpus: english_librispeech
  language: en
  manifest_filepath: "/path/en_manifest.json"
  tarred_audio_filepaths: "/path/en_audio_{0..31}.tar"
  type: nemo_tarred
  weight: 960
```

Lhotse `mux()` samples **per-utterance** based on weights (NOT per-batch), so batches naturally contain a mix of languages.

Use **temperature sampling** to control the balance:
- τ=1.0: proportional to raw weights
- τ=0.5: flattens distribution (upweights minority languages)
- τ=0: uniform sampling regardless of dataset size

## Key Source Files

| File | Contents |
|------|----------|
| `nemo/collections/asr/data/audio_to_text_lhotse_prompted.py` | PromptedAudioToTextLhotseDataset |
| `nemo/collections/common/data/lhotse/nemo_adapters.py` | LazyNeMoIterator, tarred loading |
| `nemo/collections/common/data/lhotse/dataloader.py` | LhotseDataLoadingConfig, sampler setup |
| `nemo/collections/asr/modules/audio_preprocessing.py` | Mel-spectrogram preprocessor |
| `nemo/collections/asr/parts/submodules/spectr_augment.py` | SpecAugment |
| `scripts/speech_recognition/convert_to_tarred_audio_dataset.py` | Tarred dataset creation |
