# Canary Fine-tuning Guide

## Overview

Use `examples/asr/speech_multitask/speech_to_text_aed.py` with
`examples/asr/conf/speech_multitask/fast-conformer_aed.yaml` as base config.

## Step-by-Step

### 1. Extract Config and Tokenizer from Pretrained

```python
from nemo.collections.asr.models import EncDecMultiTaskModel
from omegaconf import OmegaConf

canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b-v2')
canary_model.save_tokenizers('./canary_v2_tokenizers/')
```

### 2. Build Config (Approach A — Full Copy)

```python
base_model_cfg = OmegaConf.load("fast-conformer_aed.yaml")
base_model_cfg['model'] = canary_model._cfg

# Override paths
base_model_cfg['model']['train_ds']['manifest_filepath'] = "/data/train.json"
base_model_cfg['model']['validation_ds']['manifest_filepath'] = "/data/val.json"

# Fix tokenizer paths (point to extracted files, not .nemo internals)
# For aggregate tokenizer:
base_model_cfg['model']['tokenizer']['langs']['spl_tokens']['dir'] = "./canary_v2_tokenizers/spl_tokens"
base_model_cfg['model']['tokenizer']['langs']['en']['dir'] = "./canary_v2_tokenizers/en"

# Override optimizer for fine-tuning
base_model_cfg['model']['optim']['lr'] = 1e-4
base_model_cfg['model']['optim']['sched']['warmup_steps'] = 1000

# Fit to your GPU
base_model_cfg['model']['train_ds']['batch_duration'] = 120
```

### 3. Prepare Manifest

```json
{"audio_filepath": "/path/to/audio.wav", "text": "transcription", "duration": 1.5,
 "source_lang": "en", "target_lang": "en", "pnc": "yes"}
```

Required fields: `audio_filepath`, `text`, `duration`, `source_lang`, `target_lang`
Optional: `pnc` (default from prompt_defaults)

### 4. Launch Training

```bash
python examples/asr/speech_multitask/speech_to_text_aed.py \
    --config-path="../config" \
    --config-name="my-canary-finetune" \
    trainer.devices=1 \
    trainer.max_steps=10000 \
    exp_manager.exp_dir="results"
```

## Important Settings

### Architecture Compatibility
`vocab_size` and `num_classes` should be `None` (auto-set from tokenizer):
```yaml
transf_decoder.config_dict.vocab_size: None
head.num_classes: None
```

### max_sequence_length
Must be large enough for: `[prompt tokens] + [transcript tokens] + [EOS]`.
Default 512. Increase if you have long transcripts:
```yaml
transf_decoder.config_dict.max_sequence_length: 1024
```

### Selective Weight Initialization
```yaml
init_from_pretrained_model:
  model0:
    name: "nvidia/canary-1b-v2"
    exclude: ["transf_decoder._embedding.token_embedding", "log_softmax.mlp.layer0"]
```

### Data Mix
Include all task types you want the model to retain. Fine-tuning on only English ASR
may cause the model to forget translation capabilities.

## Training Config Recommendations

See [Training Config Tuning](training-config-tuning.md) for detailed tuning notes.

Actual config used: `experiments/canary_2026_0210/canary-custom-finetune.yaml`

```yaml
model:
  optim:
    name: adamw
    lr: 5.0e-05
    weight_decay: 0.001
    sched:
      name: InverseSquareRootAnnealing
      max_steps: ${trainer.max_steps}  # keep in sync
      warmup_steps: 2500
      min_lr: 1.0e-06
  train_ds:
    batch_duration: 2200            # tuned for B200 192GB
    quadratic_duration: 30
    num_workers: 8
  validation_ds:
    batch_duration: ${multiply:${model.train_ds.batch_duration},3}
    shuffle: false

trainer:
  precision: bf16-mixed
  devices: -1
  max_steps: 50000
  val_check_interval: 400
  check_val_every_n_epoch: null     # REQUIRED for Lhotse
  gradient_clip_val: 1.0
  limit_val_batches: 5              # cap validation time

exp_manager:
  checkpoint_callback_params:
    monitor: val_loss
    save_top_k: 5
    always_save_nemo: true
  resume_if_exists: true
  resume_ignore_no_checkpoint: true
```

## test_ds Configuration

Add `test_ds` to the YAML for post-training evaluation:

```yaml
test_ds:
  use_lhotse: true
  prompt_format: canary2              # don't forget this
  manifest_filepath: ???
  sample_rate: ${model.sample_rate}
  batch_duration: ${multiply:${model.train_ds.batch_duration},3}
  quadratic_duration: ${model.train_ds.quadratic_duration}
  max_duration: 40.0
  min_duration: 0.1
  shuffle: false
  num_workers: 8
```

Key: include `prompt_format`, `max_duration`, and `min_duration` — easy to miss.

## Checkpoints: `.ckpt` vs `.nemo`

With `always_save_nemo: true` and `save_top_k: 5`:

- **1 `.nemo` file** — the current best model (lowest `val_loss`), automatically
  overwritten each time a better checkpoint is found
- **5 `.ckpt` files** — top 5 checkpoints by `val_loss`

The `.nemo` file is already your best model. No need to convert `.ckpt` files
unless you want to evaluate a specific intermediate checkpoint:

```python
from nemo.collections.asr.models import EncDecMultiTaskModel
model = EncDecMultiTaskModel.load_from_checkpoint("path/to/specific.ckpt")
model.save_to("specific.nemo")
```

Use cases for converting other `.ckpt` files:
- Suspect overfitting — compare earlier checkpoint vs best
- Model averaging across multiple checkpoints
- Evaluate intermediate checkpoints while training is still running
