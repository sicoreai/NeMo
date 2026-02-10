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

```yaml
model:
  use_loss_mask_for_prompt: false    # Try both
  optim:
    name: adamw
    lr: 1e-4                         # Lower than from-scratch (3e-4)
    sched:
      name: InverseSquareRootAnnealing
      warmup_steps: 1000

trainer:
  precision: bf16-mixed              # Required for 1B models
  devices: -1
  max_steps: 10000
  val_check_interval: 1000

exp_manager:
  checkpoint_callback_params:
    monitor: "val_loss"
    save_top_k: 3
    always_save_nemo: True
```
