# Canary 1B v2 — Training

## Training Step

**File**: `nemo/collections/asr/models/aed_multitask_models.py:793-850`

```python
def training_step(self, batch: PromptedAudioToTextMiniBatch, batch_nb):
    # 1. Prepare decoder inputs/outputs
    input_ids, labels = batch.get_decoder_inputs_outputs()

    # 2. Forward pass (encoder + decoder + head)
    transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
        input_signal=batch.audio,
        input_signal_length=batch.audio_lens,
        transcript=input_ids,
        transcript_length=input_ids_lens,
    )

    # 3. Build loss mask (mask padding, optionally mask prompt tokens)
    loss_mask = ...  # 1 where loss should apply, 0 elsewhere

    # 4. Compute loss
    transf_loss = self.loss(log_probs=transf_log_probs, labels=labels, output_mask=loss_mask)

    # 5. Compute WER metrics (every log_every_n_steps)
    #    WARNING: calls torchmetrics.sync() which triggers NCCL allreduce
    metric_dict = self.metric.eval(batch=batch, predictions=enc_states, ...)
```

## Forward Pass

**File**: `aed_multitask_models.py:726-790`

```
1. Preprocessor:  audio → mel-spectrogram (B, 128, T)
2. SpecAugment:   random masking (training only)
3. Encoder:       FastConformer → encoded (B, T/8, 1024)
4. Projection:    Identity (1024 → 1024)
5. [Optional]:    Transformer encoder (disabled by default)
6. Decoder:       Transformer decoder with cross-attention → (B, S, 1024)
7. Head:          Token classifier → log_probs (B, S, V)
```

## Loss Function

**File**: `nemo/collections/common/losses/smoothed_cross_entropy.py`

```yaml
loss:
  _target_: nemo.collections.common.losses.smoothed_cross_entropy.SmoothedCrossEntropyLoss
  label_smoothing: 0.0      # default: no smoothing
  pad_id: <tokenizer.pad_id>  # set at runtime
```

Standard cross-entropy on the decoder output logits vs. target token IDs, with optional label smoothing. Padding positions are masked out.

### Loss Masking (aed_multitask_models.py:815-819)

Two modes controlled by `use_loss_mask_for_prompt`:
- **false** (default): Loss computed on ALL decoder tokens (prompt + text)
- **true**: Loss computed ONLY on text tokens (prompt tokens masked out)

## Metrics

**File**: `nemo/collections/asr/metrics/multitask.py`

### MultiTaskMetric

Conditionally computes different metrics based on the task:

```yaml
multitask_metrics_cfg:
  metrics:
    wer:
      _target_: nemo.collections.asr.metrics.WER
      constraint: ".source_lang==.target_lang"     # ASR: same language
    bleu:
      _target_: nemo.collections.asr.metrics.BLEU
      constraint: ".source_lang!=.target_lang"     # Translation: different languages
```

### WER Computation Gotcha

WER is computed via `self.metric.eval()` which calls `torchmetrics.sync()` → NCCL allreduce → `gather_all_tensors` → `barrier`. On multi-GPU, if one rank has significantly more data, other ranks can timeout waiting at the barrier.

**Mitigation**: Set `log_every_n_steps` to a high value (e.g., 1000) to reduce metric sync frequency. See `common-errors.md` for NCCL timeout details.

## Optimizer & Scheduler

```yaml
optim:
  name: adamw
  lr: 3e-4
  betas: [0.9, 0.98]
  weight_decay: 1e-3

  sched:
    name: InverseSquareRootAnnealing
    warmup_steps: 2500
    warmup_ratio: null
    min_lr: 1e-6
```

### InverseSquareRootAnnealing

```
LR = base_lr * min(1, step / warmup_steps) * (warmup_steps / max(step, warmup_steps))^0.5
```

- Linearly warms up for `warmup_steps`
- Then decays as `1/sqrt(step)`
- Never goes below `min_lr`

**Important**: Set `sched.max_steps: ${trainer.max_steps}` to keep them in sync.

## Validation

```yaml
trainer:
  val_check_interval: 400         # validate every 400 steps
  check_val_every_n_epoch: null   # REQUIRED with Lhotse (infinite iterator)
  limit_val_batches: 1.0          # use all val data (or set int to cap steps)
```

### Why check_val_every_n_epoch Must Be null

Lhotse uses `CutSet.repeat()` for infinite iteration. Lightning never detects epoch boundaries, so `check_val_every_n_epoch: 1` means validation NEVER triggers. Set to `null` and rely on `val_check_interval` (step-based).

## Checkpointing

```yaml
exp_manager:
  checkpoint_callback_params:
    monitor: val_loss         # or val_wer
    mode: min
    save_top_k: 3
    always_save_nemo: true    # saves best .nemo (overwrites on improvement)
```

- `save_top_k`: Keeps N best `.ckpt` files
- `always_save_nemo: true`: Also saves a single best `.nemo` file (convenient but doubles I/O per checkpoint)
- To convert `.ckpt` to `.nemo`: `EncDecMultiTaskModel.load_from_checkpoint("path.ckpt").save_to("model.nemo")`

## Multi-GPU Training

```yaml
trainer:
  devices: 4           # number of GPUs
  strategy: ddp         # DistributedDataParallel
  precision: bf16-mixed  # or 16-mixed

  # Gradient accumulation
  accumulate_grad_batches: 1  # increase for effective larger batch
```

Lhotse handles distributed sampling internally (`use_distributed_sampler: false`).

## Fine-tuning Tips

| Setting | Recommendation |
|---------|---------------|
| Learning rate | 1e-4 to 3e-4 (lower for fine-tuning) |
| Warmup steps | 500–2500 |
| Max steps | 3000–5000 for 2000h data |
| gradient_clip_val | 1.0 (prevents exploding gradients) |
| batch_duration | 2200 on B200 (192GB), lower on smaller GPUs |
| Validation batch | 3-4x training batch_duration (no backward pass) |

## Key Source Files

| File | Contents |
|------|----------|
| `nemo/collections/asr/models/aed_multitask_models.py` | training_step, forward, validation_step |
| `nemo/collections/common/losses/smoothed_cross_entropy.py` | Loss function |
| `nemo/collections/asr/metrics/multitask.py` | MultiTaskMetric, WER, BLEU |
| `examples/asr/conf/speech_multitask/fast-conformer_aed.yaml` | Reference training config |
| `examples/asr/speech_multitask/speech_to_text_aed.py` | Training entry point |
