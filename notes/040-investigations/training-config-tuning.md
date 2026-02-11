# Training Config Tuning

Lessons learned from configuring Canary fine-tuning on 4x B200 (192GB each)
with ~2000 hours of training data.

Config: `experiments/canary_2026_0210/canary-custom-finetune.yaml`

## Validation Trigger: `check_val_every_n_epoch`

**Problem**: Lhotse creates infinite iterable datasets via `CutSet.repeat()`.
PyTorch Lightning sees this as one never-ending epoch. With
`check_val_every_n_epoch: 1`, validation never triggers because the epoch
never completes.

**Fix**: Set `check_val_every_n_epoch: null` so validation is purely step-based
via `val_check_interval`.

```yaml
trainer:
  val_check_interval: 400        # every 400 steps
  check_val_every_n_epoch: null  # MUST be null with Lhotse infinite iteration
```

NeMo audio configs (e.g., `predictive_conformer.yaml`) already use
`check_val_every_n_epoch: null`.

## Choosing `val_check_interval`

With 4x B200 and `batch_duration: 2200`:
- Effective batch per step: 2200 × 4 = 8800 sec
- Steps per epoch: 7,200,000 / 8800 ≈ 818
- Target: ~2 validations per epoch → `val_check_interval: 400`

## Validation Batch Size

Validation is forward-pass only (no gradients, no optimizer states), so it
can use larger batches than training. Use `multiply` OmegaConf resolver:

```yaml
validation_ds:
  batch_size: null                                            # disable fixed batch size
  batch_duration: ${multiply:${model.train_ds.batch_duration},3}  # 3x training
  quadratic_duration: ${model.train_ds.quadratic_duration}
```

Multiply guidelines:
- 3x: safe
- 5x: risky, may OOM
- Start conservative and increase

## Limiting Validation Data: `limit_val_batches`

With large validation sets, each validation run can take many minutes
(dominated by autoregressive beam search decoding for WER).

Current split: 20 hours eval, 30 hours test. The 20-hour eval set is
reasonable for validation during training. If validation is still slow,
use Lightning's `limit_val_batches` to cap validation steps:

```yaml
trainer:
  limit_val_batches: 5   # only 5 validation batches per check
```

## Validation Set Size

A 9:1 train/val ratio is for splitting a single dataset. For monitoring
during training, validation just needs to be **representative** (stable WER):

| Val size  | Metric stability | Recommendation        |
|-----------|------------------|-----------------------|
| 1 hour    | Noisy            | Too small             |
| 10-20 hrs | Stable           | Good for during training |
| 50 hrs    | Very stable      | Acceptable            |
| 200 hrs   | Marginal gain    | Use for final eval only |

## `batch_duration` Sizing

For 4x B200 (192GB each) with canary-1b (~1B params, bf16):
- Model + optimizer + gradients: ~12 GB
- Remaining ~180 GB per GPU for activations
- `batch_duration: 2200` works with `quadratic_duration: 30`
- `quadratic_duration` penalizes long utterances to limit peak memory

## `max_steps` for Fine-tuning

For 2000 hours, steps per epoch ≈ 818 (4 GPUs, batch_duration 2200).
Fine-tuning typically needs 1-5 epochs. 50k steps ≈ 61 epochs — on the high
side, but `save_top_k: 5` with `monitor: val_loss` keeps the best checkpoints.

Watch val_loss: if it starts increasing, training is overfitting.

## LR Schedule

```yaml
optim:
  lr: 5.0e-05
  sched:
    name: InverseSquareRootAnnealing
    max_steps: ${trainer.max_steps}    # use Hydra interpolation to stay in sync
    warmup_steps: 2500
    min_lr: 1.0e-06
```

Key: `sched.max_steps` must match `trainer.max_steps`. Use `${trainer.max_steps}`
to avoid mismatch. A mismatch causes LR to hit `min_lr` early and waste compute.

## Gradient Clipping

```yaml
trainer:
  gradient_clip_val: 1.0   # prevent gradient spikes during fine-tuning
```

Without clipping (`0.0`), a single bad batch can destabilize the model.

## Progress Bar Interpretation

```
Epoch 0: | | 278/? [04:45<00:00, 0.97it/s, v_num=1, train_step_timing in s=1.080]
```

| Part | Meaning |
|------|---------|
| `Epoch N:` | Epoch counter — may increment if Lhotse uses finite epochs (single manifest) or stay at 0 (infinite mux) |
| `278/?` | Steps completed / total unknown (iterable dataset) |
| `0.97it/s` | ~1 step per second (overall wall time including data loading) |
| `train_step_timing in s=1.080` | GPU compute time per step only |

### Epoch behavior with Lhotse

- **Single manifest** (`manifest_filepath`): Lhotse creates finite epochs.
  Epoch counter increments, step counter resets each epoch. Normal.
- **Multiple datasets** (`input_cfg` with `CutSet.mux/repeat`): One infinite
  epoch. Epoch stays at 0.

Both are fine with `check_val_every_n_epoch: null` — validation triggers
every N steps regardless.

### Detecting data loading bottleneck

Compare `train_step_timing` (GPU compute) vs overall `it/s` (wall time):

```
# Healthy: GPU compute ≈ wall time
0.97it/s → 1.03s/it,  train_step_timing=1.080s  → no bottleneck

# Bottleneck: GPU compute << wall time
0.63it/s → 1.58s/it,  train_step_timing=0.266s  → 80% idle on data loading
```

If gap is large and persists:
- Increase `num_workers` (8 → 16)
- Check I/O (local NVMe vs network storage)
- Increase `shuffle_buffer_size` and `bucket_buffer_size`
- May also be transient after epoch transitions or validation runs

## `log_every_n_steps`

Log 10-20x more frequently than validation for good trend visibility.
With `val_check_interval: 400`, use `log_every_n_steps: 100` (20 points
between validations).

## Training Loss Variance

Training loss varies a lot (e.g., 0.2 to 0.5) with Lhotse dynamic batching.
This is normal — each batch has different utterance lengths and compositions.
Use TensorBoard smoothing (0.9+) to see the trend.

**Train loss higher than val loss is expected** due to:
1. Noisy/dirty data in training set (quality-failed groups sent to train)
2. SpecAugment, dropout, dither active during training only
3. Dynamic batch composition

The metric that matters: **val loss trending down**.

## `nvidia-smi` Monitoring

| Field | Meaning |
|-------|---------|
| Pwr:Usage/Cap | Current power draw / max allowed (watts) |
| GPU-Util | % time SMs were active (0-100%) |

Healthy training: GPU-Util 90-100%, power near cap.
If GPU-Util < 80%: increase `num_workers` or use tarred data.

## Temperature Sampling for Data Weighting

Canary uses temperature sampling (τ=0.5) to balance languages/corpora:

```
p_i = w_i^τ / Σ(w_j^τ)
```

τ=0.5 takes the square root of dataset sizes before normalizing, flattening
the distribution so smaller datasets get over-sampled.

Compute weights with:
```bash
python scripts/speech_recognition/estimate_data_weights.py \
    input_cfg.yaml output_cfg.yaml -t 0.5 -s num_hours
```

Implementation: `scripts/speech_recognition/estimate_data_weights.py:146-149`
