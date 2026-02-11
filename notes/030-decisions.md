# Decisions

## Fine-tuning: `speech_to_text_finetune.py` vs `speech_to_text_aed.py`

| | `speech_to_text_finetune.py` | `speech_to_text_aed.py` |
|---|---|---|
| Purpose | Adapt existing model to new data/tokenizer | Train/fine-tune Canary multitask |
| Model types | Any ASRModel (CTC, RNNT) | EncDecMultiTaskModel only |
| Architecture changes | No | Yes (defined in config) |
| Pretrained model | Required | Optional |
| Tokenizer | Simple swap (single BPE) | Aggregate or unified |
| Tasks | Single (ASR) | Multiple (ASR, translation, etc.) |

**Decision**: For Canary fine-tuning, use `speech_to_text_aed.py`.

## Config Construction for Fine-tuning

**Approach A**: Copy entire pretrained `_cfg`, override data/optim/tokenizer paths.
- Pros: Guarantees architecture match
- Cons: Must fix paths that point inside .nemo archive

**Approach B**: Start from default YAML, copy architecture fields selectively.
- Pros: Cleaner config, only set what you need
- Cons: Risk missing a field

**Decision**: Approach A is safer. Copy full config, then override:
- `train_ds.manifest_filepath`
- `validation_ds.manifest_filepath`
- `tokenizer.langs.*.dir` (fix paths)
- `optim` (lower LR for fine-tuning)
- `train_ds.batch_duration` (fit your GPU)

## Tokenizer: Aggregate vs Unified

- **Aggregate** (canary-1b, flash): Per-language tokenizers concatenated. Config uses `type: agg`.
- **Unified** (canary-1b-v2): Single SentencePiece for all languages. Config uses `type: bpe`.

**Decision**: Match the pretrained model's tokenizer type. Don't mix.

## Placeholder Tokens for Custom Behaviors

Use existing `<|spltoken0|>` through `<|spltoken29|>` rather than rebuilding tokenizer.
- Avoids embedding reinitialization
- Can fine-tune directly from pretrained checkpoints
- Use in pairs (on/off) like `<|pnc|>`/`<|nopnc|>`

## `enable_bn_se` During Fine-tuning

When freezing most of the model, selectively unfreeze BatchNorm layers:
```python
model.apply(enable_bn_se)
```
This allows BatchNorm to adapt to new data statistics (running mean/var) while
keeping other layers frozen. Critical when fine-tuning data distribution differs
from pretraining.

## Loss Masking for Prompts

`use_loss_mask_for_prompt: false` vs `true`:
- Masking helped canary-180m but not canary-1b (from tutorial tips)
- Try both when fine-tuning; depends on model size and data

## Validation Strategy for Large Datasets

**Decision**: Use small validation set during training, full test set for final eval.

- 20-hour eval set used for validation during training (stable WER/loss metrics)
- 30-hour test set used for final evaluation after training
- Use `trainer.limit_val_batches` to cap validation time if needed
- Validation set must be **representative**, not proportional to training set

## `check_val_every_n_epoch` Must Be Null with Lhotse

**Decision**: Always set `check_val_every_n_epoch: null` when using Lhotse.

Lhotse's infinite iterator (`CutSet.repeat()`) means Lightning never detects epoch
boundaries. With `check_val_every_n_epoch: 1`, validation never triggers.
Setting to `null` makes validation purely step-based via `val_check_interval`.

## Data Quality: Don't Over-Filter Training Data

**Decision**: Keep borderline-quality data in training; only remove clearly bad records.

Some noise in training helps robustness. Current approach:
- Tag clearly bad records (char_rate ≥ 30, word_len ≥ 25, repetitions ≥ 15)
- Route borderline records to train (eval uses tighter thresholds)
- Re-evaluate after seeing final WER on test set

See [Data Quality](040-investigations/data-quality.md) for threshold details.
