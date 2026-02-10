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
