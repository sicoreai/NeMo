# Data Quality & Filtering

## Data Pipeline Overview

Two scripts handle data preparation:

1. **`experiments/scripts/export_asr_records.py`** — Finds bad records and tags them
2. **`experiments/scripts/export_train_eval.py`** — Splits data into train/eval/test

## Quality Thresholds

### export_asr_records.py (tagging "bad" — OR logic)

Records matching ANY filter are exported and tagged:

| Metric | Threshold | Catches |
|--------|-----------|---------|
| `char_rate` | ≥ 30 chars/sec | Garbage transcripts |
| `text_len` | ≥ 900 chars | Abnormally long text |
| `max_word_len` | ≥ 25 chars | Recognition errors |
| `top_word_count` | ≥ 15 repeats | Hallucinated repetitions |

### export_train_eval.py (eval/test quality — AND logic)

Groups must pass ALL checks to qualify for eval/test:

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| `char_rate` | 2-25 chars/sec | Normal speech rate |
| `max_word_len` | ≤ 20 chars | Clean words |
| `top_word_count` | ≤ 10 repeats | No hallucination |

Groups that fail → sent to **train**.

## The Gap Between Thresholds

Records in these ranges are NOT tagged "bad" but ARE excluded from eval/test:

| Metric | Gap range | Goes to train |
|--------|-----------|---------------|
| `char_rate` | 25-30 | Yes |
| `max_word_len` | 20-25 | Yes |
| `top_word_count` | 10-15 | Yes |

These borderline records contribute to high/noisy training loss.

## Exclusion Tags

Both scripts exclude records tagged with:
- `music` — music segments
- `bad` — identified bad records

`export_train_eval.py` excludes from ALL splits (train, eval, test).

## Impact on Training

With dirty data in training:
- **Train loss**: Higher and noisier (~0.5 smoothed)
- **Val loss**: Lower and steady (0.13 → 0.09) because eval data is clean
- **This is normal**: The gap is explained by data quality + regularization

## When to Tighten Filters

Don't pre-optimize. Current approach:
1. Train with current data
2. Evaluate final WER on test set
3. If WER disappointing → tighten "bad" thresholds to match eval thresholds:

```python
# In export_asr_records.py — match eval quality thresholds
CHAR_RATE_MIN = 25.0     # was 30.0
MAX_WORD_LEN_MIN = 20    # was 25
TOP_WORD_COUNT_MIN = 10  # was 15
```

Then re-tag, re-export, retrain.

**Note**: Some noise in training data helps robustness — the model learns to
handle imperfect audio. Aggressive cleaning can hurt generalization.

## Group-Based Splitting

`export_train_eval.py` groups records by original source file name, ensuring
all segments from the same audio file stay in the same split. This prevents
data leakage between train and eval.

```python
# Example: all segments from one podcast episode stay together
"20201210-14-7f6b1d76-e298-4bd4-aafd-1b13d23efd88__573__2613.wav"
→ group key: "20201210-14-7f6b1d76-e298-4bd4-aafd-1b13d23efd88"
```

## Split Configuration

Current settings (from `export_train_eval.py`):
- **Train**: ~2050 hours (everything else, including borderline quality)
- **Eval**: 20 hours (quality-filtered, clean data) — used for validation during training
- **Test**: 30 hours (quality-filtered, clean data) — used for final evaluation
- **Seed**: 42 (reproducible split)
- **Total**: ~2100 hours
