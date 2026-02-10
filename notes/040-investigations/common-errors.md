# Common Errors

## 1. TokenPerSecondFilter: `'<=' not supported between instances of 'int' and 'NoneType'`

```
TypeError: '<=' not supported between instances of 'int' and 'NoneType'
```

**Cause**: `max_tps: null` in config. Bug in `TokenPerSecondFilter.__init__` —
raw parameter compared before `ifnone` conversion.

**Fix**: Remove `max_tps` from config or set explicitly:
```yaml
model.train_ds.max_tps: .inf
```

## 2. CUDA device-side assert: Embedding index out of bounds

```
vectorized_gather_kernel: Assertion `ind >=0 && ind < ind_dim_size` failed
```

**Cause**: Token IDs exceed embedding table size. Two variants:

### Variant A: Token embedding OOB
Tokenizer vocab_size > `transf_decoder.config_dict.vocab_size`.

**Fix**: Set `vocab_size: None` so it's auto-resolved from tokenizer:
```yaml
transf_decoder.config_dict.vocab_size: None
head.num_classes: None
```

**Debug**:
```python
print("Tokenizer vocab:", model.tokenizer.vocab_size)
print("Embedding size:", model.transf_decoder._embedding.token_embedding.num_embeddings)
```

### Variant B: Positional embedding OOB
Decoder sequence length exceeds `max_sequence_length`.

Error with `CUDA_LAUNCH_BLOCKING=1`:
```
transformer_modules.py:60
    embeddings = torch.embedding(self.pos_enc, position_ids)
```

**Cause**: Manifest has samples where `[prompt tokens] + [transcript tokens] + [EOS]`
exceeds `max_sequence_length` (default 512). Works with one manifest but fails with
another that has longer transcripts.

**Fix**: Increase `max_sequence_length` or filter data:
```yaml
transf_decoder.config_dict.max_sequence_length: 1024
# or
model.train_ds.max_duration: 30.0
```

**Debug**: Check longest transcript in manifest:
```python
import json
for line in open("manifest.json"):
    entry = json.loads(line)
    tokens = tokenizer.text_to_ids(entry["text"])
    total = len(tokens) + 10  # ~10 prompt tokens
    if total > max_sequence_length:
        print(f"TOO LONG: {total} tokens, duration={entry['duration']}s")
```

## 3. CUDA_LAUNCH_BLOCKING for better error messages

CUDA errors are asynchronous — reported stacktrace may be wrong.
Always re-run with:
```bash
CUDA_LAUNCH_BLOCKING=1 python examples/asr/speech_multitask/speech_to_text_aed.py ...
```

## 4. Multi-GPU model download race condition

When using `from_pretrained()` with multiple GPUs, rank 0 downloads while others wait.
The fine-tuning script handles this with sleep. The AED script avoids it by constructing
from config instead of downloading.

## 5. `.nemo` tokenizer paths after restore

After `restore_from()` or `from_pretrained()`, tokenizer paths point inside the
extracted `.nemo` archive (temp directory). Use `save_tokenizers()` to extract
to a stable location before building fine-tuning configs.
