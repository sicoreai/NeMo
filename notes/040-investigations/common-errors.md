# Common Errors

## CUDA device-side assert: Embedding index out of bounds

```
vectorized_gather_kernel: Assertion `ind >=0 && ind < ind_dim_size` failed
```

**Cause**: Token IDs exceed embedding table size. Two variants:

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

## CUDA_LAUNCH_BLOCKING for better error messages

CUDA errors are asynchronous — reported stacktrace may be wrong.
Always re-run with:
```bash
CUDA_LAUNCH_BLOCKING=1 python examples/asr/speech_multitask/speech_to_text_aed.py ...
```

**Performance impact**: 2-5x slower. Forces synchronous CUDA execution (GPU
completes each kernel before CPU launches the next). Only use for debugging,
then remove for actual training.

## `.nemo` tokenizer paths after restore

After `restore_from()` or `from_pretrained()`, tokenizer paths point inside the
extracted `.nemo` archive (temp directory). Use `save_tokenizers()` to extract
to a stable location before building fine-tuning configs.
