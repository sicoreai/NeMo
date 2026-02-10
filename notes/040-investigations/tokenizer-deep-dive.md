# Tokenizer Deep Dive

## SentencePieceTokenizer

### Legacy vs Non-Legacy Mode
- `legacy=False` (default): Special tokens must be in the model at train time
- `legacy=True`: Can add special tokens at runtime, manual parsing logic

### Key Properties
- `vocab_size`: Total vocabulary including added special tokens
- `original_vocab_size`: Base SPE model vocab (before runtime additions)
- `removed_extra_spaces`: Whether tokenizer collapses multiple spaces
- `space_sensitive`: Whether tokenization depends on surrounding context

### Whitespace Handling
Uses `☯` as internal marker to preserve extra spaces when `ignore_extra_whitespaces=False`.

### SPM Separator Trimming
`trim_spm_separator_after_special_token=True` removes spurious `▁` after special tokens:
```
"[INST] hello" → ["[INST]", "▁", "hello"] → trimmed → ["[INST]", "hello"]
```

## SentencePiece Training Parameters

### `user_defined_symbols` vs `control_symbols`
| | Encoded from text? | In decoded output? |
|---|---|---|
| `user_defined_symbols` | Yes | Yes |
| `control_symbols` | No (insert programmatically) | No (stripped) |

All Canary special tokens use `user_defined_symbols`.

### `split_by_unicode_script`
- `True` (default): Different scripts (Latin, Greek, Arabic) never merge into one subword
- `False`: Cross-script merges allowed (useful for Arabic diacritics)

### Key Training Parameters
- `character_coverage`: 1.0 for most languages, < 1.0 for CJK
- `byte_fallback`: Use byte sequences for unknown characters
- `split_digits`: Split digits into individual tokens
- `remove_extra_whitespaces`: Skip double spaces during encoding

## CanaryTokenizer (Aggregate)

### Structure
Combines multiple sub-tokenizers:
```yaml
tokenizer:
  type: agg
  langs:
    spl_tokens:       # Special tokens (task, lang, pnc, etc.)
      type: bpe
    en:               # English
      type: bpe
    es:               # Spanish
      type: bpe
```

### Building Special Tokenizer
```python
CanaryTokenizer.build_special_tokenizer(
    tokens=["translate", "transcribe", "en", "es", ...],
    model_dir="path/to/spl_tokens",
    force_rebuild=True,
)
```

### Placeholder Tokens
`<|spltoken0|>` through `<|spltoken29|>` — 30 tokens padding vocab to multiple of 64.
Free to use for custom behaviors without rebuilding tokenizer.

## CanaryBPETokenizer (Unified)
Used by canary-1b-v2. Single SentencePiece model for all languages.
```yaml
tokenizer:
  dir: /path/to/tokenizer
  type: bpe
  custom_tokenizer:
    _target_: nemo.collections.common.tokenizers.canary_tokenizer.CanaryBPETokenizer
```

## Prompt Format Functions

### Registry Pattern (`prompt_fn.py`)
```python
@registered_prompt_format_fn(Cut, Canary2PromptFormatter)
def canary2(cut, prompt):
    # Returns {"context_ids": [...], "answer_ids": [...]}
```

Lookup via MRO: tries exact (DataType, FormatterType) match, then parent classes,
then default for DataType alone.

### PromptFormatter Base (`formatter.py`)
- Template with `|slot|` placeholders replaced at runtime
- Training: last turn is OUTPUT_ROLE → returns context_ids + answer_ids + mask
- Inference: last turn is NOT OUTPUT_ROLE → returns context_ids only
- Auto-registration via `__init_subclass__` + `NAME` attribute
- Special slots: `|bos|`, `|eos|` for SentencePiece BOS/EOS insertion
