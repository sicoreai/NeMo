# Canary 1B v2 — Prompt System

## Overview

Canary uses special tokens as a "prompt" prefix to the decoder, controlling model behavior (language, task, features). The decoder sees: `[prompt tokens] [generated text tokens] [EOS]`.

## Canary v2 Prompt Template

**File**: `nemo/collections/common/prompts/canary2.py`

```
User turn:
  <|startofcontext|>{decodercontext}<|startoftranscript|>{emotion}{source_lang}{target_lang}{pnc}{itn}{timestamp}{diarize}

Assistant turn:
  {text}<|endoftext|>
```

### Prompt Slots

| Slot | Type | Values | Default |
|------|------|--------|---------|
| decodercontext | Text | Free-form previous transcript for biasing | `""` (empty) |
| emotion | TextLiteral | `<\|emo:undefined\|>`, `<\|emo:neutral\|>`, `<\|emo:angry\|>`, `<\|emo:happy\|>`, `<\|emo:sad\|>` | `<\|emo:undefined\|>` |
| source_lang | Text | `<\|en\|>`, `<\|el\|>`, `<\|es\|>`, ... | required |
| target_lang | Text | `<\|en\|>`, `<\|el\|>`, `<\|es\|>`, ... | required |
| pnc | TextLiteral | `<\|pnc\|>` / `<\|nopnc\|>` | `<\|pnc\|>` |
| itn | TextLiteral | `<\|itn\|>` / `<\|noitn\|>` | `<\|noitn\|>` |
| timestamp | TextLiteral | `<\|timestamp\|>` / `<\|notimestamp\|>` | `<\|notimestamp\|>` |
| diarize | TextLiteral | `<\|diarize\|>` / `<\|nodiarize\|>` | `<\|nodiarize\|>` |

### Concrete Example

ASR with punctuation for Greek audio:
```
<|startofcontext|><|startoftranscript|><|emo:undefined|><|el|><|el|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>
```

Translation from Greek to English:
```
<|startofcontext|><|startoftranscript|><|emo:undefined|><|el|><|en|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>
```

## Canary v1 vs v2 Prompt Format

| | Canary v1 (`canary.py`) | Canary v2 (`canary2.py`) |
|---|---|---|
| BOS | `<\|startoftranscript\|>` | `<\|startofcontext\|>...<\|startoftranscript\|>` |
| Task slot | Explicit `<\|transcribe\|>`/`<\|translate\|>` | Implicit from source_lang vs target_lang |
| Context | None | `decodercontext` for biasing |
| Emotion | None | `emotion` slot |
| ITN | None | `itn` slot |
| Timestamps | None | `timestamp` slot |
| Diarization | None | `diarize` slot |

## Multi-Task via Prompts

The same model serves multiple tasks by changing prompt tokens:

| Task | source_lang | target_lang | Notes |
|------|-------------|-------------|-------|
| ASR (transcription) | `<\|en\|>` | `<\|en\|>` | same language |
| Translation (AST) | `<\|el\|>` | `<\|en\|>` | different language |
| Language ID | partial prompt only | — | model predicts language |

## Manifest Fields → Prompt Mapping

**File**: `nemo/collections/common/prompts/canary2.py` (map function)

Manifest JSON fields are mapped to special tokens:
```
"source_lang": "en"  →  "<|en|>"
"target_lang": "el"  →  "<|el|>"
"pnc": "yes"         →  "<|pnc|>"       (accepts: "yes", "1", "true", "pnc")
"pnc": "no"          →  "<|nopnc|>"     (accepts: "no", "0", "false", "nopnc")
"taskname": "asr"     →  "<|transcribe|>"  (v1 only, v2 infers from langs)
```

## Decoder Input/Output During Training

```
Decoder input:  [prompt tokens] [text tokens]         ← teacher forcing
Decoder target: [prompt tokens shifted] [text tokens] [EOS]  ← predict next token

Loss mask options:
  - Full: loss on all tokens (prompt + text)
  - Prompt masked: loss only on text tokens (use_loss_mask_for_prompt: true)
```

The `PromptedAudioToTextMiniBatch.get_decoder_inputs_outputs()` method (audio_to_text_lhotse_prompted.py:41-47) creates input/output pairs by stripping first/last tokens.

## Placeholder Tokens for Custom Behaviors

The tokenizer includes 30 unassigned special tokens: `<|spltoken0|>` through `<|spltoken29|>`. These can be repurposed for custom behaviors by:
1. Assigning meaning through fine-tuning data
2. Using pairs (on/off) similar to `<|pnc|>`/`<|nopnc|>`

## Key Source Files

| File | Contents |
|------|----------|
| `nemo/collections/common/prompts/canary2.py` | Canary v2 prompt formatter |
| `nemo/collections/common/prompts/canary.py` | Canary v1 prompt formatter |
| `nemo/collections/common/prompts/formatter.py` | Base PromptFormatter class |
| `nemo/collections/common/data/prompt_fn.py` | Prompt format function registry |
