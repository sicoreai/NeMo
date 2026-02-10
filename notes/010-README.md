# Project Notes Index

## High-level understanding
- [Architecture](020-architecture.md) — ASR model families, tokenizers, prompt system
- [Decisions](030-decisions.md) — Key design choices and trade-offs

## Investigations
- [Tokenizer Deep Dive](040-investigations/tokenizer-deep-dive.md)
- [Canary Fine-tuning Guide](040-investigations/canary-finetuning.md)
- [Common Errors](040-investigations/common-errors.md)

## Open questions
- How does Lhotse dynamic batching interact with multi-GPU sharding?
- Best practices for multi-task data mixing ratios during fine-tuning

## Next actions
- Benchmark canary-1b-v2 on custom dataset
- Test tarred dataset pipeline end-to-end
