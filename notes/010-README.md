# Project Notes Index

## High-level understanding
- [Architecture](020-architecture.md) — ASR model families, tokenizers, prompt system
- [Decisions](030-decisions.md) — Key design choices and trade-offs

## Investigations
- [Tokenizer Deep Dive](040-investigations/tokenizer-deep-dive.md)
- [Canary Fine-tuning Guide](040-investigations/canary-finetuning.md)
- [Training Config Tuning](040-investigations/training-config-tuning.md)
- [Data Quality & Filtering](040-investigations/data-quality.md)
- [Common Errors](040-investigations/common-errors.md)

## Open questions
- Best practices for multi-task data mixing ratios during fine-tuning
- Optimal `batch_duration` for B200 (192GB) — currently using 2200, might go higher

## Next actions
- Monitor val_loss for overfitting during 50k-step training run
- Evaluate final model on 30-hour test set
- Consider tightening data quality filters if WER is disappointing
