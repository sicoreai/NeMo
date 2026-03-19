# Canary 1B v2 Study Notes

Deep-dive study of the NVIDIA Canary 1B v2 ASR model architecture and codebase.

## Contents

1. [Model Architecture](01-model-architecture.md) — Class hierarchy, encoder, decoder, head, parameter distribution
2. [Prompt System](02-prompt-system.md) — Canary v2 prompt format, slots, multi-task control
3. [Data Pipeline](03-data-pipeline.md) — Data loading, batching, augmentation, manifest format
4. [Training](04-training.md) — Training step, loss, metrics, optimizer, scheduler
5. [Inference](05-inference.md) — Transcription API, decoding strategies, streaming, chunked inference
