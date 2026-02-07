# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NVIDIA NeMo is a toolkit for building conversational AI applications, focusing on:
- **ASR** (Automatic Speech Recognition)
- **TTS** (Text-to-Speech)
- **Audio** processing

**Note:** This repo is pivoting to speech models only. Collections like `llm`, `multimodal`, `nlp`, `vision`, `vlm`, `diffusion`, `speechlm` are deprecated.

## Build and Installation

```bash
# Install with all dependencies
pip install '.[all]'

# Install specific domains
pip install '.[asr]'
pip install '.[tts]'

# Install test dependencies
pip install '.[test]'
```

## Running Tests

```bash
# Quick unit tests (no GPU required)
pytest -m "not pleasefixme" --cpu tests/collections/asr

# Full tests with model downloads
pytest -m "not pleasefixme" --with_downloads tests/collections/asr

# Run specific test file
pytest tests/collections/asr/test_asr_models.py
```

Test markers: `unit`, `integration`, `system`, `acceptance`, `pleasefixme` (broken tests), `skipduringci`

Test scripts in `tests/functional_tests/` (e.g., `L0_Unit_Tests_CPU_ASR.sh`) show environment setup for different suites.

## Code Formatting and Linting

```bash
# Check formatting
python setup.py style --scope path/to/file

# Auto-fix formatting
python setup.py style --scope path/to/file --fix
```

Configuration:
- **Black**: line length 119, Python 3.10+ target
- **isort**: black-compatible profile
- **flake8/pylint**: see `.flake8` and `.pylintrc` files (speech vs other configs)

## Architecture

### Directory Structure
```
nemo/
├── collections/          # Domain-specific modules
│   ├── asr/             # Speech recognition (active)
│   ├── tts/             # Text-to-speech (active)
│   ├── audio/           # Audio processing (active)
│   ├── common/          # Shared utilities
│   └── [deprecated]/    # llm, multimodal, nlp, vision, vlm, etc.
├── core/                # Base classes: NeuralModule, Dataset, Loss, Config
├── lightning/           # PyTorch Lightning integration & distributed training
├── export/              # ONNX, TensorRT, vLLM export
└── utils/               # Logging, callbacks, profiling
examples/                # Training/inference scripts by domain
tests/collections/       # Unit tests mirroring collections structure
scripts/                 # Dataset processing utilities
tools/                   # Data preparation, evaluators
```

### Key Technologies
- **PyTorch 2.6+** and **PyTorch Lightning** for training
- **Hydra** for configuration management
- **Megatron Core** for distributed training (TP, PP, FSDP, MoE)

### Configuration Approaches
- **NeMo 2.0**: Python-based configuration (newer LLM/VLM work)
- **NeMo 1.0**: YAML-based configuration (ASR, TTS)

## Code Style Conventions

### Naming
- Config classes: `MyModelConfig`
- Abstract models: `MyModel` (with "Model" postfix)
- Concrete models: simple names (e.g., `Conformer`, `QuartzNet`)
- Datasets: "Dataset" postfix (e.g., `AudioToSpeechLabelDataset`)
- Losses: "Loss" postfix (e.g., `CTCLoss`)
- No "I", "Interface", "NM", or "NeMo" prefixes

### Code Guidelines
- Methods should be ≤75 lines
- Use `from nemo.utils import logging` instead of `print`
- Use `raise Error` instead of `assert`
- F-strings preferred over `.format()`
- Private functions (`_name`) should only be called within their module
- Type hints required for public APIs

## CI/CD

- Add "Run CICD" label to PR to trigger CI tests
- Lint checks run automatically on changed files
- Add "skip-linting" label to skip lint checks (discouraged)
- CI selectively runs tests based on changed files

## PR Guidelines

- Send PRs to `main` branch
- Sign commits: `git commit -s`
- Tag @nithinraok for NeMo core/ASR, @blisc for TTS
