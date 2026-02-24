"""Replace the tokenizer in a canary model and export to .nemo.

This does NOT use change_vocabulary() — that method reinitializes the decoder
head with random weights, destroying trained parameters. Instead, this script
directly replaces the tokenizer files and re-registers them as NeMo artifacts,
preserving all model weights exactly.

The replacement tokenizer directory must contain tokenizer.model (and optionally
tokenizer.vocab and vocab.txt). Use rename_sentencepiece_tokens.py to prepare
a modified tokenizer directory.

Usage:
    # Download model and replace tokenizer:
    python patch_canary_tokenizer.py \
        --tokenizer_dir ./greek_fixed_tokenizer \
        --output_nemo ./canary-1b-v2-patched.nemo

    # Use a local .nemo instead of downloading:
    python patch_canary_tokenizer.py \
        --input_nemo ./canary-1b-v2.nemo \
        --tokenizer_dir ./greek_fixed_tokenizer \
        --output_nemo ./canary-1b-v2-patched.nemo
"""

import logging
import os
import sys
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(description="Replace canary tokenizer and export .nemo")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model_name", default="nvidia/canary-1b-v2", help="Pretrained model name to download")
    group.add_argument("--input_nemo", type=str, help="Path to local .nemo file (skip download)")
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="Path to replacement tokenizer directory")
    parser.add_argument("--output_nemo", type=str, required=True, help="Output .nemo file path")
    args = parser.parse_args()

    # Validate tokenizer dir
    tokenizer_model = os.path.join(args.tokenizer_dir, "tokenizer.model")
    if not os.path.isfile(tokenizer_model):
        log.error(f"tokenizer.model not found in {args.tokenizer_dir}")
        sys.exit(1)

    # Lazy imports — NeMo is heavy
    from omegaconf import OmegaConf, open_dict
    from nemo.collections.asr.models import EncDecMultiTaskModel

    # 1. Load model
    if args.input_nemo:
        log.info(f"Loading model from {args.input_nemo}...")
        model = EncDecMultiTaskModel.restore_from(args.input_nemo, map_location='cpu')
    else:
        log.info(f"Downloading {args.model_name}...")
        model = EncDecMultiTaskModel.from_pretrained(args.model_name, map_location='cpu')
    log.info("Model loaded.")

    # 2. Re-setup tokenizer from the user-provided directory.
    #    This re-registers artifact files for save_to() but does NOT
    #    touch any model weights (encoder, decoder, head).
    #    We pass individual file paths instead of dir so the saved config
    #    has dir: null (same as the official model), making the .nemo portable.
    tokenizer_dir = os.path.abspath(args.tokenizer_dir)
    model_path = os.path.join(tokenizer_dir, 'tokenizer.model')
    vocab_path = os.path.join(tokenizer_dir, 'vocab.txt')
    spe_vocab_path = os.path.join(tokenizer_dir, 'tokenizer.vocab')

    tokenizer_cfg = OmegaConf.create({
        'dir': None,
        'type': 'bpe',
        'model_path': model_path,
        'vocab_path': vocab_path if os.path.isfile(vocab_path) else None,
        'spe_tokenizer_vocab': spe_vocab_path if os.path.isfile(spe_vocab_path) else None,
    })
    if hasattr(model.cfg.tokenizer, 'custom_tokenizer'):
        with open_dict(tokenizer_cfg):
            tokenizer_cfg.custom_tokenizer = model.cfg.tokenizer.custom_tokenizer

    model._setup_tokenizer(tokenizer_cfg)
    with open_dict(model.cfg):
        model.cfg.tokenizer = tokenizer_cfg
    log.info(f"Replaced tokenizer from {tokenizer_dir} (all model weights preserved).")

    # 3. Save .nemo
    model.save_to(args.output_nemo)
    log.info(f"Saved patched model to {args.output_nemo}")


if __name__ == "__main__":
    main()
