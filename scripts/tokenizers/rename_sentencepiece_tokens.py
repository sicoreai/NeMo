"""Rename tokens in an existing sentencepiece model.

Generate sentencepiece_model_pb2.py in the directory of this script before running.
To generate run `protoc --python_out=<path_to_NeMo>/scripts/tokenizers/ sentencepiece_model.proto`
inside the src folder in sentencepiece repo.
Refer: https://github.com/google/sentencepiece/issues/121

Outputs:
    - .model file (sentencepiece binary model)
    - .vocab file (tab-separated: token<TAB>score)
    - vocab.txt file (one token per line)

Usage:
    python rename_sentencepiece_tokens.py \
        --input_file tokenizer.model \
        --output_dir ./modified_tokenizer
    
    python rename_sentencepiece_tokens.py --input_file ./canary_1b_v2_tokenizers/tokenizer.model --output_dir ./greek_fixed_tokenizer
"""

import logging
import math
import sys
from argparse import ArgumentParser
from pathlib import Path

import sentencepiece as spm

try:
    import sentencepiece_model_pb2 as spt
except (ImportError, ModuleNotFoundError):
    raise Exception("Ensure that sentencepiece_model_pb2.py has been generated from the protoc compiler")


# =============================================================================
# CONFIGURATION - Define token renames here
#
# Each entry: "old_token": ("new_token", type)
#   type: "user_defined" — always tokenized as single token, visible in output
#         "control"      — must be inserted programmatically, stripped in decoding
#         None           — keep the original type unchanged
# =============================================================================

# SentencePiece type constants:
#   1 = NORMAL, 2 = UNKNOWN, 3 = CONTROL, 4 = USER_DEFINED, 5 = UNUSED, 6 = BYTE
TYPE_NAMES = {
    "normal": 1,
    "control": 3,
    "user_defined": 4,
}

RENAME_MAP: dict[str, tuple[str, str | None]] = {
    "Ū": ("ς", None),
    # Keep original type:
    # "<|spltoken2|>": ("<|another_token|>", None),
    # Change to control symbol:
    # "<|spltoken3|>": ("<|internal_marker|>", "control"),
}


def rename_tokens(input_file: str, output_dir: str, rename_map: dict[str, tuple[str, str | None]]) -> None:
    model = spt.ModelProto()
    model.ParseFromString(open(input_file, 'rb').read())

    # Build a set of existing pieces for conflict detection
    existing_pieces = {p.piece for p in model.pieces}

    # Check for conflicts: new names must not already exist (unless they are being renamed themselves)
    for old_name, (new_name, _) in rename_map.items():
        if new_name in existing_pieces and new_name not in rename_map:
            logging.error(f"Target token '{new_name}' already exists in the model and is not being renamed itself.")
            sys.exit(1)

    # Apply renames
    renamed_count = 0
    for piece in model.pieces:
        if piece.piece in rename_map:
            old_name = piece.piece
            new_name, new_type = rename_map[old_name]
            piece.piece = new_name
            type_info = ""
            if new_type is not None:
                if new_type not in TYPE_NAMES:
                    logging.error(f"Unknown type '{new_type}'. Must be one of: {list(TYPE_NAMES.keys())}")
                    sys.exit(1)
                old_type_id = piece.type
                piece.type = TYPE_NAMES[new_type]
                type_info = f" (type: {old_type_id} -> {piece.type}/{new_type})"
            renamed_count += 1
            logging.info(f"Renamed '{old_name}' -> '{new_name}'{type_info}")

    # Check all requested renames were applied
    for old_name in rename_map:
        if old_name not in existing_pieces:
            logging.warning(f"Token '{old_name}' not found in the model — skipped.")

    if renamed_count == 0:
        logging.warning("No tokens were renamed.")
        return

    # Validate the modified model
    sp = spm.SentencePieceProcessor()
    try:
        sp.LoadFromSerializedProto(model.SerializeToString())
        logging.info(f"Validated modified model. Vocab size: {sp.get_piece_size()}")
    except Exception:
        logging.error("Could not load modified model. Check for duplicate or invalid tokens.")
        sys.exit(1)

    # Write outputs
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "tokenizer.model"
    vocab_path = out / "tokenizer.vocab"
    vocab_txt_path = out / "vocab.txt"

    # Write .model
    with open(model_path, 'wb') as f:
        f.write(model.SerializeToString())
    logging.info(f"Wrote model:     {model_path}")

    # Write .vocab (tab-separated: token<TAB>score, integer scores)
    # Note: first normal token has score -0.0; int(-0.0)==0 but original format is "-0"
    def _fmt_score(score):
        i = int(score)
        if i == 0 and math.copysign(1, score) < 0:
            return "-0"
        return str(i)

    with open(vocab_path, 'w', encoding='utf-8') as f:
        for piece in model.pieces:
            f.write(f"{piece.piece}\t{_fmt_score(piece.score)}\n")
    logging.info(f"Wrote vocab:     {vocab_path}")

    # Write vocab.txt (BERT-style: only NORMAL tokens, ▁ → strip, else → ## prefix)
    with open(vocab_txt_path, 'w', encoding='utf-8') as f:
        for piece in model.pieces:
            if piece.type != 1:  # skip non-NORMAL (UNKNOWN=2, CONTROL=3, USER_DEFINED=4, UNUSED=5, BYTE=6)
                continue
            if piece.piece.startswith('▁'):
                token = piece.piece[1:]
            else:
                token = f"##{piece.piece}"
            if len(token) > 0:
                f.write(f"{token}\n")
            else:
                f.write(f"{piece.piece[0]}\n")
    logging.info(f"Wrote vocab.txt: {vocab_txt_path}")

    logging.info(f"Renamed {renamed_count} token(s).")


def main():
    parser = ArgumentParser(description="Rename tokens in a sentencepiece model.")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to input sentencepiece .model file",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for modified model and vocab files",
    )
    args = parser.parse_args()

    rename_tokens(args.input_file, args.output_dir, RENAME_MAP)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
