import argparse
import pickle
import numpy as np
import tomllib
import torch

from cs336_basics.iter_parts import iter_parts
from cs336_basics.tokenizer import Tokenizer
from pathlib import Path

def load_config(config_path: str) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)

SPECIAL_TOKENS = list(["<|endoftext|>"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-filepath", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config_filepath)

    with open(config["merges_pickle"], "rb") as f:
        merges = pickle.load(f)
    with open(config["vocab_pickle"], "rb") as f:
        vocab = pickle.load(f)

    tokenizer = Tokenizer(
        vocab,
        merges,
        SPECIAL_TOKENS,
    )

    parts = iter_parts(config["corpus_filepath"], SPECIAL_TOKENS)
    tokens = tokenizer.encode_iterable(parts)
    tokens_array = np.fromiter(tokens, dtype = np.uint16)
    print(len(tokens_array))    

    output_path = Path(config["corpus_filepath"]).with_suffix(".tokens")
    np.save(output_path, tokens_array)
    

if __name__ == "__main__":
    main()