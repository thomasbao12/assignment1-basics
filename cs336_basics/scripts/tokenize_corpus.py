import argparse
import pickle
import numpy as np
import tomllib
import torch

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import train_bpe
from pathlib import Path

def load_config(config_path: str) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)

def encode_to_file(tokenizer, filepath):
    with open(filepath) as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    tokens_array = np.array(tokens)
     
    output_path = Path(filepath).with_suffix(".tokens")
    np.save(output_path, tokens_array)

SPECIAL_TOKENS = list(["<|endoftext|>"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-filepath", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config_filepath)
    corpus_train_filepath = config["corpus_train_filepath"]
    vocab, merges = train_bpe(
        corpus_train_filepath,
        1000,
        SPECIAL_TOKENS,
    )

    tokenizer = Tokenizer(
        vocab,
        merges,
        SPECIAL_TOKENS,
    )

    with open(config["tokenizer_filepath"], "wb") as f:
        pickle.dump(tokenizer, f)

    encode_to_file(tokenizer, corpus_train_filepath)
    encode_to_file(tokenizer, config["corpus_valid_filepath"])
    

if __name__ == "__main__":
    main()