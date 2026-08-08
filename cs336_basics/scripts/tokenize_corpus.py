import argparse
import numpy as np

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import train_bpe
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-file-path", type=str, required=True)
    args = parser.parse_args()

    corpus_file_path = args.corpus_file_path
    vocab, merges = train_bpe(
        corpus_file_path,
        1000,
        list()
    )

    tokenizer = Tokenizer(
        vocab,
        merges,
        list()
    )
    with open(corpus_file_path) as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    tokens_array = np.array(tokens)

    output_path = Path(corpus_file_path).with_suffix(".tokens")
    
    np.save(output_path, tokens_array)
    

if __name__ == "__main__":
    main()