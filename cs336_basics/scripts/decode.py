from cs336_basics.scripts.training_loop import init_transformer

import argparse
from cs336_basics.tokenizer import Tokenizer
import cs336_basics.utils as utils
import numpy as np
import pickle
import tomllib
import torch

def load_tokenizer(tokenizer_filepath: str):
    with open(tokenizer_filepath, "rb") as f:
        tokenizer: Tokenizer = pickle.load(f)
    return tokenizer

def load_config(config_path: str) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-filepath", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config_filepath)

    tokenizer = load_tokenizer(config["tokenizer_filepath"])

    #utils.run_load_checkpoint(config["model_checkpoint"])

    transformer = init_transformer(config["model"])
    utils.load_model_checkpoint(config["model_filepath"], transformer)

    prompt = config["prompt"]
    prompt_tokens = tokenizer.encode(prompt)

    while prompt_tokens[-1] != 256 and len(prompt_tokens) < 500:
        input = torch.tensor(prompt_tokens)
        logits = transformer.forward(input, torch.arange(len(prompt_tokens)))
        next_token_logits = logits[-1]
        #softmax = utils.run_softmax(next_token_logits, dim = -1)
        next_token_id = torch.argmax(next_token_logits).item()
        prompt_tokens.append(next_token_id)
        if tokenizer.decode([next_token_id]) == "<|endoftext|>":
            break

    print(
        tokenizer.decode(prompt_tokens)
    )