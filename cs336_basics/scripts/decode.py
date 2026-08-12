from training_loop import init_transformer

import argparse
import cs336_basics.utils as utils
import numpy as np
import pickle
import tomllib

def load_config(config_path: str) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


parser = argparse.ArgumentParser()
parser.add_argument("--config-filepath", type=str, required=True)
args = parser.parse_args()

config = load_config(args.config_filepath)

with open(config["tokenizer_filepath"], "rb") as f:
    tokenizer = pickle.load(f)

#utils.run_load_checkpoint(config["model_checkpoint"])

transformer = init_transformer(config["model"])
utils.load_model_checkpoint(config["model_filepath"], transformer)

tokens = np.load(
    "data/tiny_stories_valid_smoke_test.tokens.npy",
    mmap_mode="r",
)
print(tokenizer.decode(tokens))
#transformer.forward()