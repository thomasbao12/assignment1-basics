import argparse
import numpy as np
import shutil
import time
import tomllib
import torch

from cs336_basics.adamw import AdamW
from cs336_basics.experiment_logger import ExperimentLogger
from cs336_basics.transformer_lm import TransformerLM
from pathlib import Path

import cs336_basics.utils as utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to TOML configuration file.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)

def init_transformer(model_config: dict) -> TransformerLM:
    vocab_size = model_config["vocab_size"]
    context_length = model_config["context_length"]
    d_model = model_config["d_model"]
    num_layers = model_config["num_layers"]
    num_heads = model_config["num_heads"]
    d_ff = model_config["d_ff"]
    rope_theta = model_config["rope_theta"]
    device = model_config.get("device", None)
    return TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device,
    )

def get_validation_loss(config, transformer, token_positions, experiment_logger, step):
    validation_file = config["data"]["validation_file"]
    context_length = config["model"]["context_length"]
    batch_size = config["training"]["batch_size"]
    device = config["model"]["device"]
    
    
    tokenized_corpus_valid = np.load(
        validation_file,
        mmap_mode="r",
    )

    with torch.no_grad():
        input_seq, output_seq = utils.run_get_batch(
            tokenized_corpus_valid,
            batch_size,
            context_length,
            device=device,
        )

        logits = transformer(
            input_seq,
            token_positions,
        )
        
        loss = utils.run_cross_entropy(
            logits,
            output_seq,
        )

    print(f"validation loss: {loss.item():.4f}")
    experiment_logger.log(step, "valid", loss)


def main():
    args = parse_args()
    config = load_config(args.config)

    # ----------------
    # Logging
    # ----------------
    epoch_time = int(time.time())
    
    logging_dir = Path(config["logging"]["dir"])
    logging_dir.mkdir(exist_ok=True)
    shutil.copy(args.config, logging_dir)

    experiment_logger = ExperimentLogger(logging_dir)
    experiment_logger.log(0, "train", 3.14159)
    
    # ----------------
    # Data
    # ----------------

    train_file = config["data"]["train_file"]

    tokenized_corpus_train = np.load(
        train_file,
        mmap_mode="r",
    )

    # ----------------
    # Model 
    # ----------------

    model_config = config["model"]
    device_str = model_config["device"]
    device = None if device_str == "cpu" else device_str

    transformer = init_transformer(model_config)

    # ----------------
    # Training config
    # ----------------

    training_config = config["training"]

    batch_size = training_config["batch_size"]
    max_iters = training_config["max_iters"]
    log_training_loss_every = training_config["log_training_loss_every"]
    log_validation_loss_every = training_config["log_validation_loss_every"]

    

    # ----------------
    # Optimizer
    # ----------------

    optimizer_config = config["optimizer"]

    opt = AdamW(
        transformer.parameters(),
        lr=optimizer_config["learning_rate"],
        betas=(
            optimizer_config["beta1"],
            optimizer_config["beta2"],
        ),
        eps=optimizer_config["eps"],
        weight_decay=optimizer_config["weight_decay"],
    )

    # ----------------
    # Positions
    # ----------------

    context_length = model_config["context_length"]
    token_positions = torch.arange(
        context_length,
        device=device,
    ).expand(batch_size, context_length)

    # ----------------
    # Load Checkpoint
    # ----------------

    checkpoint_config = config["checkpoint"]
    input_file = checkpoint_config.get("input_file")
    if input_file is not None:
        utils.run_load_checkpoint(input_file, transformer, opt)

    # ----------------
    # Training
    # ----------------

    transformer.train()

    for step in range(max_iters):
        input_seq, output_seq = utils.run_get_batch(
                tokenized_corpus_train,
                batch_size,
                context_length,
                device=device,
            )
        
        opt.zero_grad()

        logits = transformer(input_seq, token_positions)
        
        loss = utils.run_cross_entropy(
            logits,
            output_seq,
        )
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            transformer.parameters(),
            optimizer_config["max_grad_norm"],
        )

        opt.step()

        if step % log_training_loss_every == 0:
            print(f"step: {step} train loss: {loss.item():.4f}")
            experiment_logger.log(step, "train", loss.item())

        if step % log_validation_loss_every == 0:
            get_validation_loss(config, transformer, token_positions, experiment_logger, step)

        

    # ----------------
    # Save Checkpoint
    # ----------------

    checkpoint_config = config["checkpoint"]
    output_file = checkpoint_config.get("output_file")
    if output_file is not None:
        print(f"saving model checkpoint to {output_file}")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        utils.run_save_checkpoint(transformer, opt, max_iters - 1, output_file)


if __name__ == "__main__":
    main()