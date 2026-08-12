import argparse
import tomllib

import numpy as np
import torch

import cs336_basics.utils as utils
from cs336_basics.adamw import AdamW
from cs336_basics.transformer_lm import TransformerLM

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

def main():
    args = parse_args()
    config = load_config(args.config)

    # ----------------
    # Data
    # ----------------

    train_file = config["data"]["train_file"]
    validation_file = config["data"]["validation_file"]

    tokenized_corpus_train = np.load(
        train_file,
        mmap_mode="r",
    )

    tokenized_corpus_valid = np.load(
        validation_file,
        mmap_mode="r",
    )

    # ----------------
    # Model 
    # ----------------

    model_config = config["model"]

    transformer = init_transformer(model_config)

    # ----------------
    # Training config
    # ----------------

    training_config = config["training"]

    batch_size = training_config["batch_size"]
    max_iters = training_config["max_iters"]
    log_every = training_config["log_every"]

    device_str = training_config["device"]
    device = None if device_str == "cpu" else device_str


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
        '''
        #------- DEBUG -----
        if step % log_every == 0:
            from cs336_basics.scripts.decode import load_tokenizer
            tokenizer = load_tokenizer("data/tokenizer.pkl")
            predicted_tokens = torch.argmax(logits, dim = -1)
            
            print("input:")
            print(input_seq[0])
            print(output_seq[0])
            print(tokenizer.decode(input_seq[0].tolist()))
            print("predicted:")
            print(tokenizer.decode(predicted_tokens[0].tolist()))
            print("actual:")
            print(tokenizer.decode(output_seq[0].tolist()))
        '''
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

        if step % log_every == 0:
            print(f"step: {step} train loss: {loss.item():.4f}")

    # ----------------
    # Save Checkpoint
    # ----------------

    checkpoint_config = config["checkpoint"]
    output_file = checkpoint_config.get("output_file")
    if output_file is not None:
        print(f"saving model checkpoint to {output_file}")
        utils.run_save_checkpoint(transformer, opt, max_iters - 1, output_file)

    # ----------------
    # Validation
    # ----------------

    transformer.eval()

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

        validation_loss = utils.run_cross_entropy(
            logits,
            output_seq,
        )

    print(f"validation loss: {validation_loss.item():.4f}")


if __name__ == "__main__":
    main()