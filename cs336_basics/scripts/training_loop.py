import argparse
import cs336_basics.utils as utils
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
import numpy as np

import torch

'''
# Hyperparams:

## Model

- vocab_size
- context_length
- d_model
- num_layers
- d_ff

## Optimization

- learning_rate
- weight_decay
- beta1
- beta2
- eps
- max_grad_norm

## Learning rate schedule

- warmup_iters
- cosine_cycle_iters
- min_learning_rate

## Training 

- batch_size
- max_iters
- device
- dtype

'''
'''

uv run python cs336_basics/scripts/training_loop.py \
    --tokenized-corpus-file-path data/tiny_example.tokens.npy \
    --batch-size 3 \
    --context-length 10
'''

def main():
    tokenized_corpus_train_file_path = "data/tiny_stories_train_smoke_test.tokens.npy"
    tokenized_corpus_train = np.load(tokenized_corpus_train_file_path, mmap_mode = "r")

    
    context_length = 128

    vocab_size = 1001 # from tokenize_corpus.py
    d_model = 128
    num_layers = 2
    num_heads = 4
    d_ff = 384
    rope_theta = 10000.0
    device = None #"mps"
    transformer = TransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
        device = device
    )

    batch_size = 4
    
    opt = AdamW(
        transformer.parameters(),
        lr = 0.01,
        betas = (0.9, 0.999),
        eps = 1e-8,
        weight_decay=0.01,
    )
    token_positions = torch.arange(
        context_length,
        device = device
    ).expand(batch_size, context_length)

    
    
    for step in range(1000):
        opt.zero_grad()
        input_seq, output_seq = utils.run_get_batch(
                tokenized_corpus_train, batch_size, context_length, device = device)

        
        logits = transformer.forward(input_seq, token_positions)

        loss = utils.run_cross_entropy(logits, output_seq)

        loss.backward()
        opt.step()
        
        if step % 100 == 0:
            print(f"step: {step} loss: {loss}")

    # validation
    tokenized_corpus_valid_file_path = "data/tiny_stories_valid_smoke_test.tokens.npy"
    tokenized_corpus_valid = np.load(tokenized_corpus_valid_file_path)

    opt.zero_grad()
    input_seq, output_seq = utils.run_get_batch(
        tokenized_corpus_valid, batch_size, context_length, device = device)

    
    logits = transformer.forward(input_seq, token_positions)

    loss = utils.run_cross_entropy(logits, output_seq)
    print(f"validation loss: {loss}")
    
    if step % 100 == 0:
        print(f"step: {step} loss: {loss}")
    
    pass 

if __name__ == "__main__":
    main()