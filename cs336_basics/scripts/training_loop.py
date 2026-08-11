import argparse
import cs336_basics.utils as utils
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.sgd import SGD
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
    # load tokenized ids
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized-corpus-file-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)

    args = parser.parse_args()
    
    tokenized_corpus_file_path = args.tokenized_corpus_file_path
    tokenized_corpus = np.load(tokenized_corpus_file_path)

    batch_size = args.batch_size
    context_length = args.context_length

    vocab_size = 1000 # from tokenize_corpus.py
    d_model = 64
    num_layers = 2
    num_heads = 4
    d_ff = 128
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
    opt = SGD(transformer.parameters(), 1)
    token_positions = torch.arange(
        context_length,
        device = device
    ).expand(batch_size, context_length)

    input_seq, output_seq = utils.run_get_batch(
        tokenized_corpus, batch_size, context_length, device = device)
    
    for step in range(1000):
        logits = transformer.forward(input_seq, token_positions)
        y_hat = utils.run_softmax(logits, -1)
        
        loss: torch.Tensor = utils.run_cross_entropy(
            y_hat, 
            output_seq
        )
        loss.backward()
        
        opt.step()
        if step % 100 == 0:
            print(f"step: {step} loss: {loss}")
        

    pass 

if __name__ == "__main__":
    main()