import argparse
import cs336_basics.utils as utils
import numpy as np

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

    input_seq, output_seq = utils.run_get_batch(
        tokenized_corpus, batch_size, context_length, device = "mps")
    print(input_seq, output_seq)
    pass 

if __name__ == "__main__":
    main()