import pickle
from cs336_basics.train_bpe import train_bpe

def main():
    corpus_filepath = "data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_bpe(corpus_filepath, vocab_size, special_tokens)

    with open("data/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("data/merges.pkl", "wb") as f:
        pickle.dump(merges, f)

if __name__ == "__main__":
    main()