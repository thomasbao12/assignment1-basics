from collections import defaultdict
from cs336_basics.tokenizer import (
    encode_part_into_bytes, 
    merge_pair,
    split_into_parts_and_pretokenize
)

def get_merge_pairs_to_ranks(list_of_bytes: list[bytes]) -> dict[tuple[bytes, bytes],int]:
    potential_merges = defaultdict(int)
    for i in range(0, len(list_of_bytes) - 1):
        x, y = list_of_bytes[i], list_of_bytes[i + 1]
        potential_merges[(x, y)] += 1
    
    return potential_merges

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {
        i: bytes(i)
        for i in range(256)
    }

    # todo support multiprocessing with chunking
    merges = []
    with open(input_path) as f:
        text = f.read()
        parts = split_into_parts_and_pretokenize(text)
        for part in parts:
            list_of_bytes = encode_part_into_bytes(part)
            merge_pairs_to_ranks = get_merge_pairs_to_ranks(list_of_bytes)
            list_of_bytes, merged = merge_pair(list_of_bytes, merge_pairs_to_ranks)
            if not merged:
                break
            merges.append(merged)
            vocab[len(vocab)] = merged
    return (vocab, merges)