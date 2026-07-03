from collections import defaultdict
from cs336_basics.tokenizer import (
    encode_part_into_bytes, 
    _merge_pair,
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
        i: bytes([i])
        for i in range(256)
    }
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')

    # todo support multiprocessing with chunking
    merges = []
    with open(input_path) as f:
        text = f.read()
        parts = [
            encode_part_into_bytes(part)
            for part in split_into_parts_and_pretokenize(text, special_tokens)
        ]
        
        while len(vocab) < vocab_size:
            merge_pairs_to_count: dict[tuple[bytes, bytes], int] = dict()
            for part in parts:
                for k, v in get_merge_pairs_to_ranks(part).items():
                    merge_pairs_to_count[k] = merge_pairs_to_count.get(k, 0) + v
            max_count, max_merge = max([
                (count, merge_pair)
                for merge_pair, count in merge_pairs_to_count.items()
            ])
            if max_count == 0:
                break 
            for i, part in enumerate(parts):
                part = _merge_pair(part, max_merge)
                parts[i] = part
                
            merges.append(max_merge)
            vocab[len(vocab)] = b''.join(max_merge)
    
    return (vocab, merges)