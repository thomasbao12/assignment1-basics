from collections import defaultdict
from cs336_basics.tokenizer import (
    encode_part_into_bytes, 
    _merge_pair,
    split_into_parts_and_pretokenize
)
from sortedcontainers import SortedDict

import cProfile
import pstats

def get_merge_pairs_to_ranks(list_of_bytes: list[bytes]) -> dict[tuple[bytes, bytes],int]:
    potential_merges = defaultdict(int)
    for i in range(0, len(list_of_bytes) - 1):
        x, y = list_of_bytes[i], list_of_bytes[i + 1]
        potential_merges[(x, y)] += 1
    
    return potential_merges

# merge pairs 
# index merge pairs

class BPE:
    DEBUG = False
    '''
    invariants: 
    - bytes to positions and position to bytes are inverse dictionaries
    - we should be able to reconstruct each part from the correspond position to bytes
    - byte counts individually of the bytes_pair_to_counts should equal the byte counts of parts
    '''
    
    def _assert_position_to_bytes_is_equivalent_to_parts(self):
        for i, part in enumerate(self.parts):
            reconstructed_part = list()
            for _, value in sorted(self.position_to_bytes[i].items()):
                reconstructed_part.append(value)
            assert b''.join(part) == b''.join(reconstructed_part), f'{part} != {reconstructed_part}'
    
    def _assert_position_to_bytes_equivalent_to_bytes_to_position(self):
        for b, positions in self.bytes_to_positions.items():
            for (i, j) in positions:
                assert self.position_to_bytes[i][j] == b

        for i, v in self.position_to_bytes.items():
            for j, b in self.position_to_bytes[i].items():
                assert (i,j) in self.bytes_to_positions[b]

    def __init__(self, vocab: dict[int, bytes], parts: list[list[bytes]]):
        self.parts = parts
        self.merged = []
        self.vocab = vocab 
        
        self.bytes_to_positions: dict[bytes, set] = defaultdict(set)
        # when sorted and merged, this should be equivalent to parts
        self.position_to_bytes: dict[int, dict[int, bytes]] = defaultdict(dict)
        self.bytes_pair_to_counts: dict[tuple(bytes, bytes), int] = defaultdict(int)

        for i, list_of_bytes in enumerate(self.parts):
            for j in range(len(list_of_bytes) - 1):
                x, y = list_of_bytes[j], list_of_bytes[j + 1]
                self.bytes_pair_to_counts[(x, y)] += 1
                self.bytes_to_positions[x].add((i, j))
                self.position_to_bytes[i][j] = list_of_bytes[j]
            if len(list_of_bytes) > 0:
                j = len(list_of_bytes) - 1
                self.bytes_to_positions[list_of_bytes[j]].add((i, j))
                self.position_to_bytes[i][j] = list_of_bytes[j]

        if BPE.DEBUG:
            print("init: about to assert")
            self._assert_position_to_bytes_equivalent_to_bytes_to_position()
            self._assert_position_to_bytes_is_equivalent_to_parts()
            print("init: passed asserts")
    
    # what if merged bytes are the SAME!?
    '''
    always operate on position_to_bytes/bytes_to_position
    first decrement byte_pairs_to_counts
    then update position_to_bytes/bytes_to_position on merged_bytes
    finally increment byte_pairs_to_counts on merged_bytes
    '''
    def merge_dry_run(self) -> bool:
        # TODO use sorted dict
        max_count, max_merge = max([
            (count, pair)
            for pair, count in self.bytes_pair_to_counts.items()
        ])
        if max_count == 0:
            return 0
        
        x, y = max_merge 
        merged_bytes = x + y

        # decrement byte pair counts affected by the merge prior to merging
        # we want to avoid decrementing, so we use a set
        pairs_of_positions = set()
        for (i, j) in self.bytes_to_positions[x]:
            k = j + len(x)
            if self.position_to_bytes.get((i, k)) != y:
                continue 
            pairs_of_positions.add(((i, j), (i, k)))

            h = j - 1
            while h > 0 and (i, h) not in self.position_to_bytes.keys():
                h = h - 1
            if h >= 0:
                pairs_of_positions.add(((i, h), (i, j)))
                         
            l = j + len(x) + len(y)
            if l < len(parts[i]):
                pairs_of_positions.add(((i, k), (i, l)))
        for p1, p2 in pairs_of_positions:
            a = self.position_to_bytes[p1]
            b = self.position_to_bytes[p2]
            self.bytes_pair_to_counts[a,b] -= 1

        self.vocab[len(self.vocab)] = merged_bytes
        self.merged.append(max_merge)

        del self.bytes_pair_to_counts[(x, y)]

        # update bytes_to_positions and positions_to_bytes
        bytes_and_positions_to_delete: tuple[bytes, tuple[int, int]] = []
        last_merge_boundary = -1
        for (i, j) in sorted(self.bytes_to_positions[x]):
            if j == last_merge_boundary:
                continue
            
            k = j + len(x)
            if k < len(self.parts[i]) and self.position_to_bytes[(i,k)] == y:
                self.bytes_to_positions[merged_bytes].add((i, j))
                bytes_and_positions_to_delete.append((x, (i,j)))
                bytes_and_positions_to_delete.append((y, (i,k)))
                
                self.position_to_bytes[(i, j)] = merged_bytes 
                del self.position_to_bytes[(i, j + len(x))] 
                last_merge_boundary = j + len(x)
        
        for (b, (i, j)) in bytes_and_positions_to_delete:
            self.bytes_to_positions[b].remove((i, j))

        if BPE.DEBUG:
            self._assert_position_to_bytes_is_equivalent_to_parts()
            self._assert_position_to_bytes_equivalent_to_bytes_to_position()            

        # increment byte pair counts affected by the merge after merging
        # merged bytes could be consecutive, so we want to avoid double counting
        # pairs of positions before and after, so we use a set
        pairs_of_positions = set()
        for (i, j) in self.bytes_to_positions[merged_bytes]:    
            k = j + len(merged_bytes)
            if k < len(parts[i]):
                pairs_of_positions.add(((i, j), (i, k)))                
            
            h = j - 1
            while h > 0 and (i, h) not in self.position_to_bytes.keys():
                h = h - 1
            if h >= 0:
                pairs_of_positions.add(((i, h), (i, j)))
        for p1, p2 in pairs_of_positions:
            self.bytes_pair_to_counts[(
                self.position_to_bytes[p1],
                self.position_to_bytes[p2]
            )] += 1
                
        return True 

    def __repr__(self):
        return str({
            "merged": self.merged,
            "vocab_size": len(self.vocab),
            "bytes_pair_to_counts": self.bytes_pair_to_counts,
            "bytes_to_positions": self.bytes_to_positions,
            "position_to_bytes": self.position_to_bytes
        })
    
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
    with open(input_path) as f:
        text = f.read()
        parts = [
            encode_part_into_bytes(part)
            for part in split_into_parts_and_pretokenize(text, special_tokens)
        ]
        if BPE.DEBUG:
            profiler = cProfile.Profile()
            profiler.enable()
        
        bpe = BPE(vocab, parts)
        while len(bpe.vocab) < vocab_size:
            did_merge = bpe.merge_dry_run()
            if not did_merge:
                break
        

    if BPE.DEBUG:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        stats.print_stats(20)
    
    
    return (bpe.vocab, bpe.merged)

if __name__ == "__main__":
    parts = [[
        b'1',
        b'1',
        b'1',
        b'1',
        b'1',
        b'1',
        b'1'
    ]]
    vocab = {
        i: bytes([i])
        for i in range(256)
    }
    bpe = BPE(vocab, parts)
    #print(bpe)
    bpe.merge_dry_run()
    #print(bpe.bytes_pair_to_counts)
    bpe.merge_dry_run()
    bpe.merge_dry_run()
    bpe.merge_dry_run()
    