import regex

from collections.abc import Iterable, Iterator

from dataclasses import dataclass
from typing import Literal
from cs336_basics.iter_parts import (
    NormalPart, SpecialPart, Part, iter_parts_from_text
)
            
def _merge_pair(
    list_of_bytes: list[bytes],
    pair: tuple[bytes, bytes]    
) -> list[bytes]:
    for i in range(len(list_of_bytes) - 1):
        x, y = list_of_bytes[i], list_of_bytes[i + 1]
        if (x, y) == pair:
            return list_of_bytes[:i] + [
                b''.join([
                    list_of_bytes[i], 
                    list_of_bytes[i + 1]
                ])
            ] + _merge_pair(list_of_bytes[i + 2:], pair)

    return list_of_bytes

def merge_pair(
    list_of_bytes: list[bytes],
    merge_pairs_to_ranks: dict[tuple[bytes, bytes],int]
) -> tuple[list[bytes], tuple[bytes, bytes] | None]:
    best_rank = None
    best_idx = None 
    for i in range(len(list_of_bytes) - 1):
        x, y = list_of_bytes[i], list_of_bytes[i + 1]
        rank = merge_pairs_to_ranks.get((x, y))
        if rank is not None and (best_rank is None or rank < best_rank):
            best_rank = rank 
            best_idx = i 
    
    if best_idx is None:
        return (list_of_bytes, None)
    
    pair = (list_of_bytes[best_idx], list_of_bytes[best_idx + 1])
    return (_merge_pair(list_of_bytes, pair), pair)
    
class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes,bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab 
        self.merges = merges 
        self.merge_pairs_to_ranks = {
            merge_pair: rank 
            for rank, merge_pair in enumerate(self.merges)
        }

        self.bytes_to_id = {}
        for id, b in vocab.items():
            self.bytes_to_id[b] = id
        
        self.special_tokens = special_tokens or list()
        self.special_bytes = set([
            token.encode("UTF-8") for token in self.special_tokens
        ])
            
    
    def from_files(
        cls, 
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        raise NotImplementedError()
    
    def encode_bytes(self, bytestring: bytes) -> list[int]:
        parts = [bytes([b]) for b in bytestring]
        while len(parts) >= 2:
            parts, merged = merge_pair(
                parts,
                self.merge_pairs_to_ranks
            )
            if merged is None:
                break
            
        
        ids = list()
        for b in parts:
            ids.append(
                self.bytes_to_id[b]
            )
        return ids


    def encode(self, text: str) -> list[int]:
        ids: list[int] = []

        for part in iter_parts_from_text(
            text,
            self.special_tokens,
        ):
            match part:
                case NormalPart(data=data):
                    ids.extend(
                        self.encode_bytes(
                            data
                        )
                    )

                case SpecialPart(data=data):
                    ids.append(
                        self.bytes_to_id[data]
                    )
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        bytes = b""
        for id in ids:
            if id not in self.vocab:
                print(f"{id} is missing")

            bytes += self.vocab.get(id)
        
        return bytes.decode("utf-8", errors="replace")
    