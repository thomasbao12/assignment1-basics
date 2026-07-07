import regex

from collections.abc import Iterable, Iterator

from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class NormalPart:
    text: str

@dataclass(frozen=True)
class SpecialPart:
    text: str

Part = NormalPart | SpecialPart

def pretokenize(text: str) -> list[str]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        return regex.findall(PAT, text)

def split_into_parts_and_pretokenize(
    text: str, 
    special_tokens: list[str] | None = None
) -> list[Part]:
    parts = [text]
    if special_tokens is None:
        special_tokens = set()
    if len(special_tokens) > 0:
        # we want to match the longest special if they overlap
        special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
        specials_regex = "(" + \
            "|".join(regex.escape(tok) for tok in special_tokens_sorted) + \
        ")"
        parts = regex.split(specials_regex, text)
    tokens = []
    for part in parts:
        if part in special_tokens or set():
            tokens.append(SpecialPart(part))
        elif not part:
            continue
        else:
            for piece in pretokenize(part):
                tokens.append(NormalPart(piece))
    return tokens

def encode_part_into_bytes(
    part: Part
) -> list[bytes]:
    match part:
        case NormalPart(text = s):
            return [bytes([byte]) for byte in s.encode("utf-8")]
        case SpecialPart(text = s):
            b = s.encode("utf-8")
            if isinstance(b, bytes):
                return [b]
            else:
                return [bytes([b])]
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
    
    def _encode_bytes(self, list_of_bytes: list[bytes]) -> list[int]:
        parts = list_of_bytes
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
        parts = split_into_parts_and_pretokenize(text, self.special_tokens)
        ids: list[int] = []
        for part in parts:
            list_of_bytes = encode_part_into_bytes(part)
            ids += self._encode_bytes(list_of_bytes)
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        bytes = b""
        for id in ids:
            bytes += self.vocab.get(id)
        
        return bytes.decode("utf-8", errors="replace")
    