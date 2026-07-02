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

class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes,bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab 
        self.merges = merges 
        

        self.bytes_to_id = {}
        for id, b in vocab.items():
            self.bytes_to_id[b] = id
        
        self.special_tokens = special_tokens or list()
        self.special_bytes = set([
            token.encode("UTF-8") for token in self.special_tokens
        ])
        # we want to match the longest special if they overlap
        if special_tokens:
            special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
            self.specials_regex = "(" + \
                "|".join(regex.escape(tok) for tok in special_tokens_sorted) + \
            ")"
        
            
    
    def from_files(
        cls, 
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        raise NotImplementedError()

    def pretokenize(self, text: str) -> list[str]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        return regex.findall(PAT, text)
    
    def split_into_parts_and_pretokenize(self, text: str) -> list[Part]:
        parts = [text]
        if self.special_tokens:
            parts = regex.split(self.specials_regex, text)
        tokens = []
        for part in parts:
            if part in self.special_tokens or set():
                tokens.append(SpecialPart(part))
            elif not part:
                continue
            else:
                for piece in self.pretokenize(part):
                    tokens.append(NormalPart(piece))
        return tokens
    
    def _encode_bytes(self, list_of_bytes: list[bytes]) -> list[int]:
        for x, y in self.merges:
            new_list_of_bytes: list[bytes] = []
            i = 0
            n = len(list_of_bytes)
            # TODO: could refactor into new function
            while i < n:
                if list_of_bytes[i] in self.special_bytes:
                    new_list_of_bytes.append(list_of_bytes[i])
                    i += 1
                    continue 
                if i == n - 1:
                    new_list_of_bytes.append(list_of_bytes[i])
                    i += 1
                    continue
                
                a, b = list_of_bytes[i], list_of_bytes[i + 1]
                if (x, y) == (a, b):
                    new_list_of_bytes.append(
                        a + b
                    )
                    i += 2
                else:
                    new_list_of_bytes.append(a)
                    i += 1
                
            list_of_bytes = new_list_of_bytes

        ids = list()
        for b in list_of_bytes:
            ids.append(
                self.bytes_to_id[b]
            )
        return ids


    def encode(self, text: str) -> list[int]:
        parts = self.split_into_parts_and_pretokenize(text)
        ids: list[int] = []
        for part in parts:
            match part:
                case NormalPart(text = s):
                    list_of_bytes = [bytes([byte]) for byte in s.encode("utf-8")]
                    ids += self._encode_bytes(list_of_bytes)
                case SpecialPart(text = s):
                    b = s.encode("utf-8")
                    if isinstance(b, bytes):
                        list_of_bytes = [b]
                    else:
                        list_of_bytes = [bytes([b])]
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
    