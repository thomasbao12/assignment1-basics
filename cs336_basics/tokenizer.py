import re 

from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class NormalToken:
    text: str

@dataclass(frozen=True)
class SpecialToken:
    text: str

Token = NormalToken | SpecialToken

class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes,bytes]],
        special_tokens: list[str] | None
    ): 
        self.vocab = vocab 
        self.merges = merges 
        

        self.bytes_to_id = {}
        for id, b in vocab.items():
            self.bytes_to_id[b] = id
        
        self.special_tokens = set(special_tokens or [])
        # we want to match the longest special if they overlap
        special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
        self.specials_regex = "(" + \
                "|".join(re.escape(tok) for tok in special_tokens_sorted) + \
            ")"
        
        self.special_bytes = set([
            token.encode("UTF-8") for token in self.special_tokens
        ])

    def split_into_tokens(self, text: str) -> list[Token]:
        parts = re.split(self.specials_regex, text)
        tokens = []
        for part in parts:
            if part in self.special_tokens:
                tokens.append(SpecialToken(part))
            elif not part:
                continue
            else:
                tokens.append(NormalToken(part))
        return tokens
    
    def encode(self, text: str) -> list[int]:
        tokens = self.split_into_tokens(text)
        
        list_of_bytes: list[bytes] = []
        for token in tokens:
            match token:
                case NormalToken(text = s):
                    list_of_bytes += [bytes([byte]) for byte in s.encode("utf-8")]
                case SpecialToken(text = s):
                    b = s.encode("utf-8")
                    if isinstance(b, bytes):
                        list_of_bytes.append(b)
                    else:
                        list_of_bytes.append(bytes([b]))

        # TODO: could refactor into new function
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

    def decode(self, ids: list[int]) -> str:
        bytes = b""
        for id in ids:
            bytes += self.vocab.get(id)
        
        return bytes.decode("utf-8", errors="replace")
    