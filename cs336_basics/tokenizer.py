class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes,bytes]],
        special_tokens: list[str] | None
    ): 
        self.vocab = vocab 
        self.merges = merges 
        self.special_tokens = special_tokens

        self.bytes_to_id = {}
        for id, b in vocab.items():
            self.bytes_to_id[b] = id

    def encode(self, text: str) -> list[int]:
        _byte_str: bytes = text.encode("utf-8")
        list_of_bytes: list[bytes] = [bytes([b]) for b in _byte_str]
        
        # TODO: could refactor into new function
        j = 0
        for x, y in self.merges:
            new_list_of_bytes: list[bytes] = []
            j += 1
            i = 0
            n = len(list_of_bytes)
            # TODO: could refactor into new function
            while i < n:
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
        return bytes.decode("utf-8")