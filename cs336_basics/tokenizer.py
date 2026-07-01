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
            self.bytes_to_id[id] = b

    def encode(self, text: str) -> list[int]:
        byte_str: bytes = text.encode("utf-8")
        list_of_bytes: list[bytes] = [bytes([b]) for b in byte_str]
        new_list_of_bytes: list[bytes] = []

        # TODO: could refactor into new function
        for x, y in self.merges:
            i = 0
            n = len(list_of_bytes)
            # TODO: could refactor into new function
            while i < n:
                a, b = byte_str[i], byte_str[i + 1]
                if (x, y) == (a, b):
                    new_list_of_bytes.append(
                        a + b
                    )
                    i += 2
                else:
                    new_list_of_bytes.append(a)
                    i += 1
                if i == n - 1:
                    new_list_of_bytes.append(b)
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
            bytes += self.vocab[id]
        return bytes.decode("utf-8")