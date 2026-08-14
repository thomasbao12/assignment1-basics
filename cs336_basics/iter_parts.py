from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import regex

def bytes_to_list(x: bytes) -> list[bytes]:
    return [bytes([b]) for b in x]

@dataclass(frozen=True)
class NormalPart:
    data: list[bytes]


@dataclass(frozen=True)
class SpecialPart:
    data: list[bytes]


Part = NormalPart | SpecialPart


PAT = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

def pretokenize(data: bytes) -> Iterator[NormalPart]:
    text = data.decode("utf-8")

    for match in PAT.finditer(text):
        yield NormalPart(bytes_to_list(match.group().encode("utf-8")))


def iter_parts(
    input_path: str | Path,
    special_tokens: list[str],
    chunk_size: int = 1024 * 1024,
) -> Iterator[Part]:
    special_token_bytes = [
        token.encode("utf-8")
        for token in special_tokens
    ]

    with open(input_path, "rb") as f:
        buffer = b""

        while chunk := f.read(chunk_size):
            buffer += chunk

            while True:
                match = _find_next_special_token(
                    buffer,
                    special_token_bytes,
                )

                if match is None:
                    break

                position, special_token = match

                if position > 0:
                    yield from pretokenize(buffer[:position])

                yield SpecialPart([special_token])

                buffer = buffer[
                    position + len(special_token):
                ]

        if buffer:
            yield from pretokenize(buffer)


def _find_next_special_token(
    data: bytes,
    special_tokens: list[bytes],
) -> tuple[int, bytes] | None:
    """
    Return (position, token) for the earliest special token in data.

    If multiple special tokens begin at the same position, prefer the longest.
    """
    best_position: int | None = None
    best_token: bytes | None = None

    for token in special_tokens:
        position = data.find(token)

        if position == -1:
            continue

        if (
            best_position is None
            or position < best_position
            or (
                position == best_position
                and best_token is not None
                and len(token) > len(best_token)
            )
        ):
            best_position = position
            best_token = token

    if best_position is None or best_token is None:
        return None

    return best_position, best_token


if __name__ == "__main__":
    test_path = "/tmp/test_parts.txt"
    
    test_data = (
        b"hello world"
        b"<|endoftext|>"
        b"this is story two"
        b"<|endoftext|>"
        b"final story"
    )

    with open(test_path, "wb") as f:
        f.write(test_data)

    parts = list(
        iter_parts(
            test_path,
            special_tokens=["<|endoftext|>"],
            chunk_size=5,  # deliberately tiny
        )
    )

    for part in parts:
        print(part)

    expected = [
        NormalPart(bytes_to_list(b"hello")),
        NormalPart(bytes_to_list(b" world")),
        SpecialPart([b"<|endoftext|>"]),
        NormalPart(bytes_to_list(b"this")),
        NormalPart(bytes_to_list(b" is")),
        NormalPart(bytes_to_list(b" story")),
        NormalPart(bytes_to_list(b" two")),
        SpecialPart([b"<|endoftext|>"]),
        NormalPart(bytes_to_list(b"final")),
        NormalPart(bytes_to_list(b" story")),
    ]

    assert parts == expected
    print("All tests passed!")