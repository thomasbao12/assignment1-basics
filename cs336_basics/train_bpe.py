import cProfile
import pstats

from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from cs336_basics.bpe import BPE, WeightedPart
from cs336_basics.iter_parts import iter_parts, Part, NormalPart, SpecialPart
from cs336_basics.pretokenization_example import find_chunk_boundaries
from dataclasses import dataclass
from pathlib import Path 


def count_range(
    input_path: str | Path,
    special_tokens: list[str],
    start: int,
    end: int,
) -> Counter[tuple[bytes, bool]]:
    return Counter(
        (part.data, isinstance(part, NormalPart))
        for part in iter_parts(
            input_path,
            special_tokens,
            start=start,
            end=end,
        )
    )

def _count_parts_parallel(
    input_path: str | Path,
    special_tokens: list[str],
    split_special_token: bytes,
    num_workers: int,
) -> Counter[tuple[bytes, bool]]:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=num_workers,
            split_special_token=split_special_token,
        )

    ranges = list(
        zip(boundaries[:-1], boundaries[1:])
    )

    total: Counter[tuple[bytes, bool]] = Counter()

    with ProcessPoolExecutor(
        max_workers=num_workers
    ) as executor:
        futures = [
            executor.submit(
                count_range,
                input_path,
                special_tokens,
                start,
                end,
            )
            for start, end in ranges
        ]

        for future in futures:
            total.update(future.result())

    return total

def count_parts_parallel(
    input_path: str | Path,
    special_tokens: list[str] = ["<|endoftext|>"],
    split_special_token: bytes = b"<|endoftext|>",
    num_workers: int = 8,
) -> list[WeightedPart]:
    counts = _count_parts_parallel(input_path, special_tokens, split_special_token, num_workers) 
    return [
        WeightedPart(
            data=tuple(
                [bytes([b]) for b in data]
            ) if isNormalPart else tuple([data]),
            count=count,
        )
        for (data, isNormalPart), count in counts.items()
    ]


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[
    dict[int, bytes],
    list[tuple[bytes, bytes]],
]:
    if BPE.PROFILE:
        profiler = cProfile.Profile()
        profiler.enable()

    vocab = {
        i: bytes([i])
        for i in range(256)
    }

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    compressed_parts = count_parts_parallel(
        input_path,
        special_tokens
    )

    bpe = BPE(vocab, compressed_parts)

    while len(bpe.vocab) < vocab_size:
        did_merge = bpe.merge_dry_run()

        if not did_merge:
            break

    if BPE.PROFILE:
        profiler.disable()

        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        stats.print_stats(20)

    return (
        bpe.vocab,
        bpe.merged,
    )


if __name__ == "__main__":
    BPE.DEBUG = True

    parts_iterator = [
        WeightedPart(
            tuple([
                b"1",
                b"1",
                b"1",
                b"1",
                b"1",
                b"1",
                b"1",
            ]),
            1
        ),
    ]

    vocab = {
        i: bytes([i])
        for i in range(256)
    }

    bpe = BPE(
        vocab,
        parts_iterator,
    )

    bpe.merge_dry_run()
    bpe.merge_dry_run()
    bpe.merge_dry_run()
    bpe.merge_dry_run()