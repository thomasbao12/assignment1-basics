import cProfile
import pstats

from collections import Counter
from collections.abc import Iterable
from cs336_basics.bpe import BPE, WeightedPart
from cs336_basics.iter_parts import iter_parts, Part, NormalPart, SpecialPart
from dataclasses import dataclass


def compress_parts(parts: Iterable[Part]) -> list[WeightedPart]:
    counts = Counter(
        tuple(
            [bytes([b]) for b in part.data]
        ) if isinstance(part, NormalPart) else tuple([part.data])
        for part in parts
    )

    return [
        WeightedPart(
            data=data,
            count=count,
        )
        for data, count in counts.items()
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

    parts_iterator = iter_parts(
        input_path,
        special_tokens,
    )

    compressed_parts = compress_parts(parts_iterator)
    bpe = BPE(
        vocab,
        compressed_parts,
    )
    
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