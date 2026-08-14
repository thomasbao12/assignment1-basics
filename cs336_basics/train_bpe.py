from collections import defaultdict
from collections.abc import Iterator
from cs336_basics.iter_parts import Part, NormalPart, iter_parts

import cProfile
import pstats


class BPE:
    DEBUG = False
    PROFILE = True

    """
    invariants:
    - bytes to positions and position to bytes are inverse dictionaries
    - we should be able to reconstruct each part from the corresponding
      position_to_bytes
    - byte counts individually of bytes_pair_to_counts should equal the
      byte counts of parts
    """

    def _assert_position_to_bytes_is_equivalent_to_parts(self):
        for i, part in enumerate(self.parts):
            part = part.data

            reconstructed_part = []
            sorted_items = sorted(self.position_to_bytes[i].items())

            for _, value in sorted_items:
                reconstructed_part.append(value)

            assert b"".join(part) == b"".join(reconstructed_part), (
                f"{part} != {reconstructed_part}"
            )

            for j in range(len(sorted_items) - 1):
                (index, byte_str) = sorted_items[j]
                next_index, _ = sorted_items[j + 1]

                assert index + len(byte_str) == next_index

    def _assert_position_to_bytes_equivalent_to_bytes_to_position(self):
        for b, positions in self.bytes_to_positions.items():
            for i, j in positions:
                assert self.position_to_bytes[i][j] == b

        for i, v in self.position_to_bytes.items():
            for j, b in v.items():
                assert (i, j) in self.bytes_to_positions[b]

    def _assert_merge_is_merged(
        self,
        merge_pair: tuple[bytes, bytes],
    ):
        for i in self.position_to_bytes:
            sorted_items = sorted(self.position_to_bytes[i].items())

            for j in range(len(sorted_items) - 1):
                (_, b1), (_, b2) = sorted_items[j], sorted_items[j + 1]

                assert (b1, b2) != merge_pair, (
                    f"i={i}: {sorted_items}"
                )

    def _assert_byte_pair_counts_are_accurate(self):
        byte_pair_counts = defaultdict(int)

        for i in self.position_to_bytes:
            sorted_items = sorted(self.position_to_bytes[i].items())

            for j in range(len(sorted_items) - 1):
                byte_pair = (
                    sorted_items[j][1],
                    sorted_items[j + 1][1],
                )
                byte_pair_counts[byte_pair] += 1

        for k, vleft in byte_pair_counts.items():
            vright = self.bytes_pair_to_counts[k]

            assert vleft == vright, f"""
                k: {k} vleft: {vleft} vright: {vright}
            """.strip()

        for k, v in self.bytes_pair_to_counts.items():
            assert v == byte_pair_counts[k]

    def __init__(
        self,
        vocab: dict[int, bytes] = {
            i: bytes([i])
            for i in range(256)
        },
        parts: Iterator[Part] = [],
    ):
        self.parts = parts
        self.merged = []
        self.vocab = vocab

        self.bytes_to_positions: dict[
            bytes,
            set[tuple[int, int]],
        ] = defaultdict(set)

        # first index is part
        # second index is byte offset inside part
        self.position_to_bytes: dict[
            int,
            dict[int, bytes],
        ] = defaultdict(dict)

        self.bytes_pair_to_counts: dict[
            tuple[bytes, bytes],
            int,
        ] = defaultdict(int)

        # (x, y) -> positions where x is immediately followed by y.
        #
        # Each position is (part_index, byte_offset_of_x).
        self.pair_to_positions: dict[
            tuple[bytes, bytes],
            set[tuple[int, int]],
        ] = defaultdict(set)

        for i, part in enumerate(self.parts):
            list_of_bytes = part.data

            for j in range(len(list_of_bytes) - 1):
                x, y = list_of_bytes[j], list_of_bytes[j + 1]

                self.bytes_pair_to_counts[(x, y)] += 1
                self.pair_to_positions[(x, y)].add((i, j))

                self.bytes_to_positions[x].add((i, j))
                self.position_to_bytes[i][j] = x

            if len(list_of_bytes) > 0:
                j = len(list_of_bytes) - 1
                b = list_of_bytes[j]

                self.bytes_to_positions[b].add((i, j))
                self.position_to_bytes[i][j] = b

        if BPE.DEBUG:
            print("init: about to assert")

            self._assert_position_to_bytes_equivalent_to_bytes_to_position()
            self._assert_position_to_bytes_is_equivalent_to_parts()
            self._assert_byte_pair_counts_are_accurate()

            print("init: passed asserts")

    def _find_previous_position(
        self,
        i: int,
        j: int,
    ) -> int:
        valid_positions = self.position_to_bytes[i].keys()

        h = j - 1

        while h > 0 and h not in valid_positions:
            h -= 1

        return h

    def _get_merge_positions(
        self,
        x: bytes,
        y: bytes,
    ) -> list[
        tuple[
            tuple[int, int],
            tuple[int, int],
        ]
    ]:
        positions = self.pair_to_positions.get((x, y), ())

        # If x != y, occurrences of (x, y) cannot overlap.
        if x != y:
            return [
                (
                    (i, j),
                    (i, j + len(x)),
                )
                for i, j in positions
            ]

        # If x == y, pair occurrences can overlap.
        #
        # Example:
        #
        # x x x
        # ^ ^
        #   ^ ^
        #
        # We preserve left-to-right greedy merging.
        merge_positions = []

        last_merge_boundary = (-1, -1)

        for i, j in sorted(positions):
            if (i, j) == last_merge_boundary:
                continue

            k = j + len(x)

            merge_positions.append(
                (
                    (i, j),
                    (i, k),
                )
            )

            last_merge_boundary = (i, k)

        return merge_positions

    def merge_dry_run(self) -> bool:
        # TODO use sorted dict
        max_count, max_merge = max(
            [
                (count, pair)
                for pair, count in self.bytes_pair_to_counts.items()
            ]
        )

        if max_count == 0:
            return False

        x, y = max_merge
        merged_bytes = x + y

        self.vocab[len(self.vocab)] = merged_bytes
        self.merged.append(max_merge)

        merge_positions = self._get_merge_positions(x, y)

        # ------------------------------------------------------------
        # Remove old adjacent-pair bookkeeping around each merge.
        #
        # Before:
        #
        #     A X Y B
        #
        # We need to remove:
        #
        #     (A, X)
        #     (Y, B)
        #
        # (X, Y) is removed globally below.
        # ------------------------------------------------------------

        pairs_of_positions = set()

        for (i, j), (_, k) in merge_positions:
            # Pair after Y.
            l = k + len(y)

            if l in self.position_to_bytes[i]:
                pairs_of_positions.add(
                    (
                        (i, k),
                        (i, l),
                    )
                )

            # Pair before X.
            h = self._find_previous_position(i, j)

            if h >= 0:
                pairs_of_positions.add(
                    (
                        (i, h),
                        (i, j),
                    )
                )

        for (i1, j1), (i2, j2) in pairs_of_positions:
            a = self.position_to_bytes[i1][j1]
            b = self.position_to_bytes[i2][j2]

            pair = (a, b)

            self.bytes_pair_to_counts[pair] -= 1
            self.pair_to_positions[pair].remove((i1, j1))

        # ------------------------------------------------------------
        # Remove the merged pair itself.
        #
        # All selected (x, y) occurrences disappear.
        # ------------------------------------------------------------

        self.bytes_pair_to_counts.pop((x, y), None)
        self.pair_to_positions.pop((x, y), None)

        # ------------------------------------------------------------
        # Perform the actual merges.
        # ------------------------------------------------------------

        for (i, j), (_, k) in merge_positions:
            self.bytes_to_positions[x].remove((i, j))
            self.bytes_to_positions[y].remove((i, k))

            self.bytes_to_positions[merged_bytes].add((i, j))

            self.position_to_bytes[i][j] = merged_bytes
            del self.position_to_bytes[i][k]

        # ------------------------------------------------------------
        # Add new adjacent-pair bookkeeping.
        #
        # After:
        #
        #     A XY B
        #
        # Add:
        #
        #     (A, XY)
        #     (XY, B)
        #
        # Only newly merged positions can have changed adjacency.
        # ------------------------------------------------------------

        pairs_of_positions = set()

        for (i, j), _ in merge_positions:
            # Pair after merged token.
            k = j + len(merged_bytes)

            if k in self.position_to_bytes[i]:
                pairs_of_positions.add(
                    (
                        (i, j),
                        (i, k),
                    )
                )

            # Pair before merged token.
            h = self._find_previous_position(i, j)

            if h >= 0:
                pairs_of_positions.add(
                    (
                        (i, h),
                        (i, j),
                    )
                )

        for (i1, j1), (i2, j2) in pairs_of_positions:
            pair = (
                self.position_to_bytes[i1][j1],
                self.position_to_bytes[i2][j2],
            )

            self.bytes_pair_to_counts[pair] += 1
            self.pair_to_positions[pair].add((i1, j1))

        if BPE.DEBUG:
            print("merge_dry_run: about to assert")

            self._assert_position_to_bytes_is_equivalent_to_parts()
            self._assert_position_to_bytes_equivalent_to_bytes_to_position()
            self._assert_merge_is_merged(max_merge)
            self._assert_byte_pair_counts_are_accurate()

            print("merge_dry_run: passed assert")

        return True

    def __repr__(self):
        return str(
            {
                "merged": self.merged,
                "vocab_size": len(self.vocab),
                "bytes_pair_to_counts": self.bytes_pair_to_counts,
                "bytes_to_positions": self.bytes_to_positions,
                "pair_to_positions": self.pair_to_positions,
                "position_to_bytes": self.position_to_bytes,
            }
        )


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[
    dict[int, bytes],
    list[tuple[bytes, bytes]],
]:
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

    if BPE.PROFILE:
        profiler = cProfile.Profile()
        profiler.enable()

    bpe = BPE(
        vocab,
        parts_iterator,
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
        NormalPart(
            [
                b"1",
                b"1",
                b"1",
                b"1",
                b"1",
                b"1",
                b"1",
            ]
        ),
    ].__iter__()

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