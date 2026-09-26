#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Decode a logic-analyser capture of one printed column back into head data.

WHY THIS EXISTS
    Everything the encoder does to a bus (two 180-bit code blocks, then 32
    window bits, sampled on both clock edges) was inferred by probing the
    original controller of a printer we own with an oscilloscope and a logic
    analyser, and confirmed only indirectly, by whether ink came out. A clean
    capture of that controller driving the same head settles it directly, and
    settles it in the one direction that matters: it can say the model is WRONG.

    So this is not a convenience script. It is the check the wire model is
    allowed to fail, and it should be re-run against any new capture before a
    capture is quoted as evidence for anything.

WHAT A CAPTURE HAS TO LOOK LIKE
    A Saleae-style CSV of transitions: one row per change, not per sample:

        Time [s],Channel 0,Channel 1,Channel 2

    with the data buses on channels 0..n-2 and the shift clock last. Data is
    sampled on BOTH clock edges, holding the value present just before each
    edge, because that is the value the head latches.

WHAT IT REPORTS
    Per bus: the two code blocks and the window map, the histogram of the four
    2-bit codes, and which 60-nozzle blocks are live. Plus the frame's timing
    and, if the capture is 392 edges long, a check that the block boundaries
    fall where the encoder puts them.
"""

from __future__ import annotations

import argparse
import collections
import csv
import statistics
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "encoder"))

import head_packer as hp                 # noqa: E402

# A code block is BUS_POSITIONS bits; two of them then the window map.
BLOCK_BOUNDARIES = (hp.BUS_POSITIONS, 2 * hp.BUS_POSITIONS)
NOZZLE_BLOCK = 60          # the colour column's three stacked channels


def parse_timestamp(text: str) -> int:
    """One CSV timestamp as nanoseconds. Absolute or relative, either works."""
    if "T" in text:
        clock = text.split("T")[1].split("+")[0].rstrip("Z")
        hours, minutes, seconds = clock.split(":")
        return round((int(hours) * 3600 + int(minutes) * 60
                      + float(seconds)) * 1e9)
    return round(float(text) * 1e9)


def read_transitions(path: Path) -> Tuple[List[int], List[List[int]]]:
    """(times in ns, per-channel values) for every row of a transition CSV."""
    with path.open() as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [row for row in reader if row]
    if len(header) < 3:
        raise SystemExit(f"{path}: need at least one data channel and a clock")
    times = [parse_timestamp(row[0]) for row in rows]
    values = [[int(v) for v in row[1:]] for row in rows]
    zero = times[0]
    return [t - zero for t in times], values


def sample_on_edges(times: Sequence[int], values: Sequence[Sequence[int]],
                    clock: int) -> Tuple[List[int], List[List[int]]]:
    """Data as the head sees it: the value HELD when each clock edge arrives.

    A transition list gives the value *after* each change, and the data lines
    move a few tens of nanoseconds before the edge that latches them. So the
    sample is the state carried into the edge, not the state the edge's own row
    reports; reading the latter shifts every bus by one bit and produces a
    stream that decodes to plausible nonsense.
    """
    held = [0] * len(values[0])
    edge_times: List[int] = []
    samples: List[List[int]] = []
    for time, row in zip(times, values):
        if row[clock] != held[clock]:
            edge_times.append(time)
            samples.append([held[c] for c in range(len(row)) if c != clock])
        held = list(row)
    return edge_times, [list(column) for column in zip(*samples)] if samples \
        else []


def report_timing(edge_times: Sequence[int]) -> List[List[int]]:
    """Print the frame's timing and return the edges split into bursts."""
    gaps = [edge_times[i + 1] - edge_times[i]
            for i in range(len(edge_times) - 1)]
    tight = [g for g in gaps if g < 4 * statistics.median(gaps)]
    period = 2 * round(statistics.median(tight))
    print(f"{len(edge_times)} clock edges, "
          f"{sum(1 for i in range(len(edge_times)) if i % 2 == 0)} clocks, "
          f"period ~{period} ns ({1e3 / period:.1f} MHz), "
          f"column {edge_times[-1] - edge_times[0]} ns")

    bursts: List[List[int]] = [[]]
    for index, time in enumerate(edge_times):
        if index and gaps[index - 1] > 4 * statistics.median(tight):
            print(f"  pause {gaps[index - 1]} ns before edge {index}")
            bursts.append([])
        bursts[-1].append(time)
    return bursts


def check_frame(bursts: Sequence[Sequence[int]], edges: int) -> None:
    """Do the capture's own pauses fall where the encoder puts block edges?"""
    if edges != hp.EDGE_COUNT:
        print(f"\nNOTE: {edges} edges per bus, not the {hp.EDGE_COUNT} this "
              f"head's frame has. Nothing below is checked against the model.")
        return
    lengths = [len(b) for b in bursts]
    expected = [hp.BUS_POSITIONS, hp.BUS_POSITIONS, hp.WINDOW_BITS]
    if lengths == expected:
        print(f"\nFRAME CONFIRMED: bursts of {lengths} match the encoder's "
              f"blocks exactly: two code blocks of {hp.BUS_POSITIONS} then "
              f"{hp.WINDOW_BITS} window bits, with the boundaries visible as "
              f"pauses on the wire.")
    elif len(bursts) == 1:
        print(f"\nOne continuous burst of {lengths[0]}: the block boundaries "
              f"are not marked by pauses here, so they are assumed, not seen.")
    else:
        print(f"\nFRAME MISMATCH: bursts of {lengths}, expected {expected}. "
              f"The encoder splits this frame somewhere the head does not.")


def report_bus(name: str, bits: Sequence[int]) -> None:
    if len(bits) != hp.EDGE_COUNT:
        print(f"\n=== {name} === {len(bits)} bits, cannot decode as a "
              f"{hp.EDGE_COUNT}-bit bus")
        return
    first = bits[:hp.BUS_POSITIONS]
    second = bits[hp.BUS_POSITIONS:2 * hp.BUS_POSITIONS]
    window = "".join(str(b) for b in bits[2 * hp.BUS_POSITIONS:])
    codes = [f"{first[p]}{second[p]}" for p in range(hp.BUS_POSITIONS)]
    live = [p for p, code in enumerate(codes) if code != "00"]

    print(f"\n=== {name} ===")
    print(f"  window map : {window}")
    try:
        import gen_channel_masks as gcm
        print(f"               {gcm.describe_map(window)}")
    except Exception:                                    # noqa: BLE001
        pass
    counts = collections.Counter(codes)
    print("  codes      : " + "  ".join(f"{c}x{counts[c]}"
                                        for c in ("00", "01", "10", "11")))
    if not live:
        print("  nothing fires on this bus in this column")
        return
    print(f"  positions  : {min(live)}..{max(live)} "
          f"({len(live)} of {hp.BUS_POSITIONS} non-blank)")
    for block in range(hp.BUS_POSITIONS // NOZZLE_BLOCK):
        segment = codes[block * NOZZLE_BLOCK:(block + 1) * NOZZLE_BLOCK]
        counts = collections.Counter(segment)
        state = "idle" if counts["00"] == NOZZLE_BLOCK else "  " + " ".join(
            f"{c}x{counts[c]}" for c in ("01", "10", "11") if counts[c])
        print(f"    block {block} (pos {block * NOZZLE_BLOCK:3d}-"
              f"{block * NOZZLE_BLOCK + NOZZLE_BLOCK - 1:3d}): {state}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Decode a logic-analyser capture of one printed column.")
    parser.add_argument("capture", type=Path,
                        help="transition CSV: time, data channels, clock")
    parser.add_argument("--clock", type=int, default=-1,
                        help="which data column is the shift clock "
                             "(default: the last one)")
    parser.add_argument("--names", default="",
                        help="comma-separated bus names, in channel order "
                             "(e.g. black,colour)")
    args = parser.parse_args(argv)

    times, values = read_transitions(args.capture)
    clock = args.clock % len(values[0])
    edge_times, buses = sample_on_edges(times, values, clock)
    if not edge_times:
        raise SystemExit(f"{args.capture}: no transitions on channel {clock}; "
                         f"is that the clock?")

    bursts = report_timing(edge_times)
    check_frame(bursts, len(edge_times))

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    for index, bits in enumerate(buses):
        label = names[index] if index < len(names) else f"channel {index}"
        report_bus(f"{label} (capture channel {index})", bits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
