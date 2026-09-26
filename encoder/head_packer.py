# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""How a head's nozzles are laid out on the wire.

The firmware shifts out a fixed frame: ``clock_count`` clocks of
``frame_groups`` data pins, sampled on both edges, packed contiguously into
``bytes_per_line``. What changes between heads is only what rides those pins.

A **bus** is one data pin's worth of head: 180 nozzle positions, each carrying a
2-bit code, followed by a 32-bit window-enable map: 392 bits, clocked out two
bits per clock. The bit order inside a bus is *not* code-by-code; it is two
blocks of 180:

    index   0 .... 179 | 180 .... 359 | 360 ... 391
            first bit  | fire bit     | window-enable
            of the code| of the code  | map
            positions 0..179 in both blocks

So bus position ``p`` has its code's first bit at index ``p`` and its fire bit
at ``180 + p``. Only codes 00 and 01 are emitted today, so the first block is
always zero.

Everything head-specific therefore collapses to one table per bus: which
physical nozzle feeds each of the 180 positions. On c6n90 a bus carries *two*
90-nozzle columns interleaved, which is what makes a bus a "group" there; on a
head whose bus serves a single 180-nozzle column the table is a straight run.
The two tables differ; the packing code does not.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Every head's scalars are generated from paintress-protocol's profiles and
# vendored in; only the per-bus tables live here (see the module docstring).
import head_profiles

# Nozzle positions on one bus, and the two blocks that hold their code bits.
BUS_POSITIONS = 180
# Bits of window-enable map appended to every bus.
WINDOW_BITS = 32
EDGE_COUNT = 2 * BUS_POSITIONS + WINDOW_BITS      # 392

# A bus position that no nozzle feeds.
UNUSED = None


@dataclass(frozen=True)
class BusSpec:
    """One data pin's worth of head.

    ``frame_group`` is which of the firmware's data pins carries it.
    ``positions`` has BUS_POSITIONS entries, each ``(slot, nozzle)`` naming the
    nozzle that fires from that position (``nozzle`` indexed *within its
    slot*, which is how the payload planes are indexed too), or None where the
    bus has no nozzle wired.
    """

    frame_group: int
    positions: Tuple[Optional[Tuple[int, int]], ...]

    def __post_init__(self):
        if len(self.positions) != BUS_POSITIONS:
            raise ValueError(
                f"a bus has {BUS_POSITIONS} positions, got {len(self.positions)}")
        wired = [(p, s) for p, s in enumerate(self.positions) if s is not None]
        if len(wired) != len({s for _, s in wired}):
            raise ValueError(
                "a bus fires the same nozzle from two positions")
        # Index arrays for the packer, built once: position, slot, nozzle.
        object.__setattr__(self, "_index", (
            np.array([p for p, _ in wired], dtype=np.intp),
            np.array([s[0] for _, s in wired], dtype=np.intp),
            np.array([s[1] for _, s in wired], dtype=np.intp),
        ))


@dataclass(frozen=True)
class HeadPacker:
    """A head's wire layout: its buses, its window map, and the frame they ride.

    ``window_map`` is a 32-character bit string saying which 2-bit codes fire in
    which window after the latch: an electrical property of the head, not of
    the packer. ``frame_groups`` is how many data pins the firmware drives; a
    head that uses fewer leaves the rest at zero (they are not wired), which is
    what lets a two-bus head ride the three-bus frame unchanged.
    """

    name: str
    buses: Tuple[BusSpec, ...]
    window_map: str
    slot_count: int
    frame_groups: int = 3
    clock_count: int = 196
    bytes_per_line: int = 147

    def __post_init__(self):
        if len(self.window_map) != WINDOW_BITS:
            raise ValueError(
                f"head {self.name}: window map must be {WINDOW_BITS} bits, "
                f"got {len(self.window_map)}")
        if set(self.window_map) - {"0", "1"}:
            raise ValueError(
                f"head {self.name}: window map must be 0s and 1s")
        if len(self.buses) > self.frame_groups:
            raise ValueError(
                f"head {self.name}: {len(self.buses)} buses do not fit in "
                f"{self.frame_groups} frame groups")
        groups = [bus.frame_group for bus in self.buses]
        if len(set(groups)) != len(groups):
            raise ValueError(f"head {self.name}: two buses on one frame group")
        if any(not 0 <= g < self.frame_groups for g in groups):
            raise ValueError(f"head {self.name}: frame group out of range")
        expected = (self.clock_count * self.bits_per_clock + 7) // 8
        if expected != self.bytes_per_line:
            raise ValueError(
                f"head {self.name}: {self.clock_count} clocks x "
                f"{self.bits_per_clock} bits is {expected} bytes, but the "
                f"profile says {self.bytes_per_line}")

    @property
    def bits_per_clock(self) -> int:
        """Data pins sampled on both edges."""
        return 2 * self.frame_groups

    @property
    def idle_frame_groups(self) -> Tuple[int, ...]:
        """Frame groups no bus rides: shifted out as zeros, wired to nothing."""
        used = {bus.frame_group for bus in self.buses}
        return tuple(g for g in range(self.frame_groups) if g not in used)

    @property
    def wired_nozzles(self) -> int:
        return sum(1 for bus in self.buses for p in bus.positions if p is not None)

    def describe(self) -> str:
        idle = self.idle_frame_groups
        return (f"{len(self.buses)} bus(es) x {BUS_POSITIONS} positions, "
                f"{self.wired_nozzles} nozzles wired, "
                f"{self.bytes_per_line} B/line"
                + (f", frame group(s) {', '.join(map(str, idle))} idle" if idle else ""))

    # -- packing -----------------------------------------------------------

    def encode_pass(self, slot_data: np.ndarray) -> List[bytes]:
        """Pack one pass, ``(slot_count, nozzles, width)``, into column lines.

        Row ``s`` of ``slot_data`` is the bitmap slot ``s`` fires, in physical
        slot order. Returns one ``bytes_per_line`` line per printed column.
        """
        slot_data = np.ascontiguousarray(slot_data, dtype=np.uint8)
        slots, nozzles, width = slot_data.shape
        if slots != self.slot_count:
            raise ValueError(
                f"head {self.name} has {self.slot_count} slots, got {slots}")

        window = np.array([int(c) for c in self.window_map], dtype=np.uint8)
        clock_bits = np.zeros(
            (self.clock_count, self.bits_per_clock, width), dtype=np.uint8)

        for bus in self.buses:
            position_idx, slot_idx, nozzle_idx = bus._index
            if nozzle_idx.size and int(nozzle_idx.max()) >= nozzles:
                raise ValueError(
                    f"head {self.name} wires nozzle {int(nozzle_idx.max())} of a "
                    f"slot, but the pass carries only {nozzles} rows per slot")
            stream = np.zeros((EDGE_COUNT, width), dtype=np.uint8)
            # Block 1 (indices 0..179) is the codes' first bit: always 0 while
            # only 00/01 are emitted. Block 2 is the fire bits.
            stream[BUS_POSITIONS + position_idx] = slot_data[slot_idx, nozzle_idx]
            stream[2 * BUS_POSITIONS:] = window[:, None]

            group = bus.frame_group
            clock_bits[:, group] = stream[0::2]                      # rise edge
            clock_bits[:, group + self.frame_groups] = stream[1::2]  # fall edge

        bit_stream = clock_bits.reshape(
            self.clock_count * self.bits_per_clock, width)
        packed = np.packbits(bit_stream, axis=0, bitorder="little")
        return [column.tobytes() for column in packed.T]


# ---------------------------------------------------------------------------
# The shipped heads
# ---------------------------------------------------------------------------

# c6n90: three buses, each carrying two 90-nozzle columns interleaved. Position
# 2n is the group's odd (right-hand) slot, 2n+1 the even one; the interleave
# is what binds two columns into one bus.
def _c6n90_bus(group: int) -> BusSpec:
    positions: List[Optional[Tuple[int, int]]] = []
    for n in range(BUS_POSITIONS // 2):
        positions.append((2 * group + 1, n))   # odd slot: the primary lane
        positions.append((2 * group, n))       # even slot: the secondary lane
    return BusSpec(frame_group=group, positions=tuple(positions))


# Enables only code 01 in the first window after the latch, so nozzles set to 01
# fire and nozzles set to 00 do not: the single-drop-size mode. Variable drop
# sizes would enable the other codes (10/11) in other windows.
#
# A head's definition cannot be read off the vendored profile, because the
# profile describes whichever head *this build targets*, and both heads have
# to stay describable regardless of which one that is. So each head's identity
# and scalars are written here, and `_check_against_profile` asserts that the
# definition of the targeted head agrees with what the firmware was built
# against. Drift between the two is the failure the profile exists to prevent.
C6N90_WINDOW_MAP = head_profiles.head("c6n90").window_enable_map

# The frame is the firmware's, not a head's: every head rides 196 clocks of 6
# bits into 147 bytes, which is exactly why one firmware build serves them all.
FRAME_GROUPS = head_profiles.FRAME_DATA_PINS
CLOCK_COUNT = head_profiles.CLOCK_COUNT
BYTES_PER_LINE = head_profiles.BYTES_PER_LINE

C6N90 = HeadPacker(
    name="c6n90",
    buses=tuple(_c6n90_bus(g) for g in range(
        head_profiles.head("c6n90").data_buses)),
    window_map=C6N90_WINDOW_MAP,
    slot_count=head_profiles.head("c6n90").channel_count,
    frame_groups=FRAME_GROUPS,
    clock_count=CLOCK_COUNT,
    bytes_per_line=BYTES_PER_LINE,
)


# c4n180: two buses, each carrying one 180-nozzle column. The head has no third
# data bus, so frame group 2 is shifted out as zeros and left unwired, which
# is what keeps the 147-byte frame, and the firmware, unchanged.
#
# BUS ORDER: specified, not guessed. Each bus serves a single column and its
# 180 positions run straight down it; c6n90's interleave exists only because a
# bus there serves *two* columns. On the colour bus both 180-bit blocks split
# into three contiguous runs of 60, one per channel: positions 0..59, 60..119
# and 120..179 in the first-bit block, and the same thirds again at 180+p in the
# fire-bit block. The black bus wires only the 60 positions its plumbed third
# occupies; the other 120 are shifted out dark.
#
# Which colour sits in which third is a plumbing question, not a wiring one:
# it follows the head layout's slots, so a head whose nozzle axis runs the other
# way is corrected with --ink-map (K,Y,M,C), never by re-ordering the bus.
#
# WINDOW MAP: bit 22, DERIVED 2026-07-28 from a logic-analyser capture of one
# column of a real print of this head, not yet confirmed by ejection.
#
# THE CAPTURE CONFIRMS THE FRAME ITSELF, which is the part that was inference
# until now: 392 clock edges over 196 clocks, in three bursts of 180 / 180 / 32
# separated by 4 us pauses. The block boundaries this file assumes are visible on
# the wire, and the 32-bit tail is identical on both buses. Decode any new
# capture with tools/decode_capture.py before quoting it.
#
# THE HEAD IS GREYSCALE. One real column uses all four codes (01x35 10x10 11x2
# on black, 01x14 10x6 11x28 on colour), so the three windows the dumped map
# lights are all genuinely needed. This is not a binary head whose map happens to
# carry spare bits.
#
# We print binary, so we need only code 01's bit, and DROP SIZE BECOMES A CHOICE:
#
#     {22}          w5           1 pulse   (what the original drives 01 with)
#     {18,22}       w4,w5        2 pulses
#     {18,22,26}    w4,w5,w6     3 pulses  (code 11's own ladder)
#
# all with code 00's bits (3,7,11,15,19,23,27,31) clear, which is what keeps the
# white white. If bit 22 under-inks, move up that list rather than changing
# anything in the encoder.
#
# THE 32 BITS ARE 8 WINDOWS x 4 CODES: a nibble per firing window, and inside it
# one bit per 2-bit nozzle code, in the order 11, 10, 01, 00. The dumped value
# 00000000000000001100101011000001 therefore reads as
#
#     code 11 -> w4, w5, w6      code 01 -> w5
#     code 10 -> w4, w6          code 00 -> w7, alone
#
# a greyscale drop ladder: large, medium, small, and a lone pulse for the
# blanks that nothing else shares, which is the non-ejecting tickle that keeps
# an idle meniscus moving. Monotonic in pulse count, as a drop ladder must be
# and as no other grouping of these 32 bits comes out.
#
# The cross-check the model did not get to pick: c6n90 sets bit 14 and nothing
# else, and 14 is window 3, slot 2 = code 01. One code in one window, exactly
# how that head prints. A model fitted to one head's dump lands on the other
# head's shipped value.
#
# A print only emits 01, so it only needs that code's bit: window 5, slot 2,
# index 22.
#
# The same reading explains the 2026-07-27 failure. {30,31} is window 7 slots 2
# AND 3: codes 01 and 00 firing in one window. The print put ink where the
# image was blank, which is what the bench saw. That map was not half right; it
# enabled the blanks on purpose.
#
# DISCARDED: the whole 2026-07-27 measurement, i.e. {30,31}, "bit 14 refuted", and
# the theory that the bits address 60-nozzle blocks. That head was not in a fit
# state to test, so none of those purges separates "the map is wrong" from "that
# nozzle was not ejecting".
C4N180_WINDOW_MAP: Optional[str] = head_profiles.head("c4n180").window_enable_map
C4N180_BUS_ORDER: Optional[str] = "straight"

# Which frame group each c4n180 bus rides. Bus 0 is the black column, bus 1 the
# colour column; groups 0 and 1 by default, leaving group 2 idle.
C4N180_BUS_WIRING: Tuple[int, int] = (1, 0)
# ^ MEASURED 2026-07-28, one purge per (bus, 60-nozzle block):
#
#     bus 0, positions   0- 59  ->  yellow      bus 1, all three blocks -> black
#     bus 0, positions  60-119  ->  magenta
#     bus 0, positions 120-179  ->  cyan
#
# So frame group 0 is the COLOUR column and group 1 is the BLACK one, the
# opposite of the (0, 1) this carried, which was read off the head by eye and
# never tested. Column 0 is the black column, so its entry is the group it is
# actually on: 1.
#
# Getting this backwards is what printed three of the four channels in black
# ink at wrong heights: C, M and Y all landed on the black column, which prints
# black from any position it is driven at, and K landed on the colour column.

# Whether a column's bus positions run the same way as the layout's nozzle
# index, per column. MEASURED in the same flash, from which black block ran
# parallel with which colour block:
#
#     front                                                    back
#       colour   pos 120-179 (C)   pos 60-119 (M)   pos 0-59 (Y)
#       black    pos   0- 59       pos 60-119       pos 120-179
#
# The two columns' position indices run in OPPOSITE directions. The layout
# counts every slot's first_nozzle from the front of its column, so the black
# column maps straight through and the colour column has to be reversed:
# position 179-k carries physical nozzle k.
#
# This is the "does nozzle 0 sit at the same end on both columns" assumption,
# which nobody could test until a head ejected reliably enough to read a purge.
# It is also the whole of the vertical mirror seen in the first real print,
# not the sweep direction, which was never wrong.
C4N180_COLUMN_REVERSED: Tuple[bool, bool] = (False, True)

# The bus order is specified by the head, so it is not a guess. The window map
# is derived rather than measured, and the difference has to be said at encode
# time rather than discovered at the head.
C4N180_PROVISIONAL: Tuple[str, ...] = (
    "window map 22 is DERIVED from a dump of a real print, not measured here. "
    "Confirm it with `tools/gen_channel_masks.py --mode windows` first: "
    "channel 0 must eject and channel 1 must not.",
)


def c4n180_bus(column: int,
               column_sources: Sequence[Optional[Tuple[int, int]]],
               order: str) -> BusSpec:
    """Build one c4n180 bus by ordering a column's nozzles onto its positions.

    ``column_sources[k]`` is the ``(slot, nozzle-within-slot)`` that physical
    nozzle ``k`` of the column belongs to, or None where the column has a
    nozzle the RIP never fires (the black column's idle 120). The caller builds
    it from the head layout's slots; this function only decides which position
    of the bus each of those nozzles is shifted out on.

    ``straight`` is the head's actual order: position ``k`` is nozzle ``k``, so
    a column of three 60-nozzle channels lands as three contiguous runs of 60.
    ``interleaved`` is kept as a bench escape hatch: it is what a bus serving
    two banks would need, and costs a flag to try if the head ever disagrees.

    A column whose positions run against the layout's nozzle index is then
    reversed, per ``C4N180_COLUMN_REVERSED``. That is a property of the head's
    wiring, not of the ordering scheme, so it applies after either order: the
    two are independent and conflating them would make ``interleaved`` mean
    something different on one column than on the other.
    """
    if len(column_sources) != BUS_POSITIONS:
        raise ValueError(
            f"a c4n180 column has {BUS_POSITIONS} nozzles, "
            f"got {len(column_sources)}")
    if C4N180_COLUMN_REVERSED[column]:
        column_sources = list(column_sources)[::-1]

    if order == "straight":
        positions = list(column_sources)
    elif order == "interleaved":
        half = BUS_POSITIONS // 2
        positions = []
        for n in range(half):
            positions.append(column_sources[n])
            positions.append(column_sources[half + n])
    else:
        raise ValueError(f"unknown bus order {order!r}")
    return BusSpec(frame_group=C4N180_BUS_WIRING[column],
                   positions=tuple(positions))


PACKERS: Dict[str, HeadPacker] = {C6N90.name: C6N90}


# ---------------------------------------------------------------------------
# Where the columns sit along X
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnGeometry:
    """How far behind the leading column each of a head's columns sits.

    Not a distance in mm but a count of the head's *named gaps*, so the same
    table works whatever those gaps are measured to be (col_align tunes them).
    The leading column (the one the head reaches a point with first) is all
    zeros; every other column is a whole number of gaps behind it.

    On c6n90 the six columns come in three pairs: a small gap inside a pair, a
    large one between pairs. On c4n180 there is one gap and one figure to
    measure, which is what the generalised col_align target prints.
    """

    gap_names: Tuple[str, ...]
    multiples: Tuple[Tuple[int, ...], ...]     # one row per column index
    default_gaps_mm: Dict[str, float] = None   # from the profile; see below

    def __post_init__(self):
        if self.default_gaps_mm is None:
            raise ValueError("a column geometry must carry its default gaps")
        missing = set(self.gap_names) - set(self.default_gaps_mm)
        if missing:
            raise ValueError(
                f"no default for gap(s) {', '.join(sorted(missing))}")
        for name, value in self.default_gaps_mm.items():
            if value is None:
                raise ValueError(
                    f"gap {name!r} is None in the profile: a head that names "
                    f"this gap has to give it a value")
        for column, row in enumerate(self.multiples):
            if len(row) != len(self.gap_names):
                raise ValueError(
                    f"column {column} has {len(row)} gap multiples for "
                    f"{len(self.gap_names)} named gaps")

    def offsets_px(self, gaps_mm: Dict[str, float], dpi: int) -> List[int]:
        """Pixel offset per column at ``dpi``.

        A gap left None in ``gaps_mm`` falls back to this head's own default,
        so a caller that says nothing gets the head it is encoding for rather
        than the reference head's spacing.

        Each gap is rounded up to whole pixels *before* the multiples are
        applied, not after: the offsets have to land on the same pixel grid the
        firing does, and a gap sitting exactly on a boundary would otherwise
        overshoot by a whole pixel through float rounding.
        """
        mm_to_px = dpi / 25.4
        gap_px = {
            name: math.ceil(
                (self.default_gaps_mm[name] if gaps_mm.get(name) is None
                 else gaps_mm[name]) * mm_to_px)
            for name in self.gap_names
        }
        return [
            sum(multiple * gap_px[name]
                for name, multiple in zip(self.gap_names, row))
            for row in self.multiples
        ]


def profile_gaps_mm(head: str, gap_names: Sequence[str]) -> Dict[str, float]:
    """The named gaps of ``head``, read from its profile.

    THE PROFILE IS THE ONLY SOURCE. These values used to be written out here as
    well, and the two copies drifted exactly as you would expect: c6n90's
    profile said the nominal 0.9 / 7.0 while this table held the col_align
    measurements 0.842 / 7.051, and since nothing read the profile, the
    authoritative-looking file was the wrong one. Nobody noticed because the
    behaviour was right; it would have been found the day someone trusted the
    profile.

    So the gap now travels the same path as every other head scalar (YAML,
    codegen, vendored head_profiles), and this function only picks out the
    names a given head's column geometry uses. A head that names a gap its
    profile leaves empty fails at import rather than silently at the bed.
    """
    profile = head_profiles.head(head)
    available = {"column": profile.column_gap_mm,
                 "group": profile.group_gap_mm}
    unknown = set(gap_names) - set(available)
    if unknown:
        raise ValueError(
            f"head {head}: no profile field for gap(s) "
            f"{', '.join(sorted(unknown))}")
    return {name: available[name] for name in gap_names}


COLUMN_GEOMETRY: Dict[str, ColumnGeometry] = {
    # Columns 0..5 left to right; column 5 leads. Pairs (4,5), (2,3), (0,1)
    # sit one "column" gap apart internally and one "group" gap apart between.
    "c6n90": ColumnGeometry(
        gap_names=("group", "column"),
        multiples=((2, 1), (2, 0), (1, 1), (1, 0), (0, 1), (0, 0)),
        default_gaps_mm=profile_gaps_mm("c6n90", ("group", "column")),
    ),
    # Two columns, one gap: the colour column leads, the black column trails.
    "c4n180": ColumnGeometry(
        gap_names=("column",),
        multiples=((1,), (0,)),
        default_gaps_mm=profile_gaps_mm("c4n180", ("column",)),
    ),
}


def column_geometry_for(head_layout: Optional[Dict]) -> ColumnGeometry:
    if not head_layout:
        return COLUMN_GEOMETRY["c6n90"]
    name = head_layout.get("name")
    try:
        return COLUMN_GEOMETRY[name]
    except KeyError:
        raise ValueError(
            f"no column geometry known for head {name!r}") from None


def slot_columns_for(head_layout: Optional[Dict]) -> Tuple[int, ...]:
    """The column each slot sits in, in slot order."""
    if not head_layout:
        return tuple(range(C6N90.slot_count))
    return tuple(slot["column"] for slot in head_layout["slots"])


def _c4n180_packer(head_layout: Dict,
                   window_map: Optional[str] = None,
                   bus_order: Optional[str] = None) -> HeadPacker:
    """Build the c4n180 packer from the payload's head layout.

    ``window_map`` overrides the provisional default for one encode: the bench
    loop is meant to walk through candidates, and doing that from the command
    line keeps every trial traceable to the value that produced it.
    ``bus_order`` overrides the specified order, which should not be needed.
    """
    window_map = window_map or C4N180_WINDOW_MAP
    bus_order = bus_order or C4N180_BUS_ORDER
    if window_map is None or bus_order is None:
        raise ValueError(
            f"cannot encode for head {head_layout.get('name')!r}: no window map "
            f"or bus order to pack with. Pass --window-map / --bus-order, or "
            f"set the defaults in encoder/head_packer.py."
        )

    slots = head_layout["slots"]
    columns: Dict[int, List[Optional[Tuple[int, int]]]] = {}
    for slot_index, slot in enumerate(slots):
        column = columns.setdefault(slot["column"], [None] * BUS_POSITIONS)
        for nozzle in range(slot["nozzle_count"]):
            column[slot["first_nozzle"] + nozzle] = (slot_index, nozzle)

    return HeadPacker(
        name=head_layout["name"],
        buses=tuple(c4n180_bus(index, columns[column], bus_order)
                    for index, column in enumerate(sorted(columns))),
        window_map=window_map,
        slot_count=len(slots),
        frame_groups=FRAME_GROUPS,
        clock_count=CLOCK_COUNT,
        bytes_per_line=BYTES_PER_LINE,
    )


def _with_window_map(packer: HeadPacker,
                     window_map: Optional[str]) -> HeadPacker:
    """``packer``, or a copy of it carrying ``window_map`` instead."""
    if window_map is None or window_map == packer.window_map:
        return packer
    return dataclasses.replace(packer, window_map=window_map)


def packer_for(head_layout: Optional[Dict],
               window_map: Optional[str] = None,
               bus_order: Optional[str] = None) -> HeadPacker:
    """Pick the packer for a RIP payload's ``head_layout`` metadata.

    A payload without the block predates it and is a c6n90 one. ``window_map``
    and ``bus_order`` override a head's provisional values for one encode.

    The override applies to *every* head, including the reference one. It has no
    business being silently dropped for a head whose shipped value happens to be
    measured: a bench tool walking candidate window maps would then get the
    baseline map back on every trial and read the result as if it had tried
    something. Ignoring an argument you accepted is worse than refusing it.
    """
    if not head_layout:
        return _with_window_map(C6N90, window_map)
    name = head_layout.get("name")
    if name == C6N90.name:
        return _with_window_map(C6N90, window_map)
    if name == "c4n180":
        return _c4n180_packer(head_layout, window_map, bus_order)
    raise ValueError(
        f"no wire layout known for head {name!r}; the encoder can pack "
        f"{', '.join(sorted(set(PACKERS) | {'c4n180'}))}")


def provisional_notes(packer: HeadPacker) -> Tuple[str, ...]:
    """What about this head's wire layout is a guess rather than a measurement.

    Empty for c6n90. NOT empty for c4n180: its window map is unknown, and a job
    encoded against the placeholder does not fire at all. A caller that encodes
    a job must print whatever this returns: a print that comes out wrong has
    to be traceable to the guess that produced it rather than left looking like
    a bug, which is precisely how this one was lost.
    """
    return C4N180_PROVISIONAL if packer.name == "c4n180" else ()


def _check_against_profiles() -> None:
    """Every head's window map here must equal its generated profile's.

    There is no build target any more (the host carries every head), so this
    can check them all rather than just the one that shipped. The window map is
    not hashed into any fingerprint, so nothing else would catch it drifting
    from what the profile (and the firmware built from it) expects.
    """
    for name, window_map in (("c6n90", C6N90_WINDOW_MAP),
                             ("c4n180", C4N180_WINDOW_MAP)):
        expected = head_profiles.head(name).window_enable_map
        if window_map is not None and window_map != expected:
            raise AssertionError(
                f"head_packer's window map for {name} disagrees with the "
                f"generated profile: {window_map!r} != {expected!r}. The "
                f"profile is the source; update head_packer."
            )


_check_against_profiles()
