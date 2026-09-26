# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Wire-layout tests: the reference head must not move, and the dual-column
head must pack the shape it claims.

The golden in ``golden/c6n90_encoded.json`` was frozen from the hand-written
three-group packer, before it was generalised into per-bus position tables.
Every case must still encode to the same bytes.

The c4n180 half pins the specified wire layout (one bus per column, the colour
bus split into three contiguous runs of 60 in *both* 180-bit blocks) and keeps
honest about the one value still unmeasured: the window map, carried over from
the reference head, which the tests require to be announced as a guess on every
job and swappable per encode. The frame around it must be right regardless: the
colour column packed solid, the black column's idle nozzles dark, and the
unwired third data pin silent.

Runs under pytest or as a plain script.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "encoder"))
sys.path.insert(0, str(REPO / "rip"))

import encoder as enc  # noqa: E402
import head_packer as hp  # noqa: E402
import head_profiles  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden" / "c6n90_encoded.json"
C6N90_ORDER = ["C", "M", "Y", "K", "LC", "LM"]


def _bus_stream(line: bytes, packer, frame_group: int) -> np.ndarray:
    """Recover one data pin's 392-bit stream from a packed line."""
    bits = np.unpackbits(np.frombuffer(line, dtype=np.uint8), bitorder="little")
    clocks = bits.reshape(packer.clock_count, packer.bits_per_clock)
    stream = np.zeros(hp.EDGE_COUNT, dtype=np.uint8)
    stream[0::2] = clocks[:, frame_group]
    stream[1::2] = clocks[:, frame_group + packer.frame_groups]
    return stream


# ---------------------------------------------------------------------------
# c6n90 must not move
# ---------------------------------------------------------------------------

def test_c6n90_encoding_is_unchanged():
    cases = json.loads(GOLDEN.read_text())
    assert cases, "golden fixture is empty"
    for case in cases:
        ink_map = tuple(case["ink_map"])
        rng = np.random.default_rng(case["seed"])
        passes = [rng.integers(0, 2, (6, 90, case["width"]), dtype=np.uint8)
                  for _ in range(3)]
        plumbed = {ink for ink in ink_map if ink != hp.UNUSED and ink != "-"}
        for pass_data in passes:
            for plane, channel in enumerate(C6N90_ORDER):
                if channel not in plumbed:
                    pass_data[plane] = 0

        layout = enc.PrintheadLayout(ink_map=ink_map)
        slot_passes = enc.gather_slot_passes(
            passes, enc.resolve_ink_map(ink_map, C6N90_ORDER))
        shifted = enc.apply_slot_offsets(slot_passes, layout, case["dpi"])
        encoded = enc.encode_all_passes(shifted, show_progress=False)

        blob = b"".join(b"".join(p) for p in encoded)
        label = f"{case['name']} w={case['width']} dpi={case['dpi']}"
        assert sum(len(p) for p in encoded) == case["line_count"], \
            f"{label}: line count moved"
        assert hashlib.sha256(blob).hexdigest() == case["sha256"], \
            f"{label}: packed bytes moved"
        assert layout.calculate_channel_offsets_px(case["dpi"]) == \
            case["offsets_px"], f"{label}: slot offsets moved"


def test_a_lit_nozzle_lands_on_its_own_bus_bit():
    """Bus position p carries its code's fire bit at index 180+p, the fact the
    whole per-head table rests on."""
    packer = hp.C6N90
    for slot, nozzle in ((1, 0), (0, 0), (1, 7), (3, 42), (4, 89), (5, 12)):
        data = np.zeros((6, 90, 1), dtype=np.uint8)
        data[slot, nozzle, 0] = 1
        line = packer.encode_pass(data)[0]

        bus = packer.buses[slot // 2]
        position = bus.positions.index((slot, nozzle))
        stream = _bus_stream(line, packer, bus.frame_group)
        lit = [i for i in np.flatnonzero(stream) if i < 2 * hp.BUS_POSITIONS]
        assert lit == [hp.BUS_POSITIONS + position], (
            f"slot {slot} nozzle {nozzle}: data bits at {lit}, expected "
            f"{[hp.BUS_POSITIONS + position]}")
        # The other two buses carry no data, only their window map.
        for other in packer.buses:
            if other is bus:
                continue
            quiet = _bus_stream(line, packer, other.frame_group)
            assert not quiet[:2 * hp.BUS_POSITIONS].any()


def test_window_map_rides_every_bus():
    packer = hp.C6N90
    line = packer.encode_pass(np.zeros((6, 90, 1), dtype=np.uint8))[0]
    expected = np.array([int(c) for c in packer.window_map], dtype=np.uint8)
    for bus in packer.buses:
        stream = _bus_stream(line, packer, bus.frame_group)
        assert np.array_equal(stream[2 * hp.BUS_POSITIONS:], expected)


# ---------------------------------------------------------------------------
# c4n180
# ---------------------------------------------------------------------------

C4N180_LAYOUT = {
    "name": "c4n180",
    "nozzle_pitch_npi": 180,
    "column_length": 180,
    "channel_order": ["C", "M", "Y", "K"],
    "ink_map": ["K", "C", "M", "Y"],
    "slots": [
        {"column": 0, "first_nozzle": 0, "nozzle_count": 60,
         "physical_nozzle_count": 180, "ink": "K"},
        {"column": 1, "first_nozzle": 0, "nozzle_count": 60,
         "physical_nozzle_count": 60, "ink": "C"},
        {"column": 1, "first_nozzle": 60, "nozzle_count": 60,
         "physical_nozzle_count": 60, "ink": "M"},
        {"column": 1, "first_nozzle": 120, "nozzle_count": 60,
         "physical_nozzle_count": 60, "ink": "Y"},
    ],
}


def _c4n180(order=None, window=None):
    """The c4n180 packer, optionally with a candidate order / window map."""
    return hp.packer_for(C4N180_LAYOUT, window_map=window, bus_order=order)


def test_the_column_gaps_come_from_the_profile_and_nowhere_else():
    """One source for a head's gaps, checked by value not by inspection.

    These were written out twice (once in the profile, once in
    COLUMN_GEOMETRY) and drifted: c6n90's profile said the nominal 0.9 / 7.0
    while the effective values were the col_align measurements 0.842 / 7.051.
    Nothing read the profile, so the file that looked authoritative was the
    wrong one and the behaviour stayed right, which is why it survived.

    Re-hardcoding a value here would pass every other test in this file.
    """
    for name, geometry in hp.COLUMN_GEOMETRY.items():
        profile = head_profiles.head(name)
        expected = {"column": profile.column_gap_mm,
                    "group": profile.group_gap_mm}
        for gap in geometry.gap_names:
            assert geometry.default_gaps_mm[gap] == expected[gap], (
                f"{name}'s {gap} gap is {geometry.default_gaps_mm[gap]} in "
                f"COLUMN_GEOMETRY but {expected[gap]} in the profile: the "
                f"profile is the source, so this is a second copy")


def test_a_head_may_not_name_a_gap_its_profile_leaves_empty():
    """c4n180 has one gap and no group_gap_mm. Naming one anyway must fail at
    import, not silently place a column at None mm."""
    try:
        hp.ColumnGeometry(
            gap_names=("column", "group"),
            multiples=((0, 0), (1, 0)),
            default_gaps_mm=hp.profile_gaps_mm("c4n180", ("column", "group")))
    except ValueError as exc:
        assert "group" in str(exc)
    else:
        raise AssertionError("accepted a gap the profile leaves empty")


def test_c4n180_still_announces_its_map_as_derived():
    """c4n180's map is read off a capture of a real print, which settles what
    the head DOES but not that ours ejects. Until a purge confirms it, encoding
    a job has to say so; the previous value was believed for a day on evidence
    that could not refute it, and every job in between came out wrong."""
    assert hp.provisional_notes(_c4n180()) != ()
    assert hp.provisional_notes(hp.C6N90) == ()


# The 32 bits are 8 firing windows of 4 codes: a nibble per window, and inside
# it one bit per 2-bit nozzle code, in this order.
WINDOW_CODES = ("11", "10", "01", "00")


def _windows_fired(window_map):
    """A map read back as the (window, code) pairs it enables."""
    return [(index // len(WINDOW_CODES), WINDOW_CODES[index % len(WINDOW_CODES)])
            for index, bit in enumerate(window_map) if bit == "1"]


def test_a_shipped_map_never_fires_the_blank_code():
    """The property a print depends on, checked by DERIVATION not by string.

    Pinning the literal value only catches an edit; this catches a map that
    fires code 00, which is the failure that put ink on the white parts of a
    page on 2026-07-27. That map (bits 30 and 31) is window 7 slots 2 AND 3,
    and reads as legitimate until you decompose it.
    """
    for packer in (_c4n180(), hp.C6N90):
        fired = _windows_fired(packer.window_map)
        assert [code for _, code in fired] == ["01"], (
            f"{packer.name}: a binary print may fire code 01 and nothing else, "
            f"but this map fires {fired}")


def test_each_head_fires_01_in_exactly_one_window():
    """One code in one window per head, but WHICH window is measured, never
    derived.

    The windows do not map alike across heads, and a head's own dump does not
    predict them either: c6n90 prints on this board in window 3, c4n180 has
    window 3 shut and prints in window 7, and window 5 (the one the captured
    original controller uses for code 01) does not open for us at all. A dump
    says what the HEAD does; only a purge says what this board can trigger.

    So this pins the shape (exactly one window, and it is code 01's) and refuses
    to pin the number. An earlier version of this test asserted c4n180's drop
    had to sit in windows 4-6 because the capture put it there; the bench then
    measured window 7. A test that encodes a derivation fails when the
    derivation is what was wrong.
    """
    for packer, window_map in ((_c4n180(), _c4n180().window_map),
                               (hp.C6N90, hp.C6N90_WINDOW_MAP)):
        fired = _windows_fired(window_map)
        assert len(fired) == 1, \
            f"{packer.name}: a binary print fires in one window, not {fired}"
    assert _windows_fired(_c4n180().window_map)[0][0] \
        != _windows_fired(hp.C6N90_WINDOW_MAP)[0][0], \
        "these two heads are measured to use DIFFERENT windows; if a change " \
        "makes them agree, it is copying one head's value onto the other again"


def test_the_window_map_is_still_overridable_per_encode():
    """Measured, not frozen: the flag stays, because the next head brought up on
    this frame will need it before its own map is known."""
    other = "1" + "0" * 31
    assert _c4n180(window=other).window_map == other


def test_every_head_is_describable_without_a_build_target():
    """There is no build target: the host carries every head's scalars, so both
    packers exist at once and each is checked against its own profile."""
    assert hp.C6N90.name == "c6n90"
    assert hp.C6N90.slot_count == 6 and len(hp.C6N90.buses) == 3
    assert hp.packer_for({"name": "c6n90"}) is hp.C6N90
    assert _c4n180().name == "c4n180"
    hp._check_against_profiles()


def test_c4n180_rides_the_c6n90_frame_with_one_pin_idle():
    packer = _c4n180()
    assert packer.bytes_per_line == 147, "the frame must not move"
    assert packer.clock_count == 196 and packer.bits_per_clock == 6
    assert len(packer.buses) == 2
    assert packer.idle_frame_groups == (2,), \
        "the head has two buses; the third data pin is unwired"
    # 60 black nozzles + 3 x 60 colour: the black column's idle 120 are not wired.
    assert packer.wired_nozzles == 240


def test_c4n180_packs_the_colour_column_solid_and_the_black_column_partly():
    packer = _c4n180()
    # Every slot fires every one of its 60 nozzles.
    data = np.ones((4, 60, 1), dtype=np.uint8)
    line = packer.encode_pass(data)[0]

    black = _bus_stream(line, packer, packer.buses[0].frame_group)
    colour = _bus_stream(line, packer, packer.buses[1].frame_group)
    fire = slice(hp.BUS_POSITIONS, 2 * hp.BUS_POSITIONS)

    assert colour[fire].sum() == 180, "the colour column should be packed solid"
    assert black[fire].sum() == 60, "only the black column's fired third"
    assert black[fire][:60].all() and not black[fire][60:].any(), \
        "the black column's data should sit at its bottom 60 nozzles"
    # The unwired pin says nothing at all, not even a window map.
    idle = _bus_stream(line, packer, 2)
    assert not idle.any(), "the unwired data pin must be silent"


# The c4n180's physical map, MEASURED 2026-07-28 by purging one 60-position
# block of one bus at a time and writing down the ink that came out:
#
#     bus 0, positions   0- 59 -> yellow     bus 1, positions   0- 59 -> black
#     bus 0, positions  60-119 -> magenta    bus 1, positions  60-119 -> black
#     bus 0, positions 120-179 -> cyan       bus 1, positions 120-179 -> black
#
# and, from which black block ran parallel with which colour block, the two
# columns' positions run in OPPOSITE directions: black block 0 sits beside
# colour block 2. Every number below is that table, not a reading of the head.
MEASURED_BLOCKS = {
    # ink: (frame group, first position)
    "Y": (0, 0), "M": (0, 60), "C": (0, 120), "K": (1, 0),
}


def test_the_colour_bus_splits_into_three_contiguous_runs_of_sixty():
    """Each channel owns one contiguous 60-position block, and the colour
    column's blocks run against the layout's nozzle index.

    The position table is what both 180-bit halves of the stream index (the
    first-bit block at p and the fire block at 180+p), so pinning it pins the
    split in both, including for the drop codes not emitted yet.
    """
    colour = _c4n180().buses[1]        # buses are indexed by COLUMN, not group
    assert colour.frame_group == MEASURED_BLOCKS["C"][0]
    for slot, ink in ((1, "C"), (2, "M"), (3, "Y")):
        first = MEASURED_BLOCKS[ink][1]
        run = colour.positions[first:first + 60]
        assert [src[0] for src in run] == [slot] * 60, \
            f"positions {first}..{first + 59} are not {ink} alone"
        # Reversed column: the block runs backwards through the channel's own
        # nozzles as the position index rises.
        assert [src[1] for src in run] == list(reversed(range(60))), \
            f"{ink}'s nozzles do not run against the position index"

    # The black bus is the same rule with one channel and no reversal: its
    # plumbed third is wired, the rest of the column is not.
    black = _c4n180().buses[0]
    assert black.frame_group == MEASURED_BLOCKS["K"][0]
    assert all(src == (0, n) for n, src in enumerate(black.positions[:60]))
    assert all(src is None for src in black.positions[60:])


def test_c4n180_puts_each_ink_where_the_bench_measured_it():
    """The end-to-end version: light one slot, find the lit positions.

    This is the test the previous layout passed while printing three channels
    in black ink; it asserted the head as read by eye. Anchoring it to the
    purge table instead is the point: if a change moves an ink off its measured
    block, that is a regression however tidy the code looks.
    """
    packer = _c4n180()
    for slot, ink in enumerate(("K", "C", "M", "Y")):
        group, first = MEASURED_BLOCKS[ink]
        data = np.zeros((4, 60, 1), dtype=np.uint8)
        data[slot, :, 0] = 1
        stream = _bus_stream(packer.encode_pass(data)[0], packer, group)
        lit = np.flatnonzero(stream[hp.BUS_POSITIONS:2 * hp.BUS_POSITIONS])
        assert lit.tolist() == list(range(first, first + 60)), (
            f"{ink} landed on bus {group} positions {lit.min()}..{lit.max()}, "
            f"but the bench measured it at {first}..{first + 59}")


def test_the_two_columns_run_in_opposite_directions():
    """Black block 0 sits beside colour block 2: measured, and the whole of
    the vertical mirror in the first real print.

    Nozzle 0 of the black slot and nozzle 0 of the cyan slot are the same
    distance along the head, so they must come out at opposite ends of the
    position index. A change that quietly re-aligns the two columns would
    reintroduce the mirror on three channels out of four.
    """
    packer = _c4n180()
    ends = {}
    for slot, ink in ((0, "K"), (1, "C")):
        data = np.zeros((4, 60, 1), dtype=np.uint8)
        data[slot, 0, 0] = 1                     # that slot's FIRST nozzle
        group = MEASURED_BLOCKS[ink][0]
        stream = _bus_stream(packer.encode_pass(data)[0], packer, group)
        ends[ink] = int(np.flatnonzero(
            stream[hp.BUS_POSITIONS:2 * hp.BUS_POSITIONS])[0])
    assert ends["K"] == 0, "black's first nozzle is at the low end of its bus"
    assert ends["C"] == 179, \
        "cyan's first nozzle sits at the HIGH end of the colour bus: the two " \
        "columns are wired in opposite directions"


def test_the_bench_escape_hatch_changes_only_the_table():
    """``interleaved`` is not expected to be needed (the head specifies the
    straight order), but it stays reachable by flag, and must be a re-ordering
    of the same nozzles rather than a different frame."""
    straight = _c4n180(order="straight")
    interleaved = _c4n180(order="interleaved")
    assert straight.bytes_per_line == interleaved.bytes_per_line
    assert straight.wired_nozzles == interleaved.wired_nozzles
    assert straight.buses[1].positions != interleaved.buses[1].positions, \
        "the two candidate orders must actually differ"

    data = np.ones((4, 60, 1), dtype=np.uint8)
    assert straight.encode_pass(data) != interleaved.encode_pass(data), \
        "if the order did not change the bytes, it would not be worth measuring"


# ---------------------------------------------------------------------------
# Column geometry (X offsets)
# ---------------------------------------------------------------------------

def test_c6n90_slot_offsets_are_unchanged():
    """Frozen values from the col_align calibration, at the shipped defaults."""
    layout = enc.PrintheadLayout()
    assert layout.slot_offsets_px(630) == [371, 350, 196, 175, 21, 0]
    assert layout.calculate_channel_offsets_px(630) == {
        "K": 371, "Y": 350, "LM": 196, "LC": 175, "C": 21, "M": 0,
    }


def test_gaps_round_up_before_they_are_multiplied():
    """Each gap becomes whole pixels first, then the multiples are applied;
    rounding the total instead would drift the far columns off the firing
    grid."""
    import math
    geometry = hp.COLUMN_GEOMETRY["c6n90"]
    # 7.1 mm at 630 dpi is 176.1 px: one gap rounds up to 177, so two gaps are
    # 354. Rounding the *total* instead would give ceil(352.2) = 353, a pixel
    # of drift on the far column, which is the bug this ordering avoids.
    offsets = geometry.offsets_px({"group": 7.1, "column": 0.0}, 630)
    assert math.ceil(2 * 7.1 * 630 / 25.4) == 353, "the two orderings must differ"
    assert offsets[0] == offsets[1] == 354
    assert offsets[2] == offsets[3] == 177
    assert offsets[4] == offsets[5] == 0


def test_c4n180_columns_share_one_offset_and_one_gap():
    layout = enc.PrintheadLayout(
        ink_map=("K", "C", "M", "Y"),
        slot_columns=hp.slot_columns_for(C4N180_LAYOUT),
        columns=hp.column_geometry_for(C4N180_LAYOUT),
    )
    offsets = layout.calculate_channel_offsets_px(720)
    # The three colours are in one column, so they register together with no
    # special case; only the black column trails.
    assert offsets["C"] == offsets["M"] == offsets["Y"] == 0
    assert offsets["K"] == layout.calculate_total_padding_px(720) > 0
    # One gap to measure, not five.
    assert hp.COLUMN_GEOMETRY["c4n180"].gap_names == ("column",)


def test_an_ink_map_of_the_wrong_length_for_the_head_is_refused():
    try:
        enc.PrintheadLayout(ink_map=("K", "C", "M", "Y"))  # 4 inks, 6 slots
    except ValueError as exc:
        assert "6 slots" in str(exc)
    else:
        raise AssertionError("accepted a four-slot ink map on c6n90")


# ---------------------------------------------------------------------------
# Dispatch and validation
# ---------------------------------------------------------------------------

def test_a_payload_without_a_head_block_is_c6n90():
    assert hp.packer_for(None) is hp.C6N90
    assert hp.packer_for({"name": "c6n90"}) is hp.C6N90


def test_a_window_map_override_reaches_every_head():
    """Accepting the argument and dropping it would make a bench sweep of
    candidate maps silently re-test the baseline on every trial."""
    other = "1" + "0" * 31
    for layout in (None, {"name": "c6n90"}, C4N180_LAYOUT):
        assert hp.packer_for(layout, window_map=other).window_map == other
    # An override equal to the shipped value, or none at all, keeps the
    # singleton; the c6n90 encode path must not start copying packers.
    assert hp.packer_for({"name": "c6n90"},
                         window_map=hp.C6N90_WINDOW_MAP) is hp.C6N90
    # Overriding must not disturb anything else about the head.
    swapped = hp.packer_for({"name": "c6n90"}, window_map=other)
    assert swapped.buses == hp.C6N90.buses
    assert swapped.slot_count == hp.C6N90.slot_count


def test_an_unknown_head_is_refused():
    try:
        hp.packer_for({"name": "c8n120"})
    except ValueError as exc:
        assert "c8n120" in str(exc)
    else:
        raise AssertionError("packed for a head with no known wire layout")


def test_packer_validation():
    good = dict(name="t", buses=(hp.C6N90.buses[0],), slot_count=6,
                window_map=hp.C6N90_WINDOW_MAP)
    hp.HeadPacker(**good)

    for override, reason in (
        ({"window_map": "0101"}, "short window map"),
        ({"buses": (hp.C6N90.buses[0], hp.C6N90.buses[0])}, "two buses on one pin"),
        ({"bytes_per_line": 98}, "byte count disagreeing with the frame"),
        ({"frame_groups": 0}, "no data pins"),
    ):
        try:
            hp.HeadPacker(**{**good, **override})
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted a packer with a {reason}")


def test_a_pass_with_too_few_rows_is_refused():
    """A c6n90 packer fed 60-row planes must fail, not pack whatever fits."""
    try:
        hp.C6N90.encode_pass(np.zeros((6, 60, 1), dtype=np.uint8))
    except ValueError as exc:
        assert "60 rows" in str(exc)
    else:
        raise AssertionError("packed a pass whose planes were too short")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failures else 0)
