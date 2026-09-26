# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Ink-map routing tests for the encoder.

The ink map says which ink feeds each of the head's six outputs (slots).
These tests decode packed 147-byte lines back into per-slot nozzle bits and
verify the routing end to end: a lit ink lands in exactly the slot the map
plumbs it into, at that slot's physical offset; unplumbed slots stay dark and
drop out of the padding; and the fail-loud validations fire.

Runs under pytest or as a plain script.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "encoder"))
sys.path.insert(0, str(REPO / "rip"))

import encoder as enc  # noqa: E402
import rip_payload  # noqa: E402

DPI = 630
# Slot offsets at 630 DPI with the default gaps (0.842 / 7.051 mm):
# column 21 px, group 175 px.
SLOT_OFFSETS_630 = [371, 350, 196, 175, 21, 0]

# A recognisable nozzle pattern (bits set at both ends and mid-column).
PATTERN = np.zeros(90, dtype=np.uint8)
PATTERN[[0, 1, 7, 42, 89]] = 1


def _decode_line(line: bytes) -> np.ndarray:
    """Unpack a 147-byte line back to per-slot nozzle bits, shape (6, 90).

    Inverse of the packer: 196 clocks x 6 bits (groups A/B/C on the rise
    edge, then on the fall edge), each group's 392-bit stream interleaving
    its odd slot (primary) and even slot (secondary), nozzle data in rows
    90..179 of each expanded column, window-enable map in bits 360..391.
    """
    bits = np.unpackbits(np.frombuffer(line, dtype=np.uint8), bitorder="little")
    clock_bits = bits.reshape(enc.CLOCK_COUNT, enc.BITS_PER_CLOCK)

    slots = np.zeros((enc.SLOT_COUNT, 90), dtype=np.uint8)
    window_bits = np.array([int(c) for c in enc.C6N90_WINDOW_ENABLE_MAP],
                           dtype=np.uint8)
    for group in range(enc.GROUP_COUNT):
        stream = np.zeros(enc.EDGE_COUNT, dtype=np.uint8)
        stream[0::2] = clock_bits[:, group]      # rise edges
        stream[1::2] = clock_bits[:, group + 3]  # fall edges

        merged = stream[:360]
        primary = merged[0::2]
        secondary = merged[1::2]
        # Rows 0..89 are the always-zero first bit of each 2-bit nozzle code.
        assert not primary[:90].any() and not secondary[:90].any()
        slots[2 * group + 1] = primary[90:]
        slots[2 * group] = secondary[90:]

        # The window-enable map rides every group of every column.
        assert np.array_equal(stream[360:], window_bits)
    return slots


def _encode(passes, ink_map, tmp, width):
    """Save a payload, encode it with the given ink map, return header + lines."""
    rip_json = Path(tmp) / "payload.json"
    meta = {
        "dpi": DPI,
        "image_width_px": width,
        "image_height_px": 630,
        "print_width_mm": width / DPI * 25.4,
        "print_height_mm": 630 / DPI * 25.4,
        "nozzle_count": 90,
        "passes_per_band": 7,
        "channel_order": ["C", "M", "Y", "K", "LC", "LM"],
    }
    rip_payload.save_rip(rip_json, meta, [0.0], [0.0], passes)

    out_json = Path(tmp) / "job.json"
    layout = enc.PrintheadLayout(ink_map=ink_map)
    enc.convert_rip_to_printer_format(str(rip_json), str(out_json), layout)

    header = json.loads(out_json.read_text())
    bin_bytes = out_json.with_suffix(".bin").read_bytes()
    assert len(bin_bytes) % enc.BYTES_PER_LINE == 0
    lines = [
        bin_bytes[i:i + enc.BYTES_PER_LINE]
        for i in range(0, len(bin_bytes), enc.BYTES_PER_LINE)
    ]
    return header, lines


def _single_ink_passes(ink_plane, width=1):
    """One pass, one lit plane (PATTERN in its first column), rest blank."""
    pass_data = np.zeros((6, 90, width), dtype=np.uint8)
    pass_data[ink_plane, :, 0] = PATTERN
    return [pass_data]


def test_single_ink_routes_to_each_slot():
    """K plumbed into each slot in turn: its pattern must come out of exactly
    that slot, delayed by exactly that slot's physical offset."""
    passes = _single_ink_passes(ink_plane=3)  # K
    for target_slot in range(enc.SLOT_COUNT):
        ink_map = [enc.EMPTY_SLOT] * enc.SLOT_COUNT
        ink_map[target_slot] = "K"
        with tempfile.TemporaryDirectory() as tmp:
            header, lines = _encode(passes, tuple(ink_map), tmp, width=1)

        offset = SLOT_OFFSETS_630[target_slot]
        assert len(lines) == 1 + offset, f"slot {target_slot}: padding != offset"
        for column, line in enumerate(lines):
            slots = _decode_line(line)
            if column == offset:
                assert np.array_equal(slots[target_slot], PATTERN), (
                    f"slot {target_slot}: pattern missing at its offset column"
                )
                slots[target_slot] = 0
            assert not slots.any(), (
                f"slot {target_slot}: stray drops in column {column}"
            )


def test_swapped_inks_swap_slots_and_offsets():
    """C and M swapped relative to the reference plumbing: each plane must
    fire from the other's slot, at the other's offset."""
    passes = _single_ink_passes(ink_plane=0)  # C
    passes[0][1, :, 0] = PATTERN  # M too
    swapped = ("K", "Y", "LM", "LC", "M", "C")
    with tempfile.TemporaryDirectory() as tmp:
        header, lines = _encode(passes, swapped, tmp, width=1)

    offsets = header["channel_offsets_px"]
    assert offsets["M"] == SLOT_OFFSETS_630[4] and offsets["C"] == 0
    m_slots = _decode_line(lines[SLOT_OFFSETS_630[4]])
    c_slots = _decode_line(lines[0])
    assert np.array_equal(m_slots[4], PATTERN)
    assert np.array_equal(c_slots[5], PATTERN)


def test_default_map_is_explicit_reference_plumbing():
    """The layout default equals passing the profile's plumbing explicitly."""
    passes = [np.random.default_rng(7).integers(0, 2, (6, 90, 5), dtype=np.uint8)]
    with tempfile.TemporaryDirectory() as tmp:
        _, default_lines = _encode(passes, enc.SLOT_INKS, Path(tmp), width=5)
    with tempfile.TemporaryDirectory() as tmp:
        header, explicit_lines = _encode(
            passes, ("K", "Y", "LM", "LC", "C", "M"), Path(tmp), width=5
        )
    assert default_lines == explicit_lines
    assert header["channel_offsets_px"] == {
        "K": 371, "Y": 350, "LM": 196, "LC": 175, "C": 21, "M": 0,
    }


def test_channel_offsets_carry_only_plumbed_inks():
    layout = enc.PrintheadLayout(ink_map=("K", enc.EMPTY_SLOT, enc.EMPTY_SLOT,
                                          enc.EMPTY_SLOT, enc.EMPTY_SLOT, "M"))
    offsets = layout.calculate_channel_offsets_px(DPI)
    assert offsets == {"K": 371, "M": 0}
    assert layout.calculate_total_padding_px(DPI) == 371


def test_wrong_length_ink_map_rejected():
    try:
        enc.PrintheadLayout(ink_map=("K", "Y", "M"))
    except ValueError as exc:
        assert "6 slots" in str(exc)
    else:
        raise AssertionError("short ink map accepted")


def test_duplicate_ink_rejected():
    try:
        enc.PrintheadLayout(ink_map=("K", "K", "LM", "LC", "C", "M"))
    except ValueError as exc:
        assert "more than one slot" in str(exc)
    else:
        raise AssertionError("duplicate ink accepted")


def test_unknown_ink_rejected():
    try:
        enc.resolve_ink_map(("K", "Y", "LM", "LC", "C", "W"),
                            ["C", "M", "Y", "K", "LC", "LM"])
    except ValueError as exc:
        assert "'W'" in str(exc)
    else:
        raise AssertionError("unknown ink accepted")


def test_dropped_ink_with_coverage_rejected():
    """A payload plane with drops but no slot must fail loud, not vanish."""
    passes = _single_ink_passes(ink_plane=2)  # Y has coverage
    no_y = ("K", enc.EMPTY_SLOT, "LM", "LC", "C", "M")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _encode(passes, no_y, tmp, width=1)
        except ValueError as exc:
            assert "'Y'" in str(exc) and "5 drops" in str(exc)
        else:
            raise AssertionError("dropped ink accepted")


def test_blank_plane_may_go_unmapped():
    """Blank planes (LC/LM today) may be left without a slot; that is the
    normal four-ink setup."""
    passes = _single_ink_passes(ink_plane=3)  # only K lit
    four_inks = ("K", "Y", enc.EMPTY_SLOT, enc.EMPTY_SLOT, "C", "M")
    with tempfile.TemporaryDirectory() as tmp:
        header, lines = _encode(passes, four_inks, tmp, width=1)
    assert header["channel_offsets_px"] == {"K": 371, "Y": 350, "C": 21, "M": 0}
    slots = _decode_line(lines[371])
    assert np.array_equal(slots[0], PATTERN)


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
