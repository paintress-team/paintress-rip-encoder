# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Head-layout tests: the reference head must not move, and the dual-column
head must tile.

Two things are checked here. First, that generalising the pass slicer to
per-slot geometry left ``c6n90`` byte-for-byte where it was; the golden in
``golden/c6n90_passes.json`` was frozen from the single-slot implementation.
Second, that ``c4n180`` (whose right-hand column is three 60-nozzle blocks
stacked in Y) lays every ink on every row exactly once, including through the
lead-in the taller slots need and through the feathered seams.

The plumbing tests are the point of the split between layout and ink map: on
this head, re-plumbing an ink moves the rows it reaches.

Runs under pytest or as a plain script.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "rip"))

import head_layout  # noqa: E402
import rip  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden" / "c6n90_passes.json"
CMYK = ("C", "M", "Y", "K")


def _synthetic(height, width, seed):
    """The same pseudo-random halftone the golden was frozen from."""
    rng = np.random.default_rng(seed)
    return {ch: (rng.random((height, width)) < f).astype(np.uint8)
            for ch, f in zip(CMYK, (0.3, 0.4, 0.2, 0.5))}


def _deposit_counts(config, passes, y_positions, height, width, channels):
    """How many times each (channel, row, x) pixel is deposited.

    Replays the passes the way the machine would: put the head at the pass's Y,
    and let every nozzle of every slot fire onto the row it physically covers.
    """
    geometry = config.geometry
    counts = np.zeros((len(channels), height, width), dtype=int)
    for pass_data, y in zip(passes, y_positions):
        head_line = int(round(y / geometry.line_spacing_mm))
        for ci, channel in enumerate(channels):
            slot = geometry.slot_for(channel)
            for nozzle in range(slot.nozzle_count):
                line = head_line + (slot.first_nozzle + nozzle) * geometry.nozzle_pitch_lines
                row = height - 1 - line
                if 0 <= row < height:
                    counts[ci, row] += pass_data[ci, nozzle]
    return counts


# ---------------------------------------------------------------------------
# c6n90 must not move
# ---------------------------------------------------------------------------

def test_c6n90_output_is_unchanged():
    """Every frozen case still slices to the same bytes and the same Y list."""
    cases = json.loads(GOLDEN.read_text())
    assert cases, "golden fixture is empty"
    for case in cases:
        dpi, height, width = case["dpi"], case["height"], case["width"]
        overlap = case["band_overlap"]
        config = rip.PrintheadConfig(dpi=dpi)
        passes = rip.generate_print_passes(
            _synthetic(height, width, seed=dpi + height), config, CMYK,
            band_overlap=overlap)
        stacked = np.stack(passes)
        y = rip.compute_pass_y_positions_mm(
            height / dpi * 25.4, config, band_overlap=overlap)

        label = f"dpi={dpi} h={height} overlap={overlap}"
        assert list(stacked.shape) == case["shape"], f"{label}: shape moved"
        assert hashlib.sha256(stacked.tobytes()).hexdigest() == case["sha256"], \
            f"{label}: pass data moved"
        assert [round(v, 9) for v in y] == case["y_positions_mm"], \
            f"{label}: Y positions moved"


def test_c6n90_legacy_names_still_mean_what_they_did():
    config = rip.PrintheadConfig(dpi=630)
    assert config.nozzle_count == 90
    assert config.passes_per_band == 7
    assert config.lines_per_band == 630          # a one-inch band
    assert config.channel_order == ("C", "M", "Y", "K", "LC", "LM")
    assert config.geometry.lead_in_bands() == 0  # every slot starts at the bottom


# ---------------------------------------------------------------------------
# c4n180 geometry
# ---------------------------------------------------------------------------

def test_c4n180_band_step_is_the_shortest_slot_not_the_column():
    """The column is a full inch, but the smallest ink block is 1/3", and it
    is the block that sets the advance."""
    config = rip.PrintheadConfig(dpi=720, head="c4n180")
    geometry = config.geometry
    assert geometry.nozzle_pitch_lines == 4          # 720 dpi / 180 npi
    assert geometry.band_step_nozzles == 60
    assert geometry.band_step_lines == 240           # a third of an inch
    assert abs(geometry.band_step_mm - 25.4 / 3) < 1e-9
    assert geometry.max_group_nozzles == 60          # payload nozzle dimension
    # The black column is 180 nozzles but fires only its bottom third.
    black = geometry.slot_for("K")
    assert (black.nozzle_count, black.physical_nozzle_count) == (60, 180)
    assert black.idle_nozzle_count == 120


def test_c4n180_leads_in_below_the_image():
    """Yellow sits two blocks up the column, so the head has to start two band
    steps below the image before it can reach the bottom rows."""
    config = rip.PrintheadConfig(dpi=720, head="c4n180")
    geometry = config.geometry
    assert geometry.lead_in_bands() == 2
    assert abs(geometry.lead_in_mm() - 2 * 25.4 / 3) < 1e-9

    y = rip.compute_pass_y_positions_mm(900 / 720 * 25.4, config)
    assert min(y) < 0, "no lead-in: the top slot can never reach the bottom rows"
    assert abs(min(y) + geometry.lead_in_mm()) < 1e-9
    assert y == sorted(y), "passes must advance monotonically in Y"


def test_c4n180_prints_every_pixel_exactly_once():
    height, width, dpi = 900, 4, 720
    config = rip.PrintheadConfig(dpi=dpi, head="c4n180")
    bitmaps = {ch: np.ones((height, width), dtype=np.uint8) for ch in CMYK}

    passes = rip.generate_print_passes(bitmaps, config, CMYK)
    y = rip.compute_pass_y_positions_mm(height / dpi * 25.4, config)
    assert len(passes) == len(y), "a pass without a Y position (or the reverse)"

    counts = _deposit_counts(config, passes, y, height, width, CMYK)
    for ci, channel in enumerate(CMYK):
        assert (counts[ci] == 1).all(), (
            f"{channel}: deposited {sorted(np.unique(counts[ci]))} times per pixel, "
            f"expected exactly once"
        )


def test_c4n180_feathered_seams_still_print_every_pixel_once():
    """Each slot meets the band seam at its own height; the stochastic split
    must still hand every row to exactly one band."""
    height, width, dpi = 900, 6, 720
    config = rip.PrintheadConfig(dpi=dpi, head="c4n180")
    bitmaps = {ch: np.ones((height, width), dtype=np.uint8) for ch in CMYK}

    passes = rip.generate_print_passes(bitmaps, config, CMYK, band_overlap=8)
    y = rip.compute_pass_y_positions_mm(height / dpi * 25.4, config, band_overlap=8)
    counts = _deposit_counts(config, passes, y, height, width, CMYK)
    for ci, channel in enumerate(CMYK):
        assert (counts[ci] == 1).all(), (
            f"{channel}: feathered seam left {sorted(np.unique(counts[ci]))} "
            f"deposits per pixel"
        )


def test_c4n180_carries_no_light_ink_planes():
    config = rip.PrintheadConfig(dpi=720, head="c4n180")
    assert config.geometry.blank_channels == ()
    assert config.channel_order == CMYK


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def test_replumbing_moves_which_rows_an_ink_reaches():
    """The whole point of resolving geometry and plumbing together: swap two
    inks between slots and their rows swap with them."""
    reference = rip.PrintheadConfig(dpi=720, head="c4n180")
    swapped = rip.PrintheadConfig(dpi=720, head="c4n180",
                                  ink_map=("Y", "C", "M", "K"))

    # Reference: K is the tall left column (bottom), Y is the top block.
    assert reference.geometry.slot_for("K").column == 0
    assert reference.geometry.slot_for("Y").first_nozzle == 120
    # Swapped: they trade places, and the idle nozzles follow the *slot*.
    assert swapped.geometry.slot_for("Y").column == 0
    assert swapped.geometry.slot_for("Y").physical_nozzle_count == 180
    assert swapped.geometry.slot_for("K").first_nozzle == 120

    height, width, dpi = 480, 3, 720
    bitmaps = {ch: np.ones((height, width), dtype=np.uint8) for ch in CMYK}
    for config in (reference, swapped):
        passes = rip.generate_print_passes(bitmaps, config, CMYK)
        y = rip.compute_pass_y_positions_mm(height / dpi * 25.4, config)
        counts = _deposit_counts(config, passes, y, height, width, CMYK)
        assert (counts == 1).all(), "re-plumbed head no longer tiles"


def test_replumbing_changes_the_sweep_not_the_rows_a_dead_nozzle_ruins():
    """A slot's height is always a whole number of band steps (the layout
    refuses anything else, or the inks could not tile), so moving an ink to a
    slot higher up the column shifts *which sweep* deposits a row, not which
    rows a given nozzle owns. The streak stays put; the pass it happens on
    does not."""
    height = 480
    dead = {"K": [0]}
    reference = rip.PrintheadConfig(dpi=720, head="c4n180")
    swapped = rip.PrintheadConfig(dpi=720, head="c4n180",
                                  ink_map=("Y", "C", "M", "K"))

    ref_rows = rip.dead_nozzle_row_masks(dead, height, reference)["K"]
    moved_rows = rip.dead_nozzle_row_masks(dead, height, swapped)["K"]
    assert ref_rows.any(), "a dead nozzle that ruins no row at all"
    assert np.array_equal(ref_rows, moved_rows)

    # ... and both masks are right, checked against the slicer itself.
    for config in (reference, swapped):
        assert np.array_equal(
            rip.dead_nozzle_row_masks(dead, height, config)["K"],
            _rows_a_nozzle_prints(config, "K", 0, height))

    # The sweeps really do differ, which is what re-plumbing changes.
    assert (_passes_a_nozzle_prints_on(reference, "K", 0, height)
            != _passes_a_nozzle_prints_on(swapped, "K", 0, height))


def _rows_a_nozzle_prints(config, channel, nozzle, height):
    """Rows a nozzle reaches, straight from the schedule (no N2 arithmetic)."""
    geometry = config.geometry
    slot = geometry.slot_for(channel)
    printed = np.zeros(height, dtype=bool)
    for head_line in geometry.pass_schedule(height):
        line = head_line + (slot.first_nozzle + nozzle) * geometry.nozzle_pitch_lines
        row = height - 1 - line
        if 0 <= row < height:
            printed[row] = True
    return printed


def _passes_a_nozzle_prints_on(config, channel, nozzle, height):
    geometry = config.geometry
    slot = geometry.slot_for(channel)
    return [i for i, head_line in enumerate(geometry.pass_schedule(height))
            if 0 <= height - 1 - (head_line + (slot.first_nozzle + nozzle)
                                  * geometry.nozzle_pitch_lines) < height]


def test_dead_nozzle_mask_matches_the_rows_the_nozzle_actually_prints():
    """The N2 mask and the pass slicer must agree on which rows a nozzle owns:
    they are separate pieces of arithmetic over the same lattice."""
    height = 480
    for head, dpi, nozzle, channel in (("c6n90", 630, 42, "C"),
                                       ("c6n90", 630, 0, "LM"),
                                       ("c4n180", 720, 17, "M"),
                                       ("c4n180", 720, 3, "K"),
                                       ("c4n180", 360, 59, "Y")):
        config = rip.PrintheadConfig(dpi=dpi, head=head)
        mask = rip.dead_nozzle_row_masks({channel: [nozzle]}, height, config)[channel]
        assert np.array_equal(
            mask, _rows_a_nozzle_prints(config, channel, nozzle, height)), \
            f"{head}@{dpi}/{channel}[{nozzle}]: N2 mask disagrees with the slicer"


def test_ink_map_is_validated_against_the_head():
    layout = head_layout.get_layout("c4n180")
    for bad, reason in (
        (("K", "C", "M"), "too short"),
        (("K", "C", "M", "Y", "LC"), "too long"),
        (("K", "C", "C", "Y"), "duplicate ink"),
        (("K", "C", "M", "LM"), "ink the head does not carry"),
    ):
        try:
            layout.validate_ink_map(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted an ink map with {reason}: {bad}")

    # A six-slot c6n90 map is not a four-slot c4n180 map.
    try:
        layout.validate_ink_map(("K", "Y", "LM", "LC", "C", "M"))
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a c6n90 ink map on c4n180")


def test_unplumbed_ink_is_refused_rather_than_dropped():
    config = rip.PrintheadConfig(dpi=720, head="c4n180",
                                 ink_map=("K", "C", "M", head_layout.EMPTY_SLOT))
    assert config.geometry.imaging_channels == ("C", "M", "K")
    bitmaps = {ch: np.ones((60, 2), dtype=np.uint8) for ch in CMYK}
    try:
        rip.generate_print_passes(bitmaps, config, CMYK)
    except ValueError as exc:
        assert "'Y'" in str(exc) and "not plumbed" in str(exc)
    else:
        raise AssertionError("an unplumbed ink with data was silently dropped")


# ---------------------------------------------------------------------------
# Layout validation and DPI
# ---------------------------------------------------------------------------

def test_dpi_must_be_a_multiple_of_the_nozzle_pitch():
    # 630 is fine on a 90 npi head and impossible on a 180 npi one.
    rip.PrintheadConfig(dpi=630, head="c6n90")
    try:
        rip.PrintheadConfig(dpi=630, head="c4n180")
    except ValueError as exc:
        assert "180" in str(exc)
    else:
        raise AssertionError("630 dpi accepted on a 180 npi head")
    assert head_layout.get_layout("c4n180").default_dpi == 720
    assert head_layout.get_layout("c6n90").default_dpi == 630


def test_a_slot_that_cannot_tile_is_refused():
    """A slot that is not a whole multiple of the smallest one leaves rows no
    band step can reach; reject the layout, do not print gaps."""
    try:
        head_layout.HeadLayout(
            name="broken", nozzle_pitch_npi=180, column_length=180,
            channel_order=("C", "K"),
            slots=(head_layout.NozzleSlot(0, 0, 60),
                   head_layout.NozzleSlot(1, 0, 90)),
            default_inks=("K", "C"),
        )
    except ValueError as exc:
        assert "tile" in str(exc)
    else:
        raise AssertionError("accepted slots that cannot tile")


def test_overlapping_slots_are_refused():
    try:
        head_layout.HeadLayout(
            name="broken", nozzle_pitch_npi=180, column_length=180,
            channel_order=("C", "K"),
            slots=(head_layout.NozzleSlot(0, 0, 60),
                   head_layout.NozzleSlot(0, 0, 60)),
            default_inks=("K", "C"),
        )
    except ValueError as exc:
        assert "overlaps" in str(exc)
    else:
        raise AssertionError("accepted two slots on the same nozzles")


# ---------------------------------------------------------------------------
# Calibration targets (E6)
# ---------------------------------------------------------------------------

def test_nozzle_check_follows_the_head():
    config = rip.PrintheadConfig(dpi=720, head="c4n180")
    _, layout = rip.build_nozzle_check_channels(dpi=720, config=config)
    assert layout["channels"] == ["C", "M", "Y", "K"]
    assert layout["nozzle_counts"] == {ch: 60 for ch in CMYK}
    assert layout["head"] == "c4n180"
    # Only the nozzles a print fires are checked; the black column's 120
    # idle nozzles have no index in the dead-nozzle records.
    assert len(layout["nozzles"]) == 4 * 60

    reference = rip.build_nozzle_check_channels(dpi=630)[1]
    assert reference["channels"] == ["C", "M", "Y", "K", "LC", "LM"]
    assert reference["nozzle_counts"] == {ch: 90 for ch in reference["channels"]}


def test_every_nozzle_check_dash_is_printed_by_its_own_nozzle():
    """The comb is only readable if a dash is fired by exactly the nozzle it is
    labelled with; the target's row arithmetic and the pass slicer's have to
    agree, on either head."""
    for head, dpi in (("c6n90", 630), ("c4n180", 720)):
        config = rip.PrintheadConfig(dpi=dpi, head=head)
        _, layout = rip.build_nozzle_check_channels(dpi=dpi, config=config)
        height = layout["image_height_px"]
        geometry = config.geometry

        for entry in layout["nozzles"]:
            channel, nozzle = entry["channel"], entry["nozzle"]
            rows = _rows_a_nozzle_prints(config, channel, nozzle, height)
            dash_rows = np.zeros(height, dtype=bool)
            dash_rows[entry["y0"]:entry["y1"]] = True
            assert not (dash_rows & ~rows).any(), (
                f"{head}: {channel} dash {nozzle} covers rows nozzle {nozzle} "
                f"never prints"
            )


def test_col_align_measures_one_figure_per_column():
    """Inks sharing a column are at the same X by construction, so there is
    nothing to measure between them: c4n180 reduces to a single comparison."""
    reference, tests = rip.resolve_col_align_channels(
        rip.PrintheadConfig(dpi=630, head="c6n90"))
    assert (reference, tests) == ("M", ("C", "Y", "K"))

    config = rip.PrintheadConfig(dpi=720, head="c4n180")
    reference, tests = rip.resolve_col_align_channels(config)
    assert reference == "C" and tests == ("K",)
    # ... and the reference is the ink sharing the test's band, so the sandwich
    # stays common mode. M would not be: it starts 60 nozzles higher.
    geometry = config.geometry
    assert geometry.slot_for(reference).first_nozzle == \
        geometry.slot_for(tests[0]).first_nozzle
    assert geometry.slot_for("M").first_nozzle != geometry.slot_for("K").first_nozzle

    _, layout = rip.build_col_align_channels(dpi=720, config=config)
    assert layout["ref"] == "C" and layout["channels"] == ["K"]
    assert {cell["channel"] for cell in layout["cells"]} == {"K"}


def test_col_align_needs_two_inked_columns():
    config = rip.PrintheadConfig(
        dpi=720, head="c4n180",
        ink_map=("K", head_layout.EMPTY_SLOT, head_layout.EMPTY_SLOT,
                 head_layout.EMPTY_SLOT))
    try:
        rip.resolve_col_align_channels(config)
    except ValueError as exc:
        assert "two inked columns" in str(exc)
    else:
        raise AssertionError("col_align accepted a head with one inked column")


def test_payload_metadata_describes_the_geometry():
    config = rip.PrintheadConfig(dpi=720, head="c4n180")
    meta = config.geometry.to_metadata()
    assert meta["name"] == "c4n180"
    assert meta["nozzle_pitch_npi"] == 180
    assert meta["ink_map"] == ["K", "C", "M", "Y"]
    assert meta["lead_in_bands"] == 2
    assert [s["ink"] for s in meta["slots"]] == ["K", "C", "M", "Y"]
    assert [s["first_nozzle"] for s in meta["slots"]] == [0, 0, 60, 120]
    assert meta["slots"][0]["physical_nozzle_count"] == 180


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
