# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the col_align target (rip.py) and tools/col_align_gaps.py.

Layout invariants first (the sandwich rows must align to whole nozzles of a
single band, or the ref/test comparison stops being sweep-coherent), then the
reading-to-gaps arithmetic: the tool must reconstruct exactly the offsets the
encoder applied, and a by-eye reading of a known geometry must fit it back.

Runs under pytest or as a plain script.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "rip"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "encoder"))

import rip  # noqa: E402
import col_align_gaps as colal  # noqa: E402
import encoder  # noqa: E402


# ---------------------------------------------------------------------------
# Layout invariants
# ---------------------------------------------------------------------------

def _machine_band(layout, y0, y1):
    """Band index of an image-row span, asserting whole-nozzle alignment."""
    height = layout["image_height_px"]
    ppb = layout["passes_per_band"]
    lpb = layout["lines_per_band"]
    lo, hi = height - y1, height - y0
    assert lo % ppb == 0 and hi % ppb == 0, "segment not whole nozzles"
    assert lo // lpb == (hi - 1) // lpb, "segment crosses a band boundary"
    return lo // lpb


def test_layout_invariants():
    for dpi in (90, 360, 630):
        planes, layout = rip.build_col_align_channels(dpi=dpi)
        width, height = layout["image_width_px"], layout["image_height_px"]
        for plane in planes.values():
            assert plane.shape == (height, width)

        for cell in layout["cells"]:
            boxes = [cell["test"]] + cell["refs"]
            bands = set()
            for box in boxes:
                assert 0 <= box["x0"] < box["x1"] <= width
                assert 0 <= box["y0"] < box["y1"] <= height
                bands.add(_machine_band(layout, box["y0"], box["y1"]))
            # The whole sandwich shares one band (sweep coherence), above
            # the yaw band (machine band 0).
            assert bands == {1 + cell["repeat"]}

            # Test mark commanded at ref x + k.
            ref_cx = (cell["refs"][0]["x0"] + cell["refs"][0]["x1"]) / 2
            test_cx = (cell["test"]["x0"] + cell["test"]["x1"]) / 2
            assert test_cx - ref_cx == cell["k"]

            # Test segment vertically centred between the two refs, so
            # linear head yaw cancels in the two-ref average.
            refs_cy = np.mean([(r["y0"] + r["y1"]) / 2 for r in cell["refs"]])
            test_cy = (cell["test"]["y0"] + cell["test"]["y1"]) / 2
            assert abs(test_cy - refs_cy) < 1e-9

            # Marks are actually painted, in the right planes.
            t = cell["test"]
            assert (planes[cell["channel"]][t["y0"]:t["y1"], t["x0"]:t["x1"]] == 255).all()
            for r in cell["refs"]:
                assert (planes[layout["ref"]][r["y0"]:r["y1"], r["x0"]:r["x1"]] == 255).all()

        for line in layout["yaw_lines"]:
            assert _machine_band(layout, line["y0"], line["y1"]) == 0


def test_applied_offsets_match_encoder():
    """The tool must reconstruct exactly the offsets the encoder applies."""
    for dpi in (360, 630, 1440):
        for cg, gg in ((0.9, 7.0), (0.83, 7.09), (1.2, 6.5)):
            enc = encoder.PrintheadLayout(
                column_gap_mm=cg, group_gap_mm=gg
            ).calculate_channel_offsets_px(dpi)
            tool = colal.applied_offsets_px(dpi, cg, gg)
            for ch, off in tool.items():
                assert off == enc[ch], (dpi, cg, gg, ch)
            assert enc["M"] == 0


# ---------------------------------------------------------------------------
# From a reading to the gaps
# ---------------------------------------------------------------------------

def test_a_reading_of_a_known_geometry_fits_it_back():
    """Encode with one pair of gaps, pretend the head really sits at another,
    and read the k that would look straightest (the nearest whole pixel)."""
    dpi, enc_cg, enc_gg = 180, 0.9, 7.0
    true_cg, true_gg = 0.83, 7.09
    scale = dpi / 25.4
    applied = colal.applied_offsets_px(dpi, enc_cg, enc_gg)
    true_d = {"C": true_cg, "Y": 2 * true_gg, "K": 2 * true_gg + true_cg}
    readings = {ch: -round(applied[ch] - d * scale) for ch, d in true_d.items()}

    report = colal.summarise({ch: -k for ch, k in readings.items()},
                             dpi, enc_cg, enc_gg)
    fit = report["fit"]
    px_mm = 25.4 / dpi
    # A by-eye reading is whole pixels, so the fit is good to about one.
    assert abs(fit["column_gap_mm"] - true_cg) < px_mm
    assert abs(fit["group_gap_mm"] - true_gg) < px_mm


def test_load_target_refuses_other_payloads():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "nz.json"
        path.write_text(json.dumps({"metadata": {"target": {"kind": "nozzle_check"}}}))
        try:
            colal.load_target(str(path))
        except ValueError as exc:
            assert "col_align" in str(exc)
        else:
            raise AssertionError("accepted a nozzle_check payload")


def test_encoder_default_layout_is_calibrated():
    """The encoder defaults are the measured c6n90 geometry (3 / 25 nozzle
    pitches of 1/90", guarded 5 um below the ceil() boundary), so the applied
    offsets land exactly on the design grid at every pitch-aligned DPI."""
    layout = encoder.PrintheadLayout()
    for dpi in (360, 630, 720, 900):
        pitch_px = dpi // 90  # one nozzle pitch (1/90") in pixels
        off = layout.calculate_channel_offsets_px(dpi)
        assert off["M"] == 0
        assert off["C"] == 3 * pitch_px
        assert off["LC"] == 25 * pitch_px
        assert off["LM"] == 28 * pitch_px
        assert off["Y"] == 50 * pitch_px
        assert off["K"] == 53 * pitch_px


def test_reading_math():
    assert colal.parse_readings("C:-2;Y:3;K:5") == {"C": -2.0, "Y": 3.0, "K": 5.0}
    # k_best = -2 means the -2 px commanded shift straightened the sandwich,
    # so the residual is +2 px and the true distance is applied - 2 px.
    report = colal.summarise({"C": 2.0}, 360, 0.9, 7.0)
    applied = colal.applied_offsets_px(360, 0.9, 7.0)
    expected = (applied["C"] - 2.0) * 25.4 / 360
    assert abs(report["channels"]["C"]["true_distance_mm"] - expected) < 1e-9


if __name__ == "__main__":
    test_layout_invariants()
    test_applied_offsets_match_encoder()
    test_a_reading_of_a_known_geometry_fits_it_back()
    test_load_target_refuses_other_payloads()
    test_encoder_default_layout_is_calibrated()
    test_reading_math()
    print("OK")
