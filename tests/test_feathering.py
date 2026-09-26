# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for R7 band-boundary feathering: overlapping bands with a stochastic
seam must still print every pixel exactly once (no gap, no double)."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "rip"))

import rip  # noqa: E402

CHANNELS = ("C", "M", "Y", "K")


def _reconstruct(passes, y_positions, config, H, W):
    """Count how many times each image pixel is deposited across all passes."""
    ppb = config.passes_per_band
    spacing = config.line_spacing_mm
    counts = np.zeros((len(CHANNELS), H, W), dtype=int)
    for p, y in zip(passes, y_positions):
        first_line = int(round(y / spacing))
        for n in range(config.nozzle_count):
            frow = first_line + n * ppb
            row = H - 1 - frow
            if 0 <= row < H:
                counts[:, row, :] += p[:, n, :]
    return counts


def test_feathering_prints_every_pixel_once():
    dpi = 180                                   # ppb = 2, lpb = 180
    config = rip.PrintheadConfig(dpi=dpi)
    H, W = 10 * config.lines_per_band, 16       # several full bands
    rng = np.random.default_rng(7)
    halftone = {ch: rng.integers(0, 2, size=(H, W), dtype=np.uint8) for ch in CHANNELS}
    stacked = np.stack([halftone[ch] for ch in CHANNELS])

    for overlap in (0, 14):
        passes = rip.generate_print_passes(halftone, config, band_overlap=overlap)
        y = rip.compute_pass_y_positions_mm(H / dpi * 25.4, config, overlap)
        assert len(passes) == len(y)
        counts = _reconstruct(passes, y, config, H, W)
        # Every pixel deposited exactly once -> counts == the halftone (0/1),
        # so nothing is lost (gap) and nothing is doubled.
        assert np.array_equal(counts, stacked), f"overlap={overlap}: not lossless"

    # Feathering advances less per band, so it uses more (shorter) bands.
    n_plain = len(rip.generate_print_passes(halftone, config, band_overlap=0))
    n_feather = len(rip.generate_print_passes(halftone, config, band_overlap=14))
    assert n_feather > n_plain


def test_pipeline_band_overlap_records_and_covers():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        grad = np.tile(np.linspace(30, 255, 24, dtype=np.uint8), (200, 1))
        img = tmp / "in.png"
        Image.fromarray(grad, mode="L").convert("RGB").save(img)

        prof = rip.CalibrationProfile.default()
        prof.dpi_curves = {}
        prof.band_overlap = 10
        out = tmp / "job.json"
        rip.process_image_for_printing(str(img), str(out), dpi=90, calibration=prof)
        meta = json.loads(out.read_text())["metadata"]
        assert meta["processing"]["band_overlap"] == 10
        assert meta["total_passes"] > 0


if __name__ == "__main__":
    test_feathering_prints_every_pixel_once()
    print("PASS test_feathering_prints_every_pixel_once")
    test_pipeline_band_overlap_records_and_covers()
    print("PASS test_pipeline_band_overlap_records_and_covers")
