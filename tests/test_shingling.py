# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for R8 shingling: splitting each swath into column-interleaved sweeps
for non-absorbent media."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "rip"))

import rip  # noqa: E402


def test_shingling_is_lossless_and_disjoint():
    rng = np.random.default_rng(0)
    passes = [rng.integers(0, 2, size=(6, 8, 20), dtype=np.uint8) for _ in range(3)]
    ys = [0.0, 0.04, 0.08]

    out_p, out_y = rip.apply_shingling(passes, ys, 2)
    assert len(out_p) == len(out_y) <= 2 * len(passes)

    for p, y in zip(passes, ys):
        subs = [sp for sp, sy in zip(out_p, out_y) if sy == y]
        # Union of the sub-sweeps reproduces the swath exactly (nothing lost).
        recon = np.zeros_like(p)
        for sp in subs:
            recon |= sp
        assert np.array_equal(recon, p)
        # ...and the sub-sweeps are column-disjoint (no drop fired twice).
        assert sum(int(sp.sum()) for sp in subs) == int(p.sum())

    # N=1 is a no-op.
    same_p, same_y = rip.apply_shingling(passes, ys, 1)
    assert len(same_p) == len(passes) and same_y == ys


def test_pipeline_shingling_doubles_sweeps():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        grad = np.tile(np.linspace(40, 255, 24, dtype=np.uint8), (16, 1))
        img = tmp / "in.png"
        Image.fromarray(grad, mode="L").convert("RGB").save(img)

        prof = rip.CalibrationProfile.default()
        prof.dpi_curves = {}
        base = tmp / "base.json"
        rip.process_image_for_printing(str(img), str(base), dpi=90, calibration=prof)
        n_base = json.loads(base.read_text())["metadata"]["total_passes"]

        prof.shingle_passes = 2
        sh = tmp / "sh.json"
        rip.process_image_for_printing(str(img), str(sh), dpi=90, calibration=prof)
        meta = json.loads(sh.read_text())["metadata"]
        assert meta["processing"]["shingle_passes"] == 2
        # Each non-empty swath becomes 2 sweeps (a solid-ish gradient has ink in
        # both column classes everywhere), so the count roughly doubles.
        assert meta["total_passes"] > n_base


if __name__ == "__main__":
    test_shingling_is_lossless_and_disjoint()
    print("PASS test_shingling_is_lossless_and_disjoint")
    test_pipeline_shingling_doubles_sweeps()
    print("PASS test_pipeline_shingling_doubles_sweeps")
