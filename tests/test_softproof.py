# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the R9 soft-proof + ink estimation in viewer.py."""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "rip"))

import viewer  # noqa: E402


def test_drop_volume_sphere():
    # 40 um sphere: (pi/6) d^3 / 1000 pL.
    assert abs(viewer.drop_volume_pl(40.0) - 33.5) < 0.5


def test_estimate_ink_counts_dots():
    bmps = {
        "C": np.ones((10, 10), np.uint8),          # 100 dots
        "M": np.zeros((10, 10), np.uint8),         # 0
        "Y": (np.indices((10, 10)).sum(0) % 2).astype(np.uint8),  # 50 dots
    }
    ink = viewer.estimate_ink(bmps, drop_pl=10.0)
    assert ink["C"][0] == 100 and ink["M"][0] == 0 and ink["Y"][0] == 50
    # volume = dots * pL / 1e6 uL
    assert abs(ink["C"][1] - 100 * 10.0 / 1e6) < 1e-9


def test_soft_proof_reflects_channel_ink():
    h = w = 40
    solid_cyan = {"C": np.ones((h, w), np.uint8),
                  "M": np.zeros((h, w), np.uint8),
                  "Y": np.zeros((h, w), np.uint8),
                  "K": np.zeros((h, w), np.uint8)}
    proof = viewer.render_soft_proof(solid_cyan, dpi=630, drop_diameter_um=40.0, max_dim_px=64)
    arr = np.asarray(proof, dtype=np.float32)
    mean = arr.reshape(-1, 3).mean(axis=0)   # (R, G, B)
    # Cyan absorbs red: R should be much lower than G and B.
    assert mean[0] < mean[1] - 40 and mean[0] < mean[2] - 40


if __name__ == "__main__":
    test_drop_volume_sphere()
    print("PASS test_drop_volume_sphere")
    test_estimate_ink_counts_dots()
    print("PASS test_estimate_ink_counts_dots")
    test_soft_proof_reflects_channel_ink()
    print("PASS test_soft_proof_reflects_channel_ink")
