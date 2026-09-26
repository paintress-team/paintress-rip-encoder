# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for dead-nozzle compensation (N2): reroute (N2a) and retouch (N2b).

The retouch test physically simulates the print: it deposits each pass's
nozzles into an image but SKIPS dead nozzles (they eject nothing), and
checks the dead rows end up covered by a healthy neighbour without double-
printing the healthy rows. Runs under pytest or as a plain script.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "rip"))

import rip  # noqa: E402

CH = ("C", "M", "Y", "K")
CI = {"C": 0, "M": 1, "Y": 2, "K": 3}


def _simulate_print(passes, y_positions, dead, config, H, W):
    """Deposit passes into an image, skipping dead nozzles (physical print)."""
    ppb = config.passes_per_band
    spacing = config.line_spacing_mm
    dep = {ch: np.zeros((H, W), dtype=np.uint8) for ch in CH}
    for p, y in zip(passes, y_positions):
        first_line = int(round(y / spacing))
        for ch in CH:
            dead_set = set(dead.get(ch, []))
            for j in range(config.nozzle_count):
                if j in dead_set:
                    continue  # dead nozzle ejects nothing
                row = H - 1 - (first_line + j * ppb)
                if 0 <= row < H:
                    dep[ch][row, :] |= p[CI[ch], j, :]
    return dep


def _solid_halftone(H, W):
    return {"C": np.ones((H, W), np.uint8), "M": np.zeros((H, W), np.uint8),
            "Y": np.zeros((H, W), np.uint8), "K": np.zeros((H, W), np.uint8)}


def test_retouch_covers_dead_rows_via_healthy_neighbour():
    dpi = 180                      # ppb = 2 (dead nozzle owns 2 consecutive rows)
    config = rip.PrintheadConfig(dpi=dpi)
    H, W = 2 * config.lines_per_band, 16   # 2 full bands
    halftone = _solid_halftone(H, W)       # every row wants ink
    dead = {"C": [45]}             # one interior nozzle dead (neighbours healthy)
    masks = rip.dead_nozzle_row_masks(dead, H, config)
    dead_rows = masks["C"]
    assert dead_rows.sum() == 2 * config.passes_per_band  # 2 rows/band * 2 bands

    passes = rip.generate_print_passes(halftone, config)
    y = rip.compute_pass_y_positions_mm(H / dpi * 25.4, config)

    # Without compensation: the dead nozzle's rows are blank.
    dep0 = _simulate_print(passes, y, dead, config, H, W)
    assert dep0["C"][dead_rows].sum() == 0, "dead rows unexpectedly printed"
    assert dep0["C"][~dead_rows].all(), "healthy rows should all be printed"

    # With retouch: dead rows covered, healthy rows unchanged (no double print).
    rpasses, ry = rip.generate_retouch_passes(halftone, dead, config, y)
    dep1 = _simulate_print(passes + rpasses, list(y) + ry, dead, config, H, W)
    assert dep1["C"][dead_rows].all(), "retouch did not cover the dead rows"
    assert dep1["C"].all(), "retouch left gaps or damaged healthy rows"


def test_classify_retouch_donors_picks_nearest_healthy():
    nc = 90
    # Isolated nozzle borrows from n-1.
    up, down, skip = rip.classify_retouch_donors({"C": [45]}, nc)
    assert up == {"C": [45]} and not down and not skip
    # Edge nozzle 0 has no n-1 -> borrows from n+1.
    up, down, skip = rip.classify_retouch_donors({"C": [0]}, nc)
    assert down == {"C": [0]} and not up and not skip
    # Edge nozzle 89 has no n+1 -> borrows from n-1.
    up, down, skip = rip.classify_retouch_donors({"C": [89]}, nc)
    assert up == {"C": [89]} and not down and not skip
    # Cluster: only the edges have a healthy neighbour; the interior is skipped.
    up, down, skip = rip.classify_retouch_donors({"C": [40, 41, 42]}, nc)
    assert up == {"C": [40]} and down == {"C": [42]} and skip == {"C": [41]}


def test_retouch_covers_cluster_edges_and_skips_interior():
    dpi = 180
    config = rip.PrintheadConfig(dpi=dpi)
    H, W = 2 * config.lines_per_band, 16
    halftone = _solid_halftone(H, W)
    dead = {"C": [40, 41, 42]}      # 40 via 39 (up), 42 via 43 (down), 41 unreachable
    passes = rip.generate_print_passes(halftone, config)
    y = rip.compute_pass_y_positions_mm(H / dpi * 25.4, config)

    rpasses, ry = rip.generate_retouch_passes(halftone, dead, config, y)
    dep = _simulate_print(passes + rpasses, list(y) + ry, dead, config, H, W)

    rows = {n: rip.dead_nozzle_row_masks({"C": [n]}, H, config)["C"] for n in (40, 41, 42)}
    assert dep["C"][rows[40]].all(), "edge nozzle 40 not covered by donor 39"
    assert dep["C"][rows[42]].all(), "edge nozzle 42 not covered by donor 43"
    assert dep["C"][rows[41]].sum() == 0, "interior nozzle 41 should stay uncompensated"


def test_reroute_blanks_dead_rows_and_preserves_ink():
    dpi = 180
    config = rip.PrintheadConfig(dpi=dpi)
    H, W = config.lines_per_band, 64
    field = np.full((H, W), 128, np.uint8)   # mid grey
    dead = {"C": [20]}
    masks = rip.dead_nozzle_row_masks(dead, H, config)

    plain = rip.apply_floyd_steinberg(field)
    rerouted = rip.apply_floyd_steinberg(field, dead_rows=masks["C"])

    dead_rows = masks["C"]
    assert rerouted[dead_rows].sum() == 0, "reroute left dots on dead rows"
    # Ink isn't lost: the dead rows' dots move to neighbours, so total is close.
    assert abs(int(rerouted.sum()) - int(plain.sum())) < 0.05 * plain.sum()
    # Rows adjacent to the dead band gain ink vs the plain dither.
    below = np.where(dead_rows)[0].max() + 1
    if below < H:
        assert rerouted[below].sum() >= plain[below].sum()


def test_limit_reroute_ink_keeps_local_drops_far():
    reach = 2
    H, W = 40, 4
    channel = np.full((H, W), 200, np.uint8)
    dead = np.zeros(H, bool)
    dead[10:12] = True        # isolated dead nozzle (2 rows) -> healthy within reach
    dead[20:32] = True        # big cluster (12 rows) -> interior unreachable

    out = rip.limit_reroute_ink(channel, dead, reach)
    assert (out[10:12] == 200).all(), "isolated dead rows should keep their ink"
    assert (out[20:22] == 200).all() and (out[30:32] == 200).all(), "cluster edges kept"
    assert (out[24:28] == 0).all(), "deep cluster interior should be dropped"
    assert (out[0:10] == 200).all() and (out[32:] == 200).all(), "healthy rows untouched"


def test_reroute_confines_ink_below_a_dark_cluster():
    # A dark band with a big dead cluster over a light region. Without the
    # limit, the dead rows' ink cascades down and dumps far into the light
    # region; the limit keeps it local, so the light region stays clean.
    dpi = 180
    config = rip.PrintheadConfig(dpi=dpi)
    reach = config.passes_per_band
    H, W = 80, 32
    field = np.full((H, W), 15, np.uint8)     # light everywhere
    field[10:40, :] = 255                     # dark band
    dead = np.zeros(H, bool)
    dead[14:36] = True                        # big cluster inside the dark band
    below = slice(46, 70)                     # light region, well below the band

    plain = rip.apply_floyd_steinberg(field, dead_rows=dead.astype(np.uint8))
    limited = rip.apply_floyd_steinberg(
        rip.limit_reroute_ink(field, dead, reach), dead_rows=dead.astype(np.uint8))
    assert limited[below].sum() < plain[below].sum(), "limit did not curb the far cascade"


if __name__ == "__main__":
    test_retouch_covers_dead_rows_via_healthy_neighbour()
    print("PASS test_retouch_covers_dead_rows_via_healthy_neighbour")
    test_classify_retouch_donors_picks_nearest_healthy()
    print("PASS test_classify_retouch_donors_picks_nearest_healthy")
    test_retouch_covers_cluster_edges_and_skips_interior()
    print("PASS test_retouch_covers_cluster_edges_and_skips_interior")
    test_reroute_blanks_dead_rows_and_preserves_ink()
    print("PASS test_reroute_blanks_dead_rows_and_preserves_ink")
    test_limit_reroute_ink_keeps_local_drops_far()
    print("PASS test_limit_reroute_ink_keeps_local_drops_far")
    test_reroute_confines_ink_below_a_dark_cluster()
    print("PASS test_reroute_confines_ink_below_a_dark_cluster")
