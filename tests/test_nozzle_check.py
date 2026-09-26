# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the nozzle-check target generator (rip.py --target nozzle_check).

Builds the target, slices it into passes exactly as production does, then
reconstructs the printed image by inverting the interleave and checks that
each nozzle's dash is carried by *that* nozzle, i.e. zeroing nozzle-row n in
the passes removes exactly nozzle n's dash (the premise the N-track masking
relies on). Runs under pytest or as a plain script.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "rip"))

import rip  # noqa: E402
import rip_payload  # noqa: E402

CH_INDEX = {"C": 0, "M": 1, "Y": 2, "K": 3, "LC": 4, "LM": 5}


def _reconstruct(passes, meta):
    """Reconstruct the printed *image* from the passes: the sheet as it comes
    off the bed (and as a scan sees it), which metadata.target is in.

    The slicer flips the image onto the bed (mirrored-prints fix): machine line
    L = b*lpb + off + n*ppb carries image row H-1-L. Undoing that flip here
    yields image space, so this validates that metadata.target matches the
    real printed pixels (nozzle n's dash lands in nozzle n's recorded box)."""
    ppb = meta["passes_per_band"]
    nozzle = meta["nozzle_count"]
    H, W = meta["image_height_px"], meta["image_width_px"]
    lpb = nozzle * ppb
    img = {ch: np.zeros((H, W), dtype=np.uint8) for ch in CH_INDEX}
    for i, p in enumerate(passes):
        b, off = divmod(i, ppb)
        for n in range(nozzle):
            line = b * lpb + off + n * ppb
            if line < H:
                for ch, idx in CH_INDEX.items():
                    img[ch][H - 1 - line, :] = p[idx, n, :]
    return img


def _build_and_reconstruct(dpi):
    with tempfile.TemporaryDirectory() as td:
        payload = Path(td) / "nz.json"
        rip.process_target_for_printing(str(payload), kind="nozzle_check", dpi=dpi)
        loaded = rip_payload.load_rip(str(payload))
        header = __import__("json").loads(payload.read_text())
        return header["metadata"], loaded["passes"]["data"]


def test_every_nozzle_dash_is_present_and_isolated():
    for dpi in (90, 630):
        meta, passes = _build_and_reconstruct(dpi)
        layout = meta["target"]
        img = _reconstruct(passes, meta)

        for nz in layout["nozzles"]:
            region = img[nz["channel"]][nz["y0"]:nz["y1"], nz["x0"]:nz["x1"]]
            assert region.size > 0 and region.min() == 1, \
                f"dpi {dpi}: {nz['channel']} nozzle {nz['nozzle']} dash not solid"

        # Fiducials solid in K.
        for f in layout["fiducials"]:
            assert img["K"][f["y0"]:f["y1"], f["x0"]:f["x1"]].min() == 1


def test_zeroing_a_nozzle_row_removes_exactly_that_dash():
    # The N-track masking premise: nozzle index is a raster dimension, so
    # zeroing nozzle-row n in every pass must blank exactly nozzle n's dash.
    meta, passes = _build_and_reconstruct(90)
    layout = meta["target"]
    target = next(nz for nz in layout["nozzles"] if nz["channel"] == "M" and nz["nozzle"] == 40)
    ci = CH_INDEX["M"]

    masked = [p.copy() for p in passes]
    for p in masked:
        p[ci, 40, :] = 0  # kill nozzle 40 of channel M everywhere

    img = _reconstruct(masked, meta)
    # Nozzle 40's dash is now gone...
    reg = img["M"][target["y0"]:target["y1"], target["x0"]:target["x1"]]
    assert reg.max() == 0, "masked nozzle dash still present"
    # ...but its neighbours are untouched.
    for other in (39, 41):
        nz = next(n for n in layout["nozzles"] if n["channel"] == "M" and n["nozzle"] == other)
        reg2 = img["M"][nz["y0"]:nz["y1"], nz["x0"]:nz["x1"]]
        assert reg2.min() == 1, f"neighbour nozzle {other} damaged by the mask"


# --- recording tool (tools/record_dead_nozzles.py) ----------------------------

sys.path.insert(0, str(REPO / "tools"))
import record_dead_nozzles as rdn  # noqa: E402


def test_parse_dead():
    assert rdn.parse_dead("C:3,17;M:42") == {"C": [3, 17], "M": [42]}
    assert rdn.parse_dead(" K : 30 ; ;Y:11,10") == {"K": [30], "Y": [10, 11]}
    assert rdn.parse_dead("") == {}


def test_update_profile_stores_dead_by_slot():
    """Nozzles are named per ink but must persist per physical slot, via the
    profile's ink_map, so the data survives re-plumbing."""
    import json
    # Reference plumbing: slot index of each ink (K,Y,LM,LC,C,M left to right).
    dead = {"C": [5], "K": [12, 13], "LC": [7]}
    with tempfile.TemporaryDirectory() as td:
        prof = Path(td) / "profile.json"
        prof.write_text(json.dumps({"media": "t", "dead_nozzles": {}}))
        out = Path(td) / "out.json"
        rdn.update_profile(str(prof), str(out), dead)
        data = json.loads(out.read_text())

    # ink_map stamped (reference plumbing) and dead stored against slots:
    # K -> slot 0, C -> slot 4, LC -> slot 3.
    assert data["ink_map"] == ["K", "Y", "LM", "LC", "C", "M"]
    assert data["dead_nozzles"] == {"0": [12, 13], "4": [5], "3": [7]}

    # And the RIP reads it back into the same channel-keyed view compensation uses.
    profile = rip.CalibrationProfile.from_dict(data)
    assert profile.dead_nozzles == {"K": [12, 13], "C": [5], "LC": [7]}


def test_profile_dead_survives_replumbing():
    """Swapping two inks in the map re-attributes the same physical slot defect
    to the newly-plumbed ink; the slot-keyed store is plumbing-independent."""
    slot_keyed = {"media": "t", "ink_map": ["K", "Y", "LM", "LC", "M", "C"],
                  "dead_nozzles": {"4": [9]}}  # slot 4 now carries M, not C
    profile = rip.CalibrationProfile.from_dict(slot_keyed)
    assert profile.dead_nozzles == {"M": [9]}


if __name__ == "__main__":
    test_every_nozzle_dash_is_present_and_isolated()
    print("PASS test_every_nozzle_dash_is_present_and_isolated")
    test_zeroing_a_nozzle_row_removes_exactly_that_dash()
    print("PASS test_zeroing_a_nozzle_row_removes_exactly_that_dash")
    test_parse_dead()
    print("PASS test_parse_dead")
    test_update_profile_stores_dead_by_slot()
    print("PASS test_update_profile_stores_dead_by_slot")
    test_profile_dead_survives_replumbing()
    print("PASS test_profile_dead_survives_replumbing")
