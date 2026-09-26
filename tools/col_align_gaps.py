# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Turn a by-eye reading of a printed col_align target into encoder gaps.

Print ``rip.py --target col_align`` through the normal encode path, with the
``--column-gap`` / ``--group-gap`` you want to check. Each test channel prints
a row of sandwich cells against the magenta reference, one per commanded
offset ``k`` on the ruler. Find the straightest sandwich per channel (a loupe
helps), read its ``k``, and pass the readings here:

    python tools/col_align_gaps.py "C:-2;Y:3;K:5" --payload col_align.json \\
        [--column-gap 0.842] [--group-gap 7.051] [--json OUT.json]

The straightest cell at ``k`` means that shift cancelled the error, so the
residual is ``-k`` px. From it the tool reports the true distance of each
colour's columns behind the reference, then least-squares fits the encoder's
two layout parameters and prints ready-to-paste flags.

The gaps passed here must be the ones the target was ENCODED with, so the
integer-pixel offsets the encoder applied can be reconstructed exactly (it
ceil()s each gap to pixels *before* combining them).

If no sandwich is straight, the configured gaps are off by more than the sweep
covers: reprint with a wider sweep, e.g. ``--span-mm 2 --cell-pitch-mm 5``.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Distance behind the reference column (M) per channel, as coefficients
# (a, b) of a*column_gap + b*group_gap, the encoder's PrintheadLayout
# model (physical order K -> Y -> LM -> LC -> C -> M, M leading).
MODEL_ROWS: Dict[str, Tuple[int, int]] = {
    "C": (1, 0), "LC": (0, 1), "LM": (1, 1), "Y": (0, 2), "K": (1, 2),
}


def applied_offsets_px(dpi: int, column_gap_mm: float, group_gap_mm: float) -> Dict[str, int]:
    """The integer-pixel delays the encoder applied, reconstructed exactly:
    each gap is ceil()ed to pixels *separately*, then combined."""
    scale = dpi / 25.4
    col = math.ceil(column_gap_mm * scale)
    grp = math.ceil(group_gap_mm * scale)
    return {ch: a * col + b * grp for ch, (a, b) in MODEL_ROWS.items()}


def load_target(payload_path: str) -> Tuple[int, Dict]:
    """``(dpi, layout)`` from a col_align payload's ``metadata.target``."""
    meta = json.loads(Path(payload_path).read_text()).get("metadata", {})
    layout = meta.get("target")
    if not layout or layout.get("kind") != "col_align":
        raise ValueError(f"{payload_path} is not a col_align payload")
    return int(layout.get("dpi", meta.get("dpi"))), layout


def parse_readings(spec: str) -> Dict[str, float]:
    """Parse 'C:-2;Y:3;K:5' (straightest k per channel) -> {'C': -2.0, ...}."""
    out: Dict[str, float] = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        channel, _, k = part.partition(":")
        out[channel.strip()] = float(k)
    return out


def summarise(
    residuals_px: Dict[str, float],
    dpi: int,
    column_gap_mm: float,
    group_gap_mm: float,
) -> Dict:
    """Turn per-channel residuals (px) into true distances and a
    least-squares (column_gap, group_gap) fit."""
    px_mm = 25.4 / dpi
    applied = applied_offsets_px(dpi, column_gap_mm, group_gap_mm)

    channels: Dict[str, Dict] = {}
    for ch, r in residuals_px.items():
        channels[ch] = {
            "residual_px": float(r),
            "applied_px": applied[ch],
            "true_distance_mm": applied[ch] * px_mm - r * px_mm,
        }

    fit: Optional[Dict] = None
    fit_rows = [ch for ch in channels if ch in MODEL_ROWS]
    if len(fit_rows) >= 2:
        a = np.array([MODEL_ROWS[ch] for ch in fit_rows], dtype=np.float64)
        y = np.array([channels[ch]["true_distance_mm"] for ch in fit_rows])
        if np.linalg.matrix_rank(a) == 2:
            (cg, gg), *_ = np.linalg.lstsq(a, y, rcond=None)
            fit = {"column_gap_mm": float(cg), "group_gap_mm": float(gg),
                   "model_residual_um": {}}
            for ch in fit_rows:
                ai, bi = MODEL_ROWS[ch]
                err = (ai * cg + bi * gg) - channels[ch]["true_distance_mm"]
                fit["model_residual_um"][ch] = float(err * 1000.0)
            # The encoder quantises each gap with ceil(): a flag that lands
            # exactly on (or is rounded a hair above) a pixel boundary
            # overshoots the intended offset by one whole pixel. Real heads
            # sit on the nozzle-pitch grid, so the fit often IS a boundary;
            # recommend flags a hair below the fitted distances.
            fit["flag_column_gap_mm"] = max(float(cg) - 0.005, 0.0)
            fit["flag_group_gap_mm"] = max(float(gg) - 0.005, 0.0)

    return {"dpi": dpi, "px_mm": px_mm, "channels": channels, "fit": fit}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fit encoder column gaps from a by-eye col_align reading")
    ap.add_argument("readings",
                    help='Straightest k per channel, read off the ruler, e.g. "C:-2;Y:3;K:5"')
    ap.add_argument("--payload", required=True,
                    help="RIP payload JSON of the printed col_align target")
    ap.add_argument("--column-gap", type=float, default=0.842,
                    help="column_gap_mm the target was ENCODED with "
                         "(default mirrors the encoder default: 0.842)")
    ap.add_argument("--group-gap", type=float, default=7.051,
                    help="group_gap_mm the target was ENCODED with "
                         "(default mirrors the encoder default: 7.051)")
    ap.add_argument("--json", default=None, help="Write the report as JSON here")
    args = ap.parse_args()

    try:
        dpi, layout = load_target(args.payload)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc))

    readings = parse_readings(args.readings)
    if not readings:
        raise SystemExit("no readings given (expected e.g. \"C:-2;Y:3;K:5\")")
    unknown = sorted(set(readings) - set(MODEL_ROWS))
    if unknown:
        raise SystemExit(f"no column model for {', '.join(unknown)} "
                         f"(known: {', '.join(sorted(MODEL_ROWS))})")
    # Straightest cell at k_best means the commanded shift cancelled the
    # residual: r = -k_best.
    report = summarise({ch: -k for ch, k in readings.items()},
                       dpi, args.column_gap, args.group_gap)

    px_um = report["px_mm"] * 1000.0
    print(f"col_align @ {dpi} dpi (1 px = {px_um:.1f} um), encoded with "
          f"column_gap {args.column_gap} mm / group_gap {args.group_gap} mm")
    print(f"   {'ch':<3} {'residual':>12} {'applied':>9} {'true distance':>14}")
    for ch, row in sorted(report["channels"].items()):
        print(f"   {ch:<3} "
              f"{row['residual_px']:>+7.2f} px "
              f"{row['applied_px']:>6d} px "
              f"{row['true_distance_mm']:>11.3f} mm")

    fit = report["fit"]
    if fit:
        # Heads are manufactured on the nozzle-pitch grid: fitted distances
        # landing on integer pitches corroborate the reading.
        pitch_mm = 25.4 * layout["passes_per_band"] / dpi
        print(f"\n   fit: column_gap = {fit['column_gap_mm']:.3f} mm "
              f"({fit['column_gap_mm'] / pitch_mm:.2f} nozzle pitches), "
              f"group_gap = {fit['group_gap_mm']:.3f} mm "
              f"({fit['group_gap_mm'] / pitch_mm:.2f} pitches)")
        worst = 0.0
        for ch, err in sorted(fit["model_residual_um"].items()):
            print(f"        {ch}: model residual {err:+.0f} um "
                  f"({err / px_um:+.2f} px)")
            worst = max(worst, abs(err) / px_um)
        if worst > 0.5:
            print("   WARNING: the two-parameter layout model misses by more "
                  "than half a pixel for some channel: the head's columns "
                  "are not uniformly spaced. Per-channel offsets would need "
                  "encoder support; until then pick the gaps that favour the "
                  "channels you care about.")
        print(f"\n   encoder flags: --column-gap {fit['flag_column_gap_mm']:.3f} "
              f"--group-gap {fit['flag_group_gap_mm']:.3f}")
        print("   (a hair below the fit: the encoder ceil()s each gap, and a "
              "flag on a pixel boundary would overshoot by 1 px)")
    else:
        print("   (fewer than two model channels read: no gap fit)")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\n   report written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
