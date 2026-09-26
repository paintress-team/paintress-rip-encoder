# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build per-channel linearisation LUTs from a flatbed scan of a wedge target.

Second half of the R1 measured-linearisation workflow. Given a scan of a
printed ``--target wedge`` (see ``rip.py``) and the RIP payload that produced
it, this tool:

    1. Loads the exact patch geometry from the payload's ``metadata.target``
       (nothing about the layout is hard-coded here).
    2. Finds the four solid-K corner fiducials in the scan and solves the
       affine map from target space to scan space (handles translation,
       rotation, scale and a little shear from a hand-placed print).
    3. Samples the mean scanner response of every patch in the channel that
       ink absorbs (C->red, M->green, Y->blue, K/neutral->luminance),
       normalised by the paper white (the 0 % patch).
    4. Converts each series' response to perceptual tone (L*) and inverts the
       commanded-coverage -> measured-tone curve into a 256-entry LUT that
       maps a desired channel value to the coverage that prints it linearly.
    5. Writes the LUTs into the calibration profile under
       ``dpi_curves[<dpi>]`` so the RIP can linearise (R1 task) in place of
       the dot-gain gamma and the DPI ink power.

A flatbed scanner is not colorimetric, but for a *monotonic per-channel*
linearisation, consistency (auto-everything off, fixed settings between
sessions) is what matters, not absolute colour.

Usage:
    python tools/calibrate_from_scan.py SCAN.png --payload wedge_630.json \\
        --profile rip/profiles/default.json [--out rip/profiles/default.json] \\
        [--preview overlay.png]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

# Which scanner channel each ink primarily absorbs. "L" = Rec.709 luminance.
# LC/LM absorb the same band as their full-strength siblings (light cyan in red,
# light magenta in green); the nozzle check tells them apart spatially (each
# ink has its own comb), not by colour, so they share the scan channel.
SERIES_SCAN_CHANNEL = {"C": "R", "M": "G", "Y": "B", "K": "L", "NEUTRAL": "L",
                       "LC": "R", "LM": "G"}
# Ink series that get a per-channel LUT (NEUTRAL is kept for gray balance, R2).
LUT_SERIES = ("C", "M", "Y", "K")


# ---------------------------------------------------------------------------
# Payload / layout
# ---------------------------------------------------------------------------

def load_target_layout(payload_path: str) -> Tuple[int, Dict]:
    """Return ``(dpi, layout)`` from a RIP payload's ``metadata.target``."""
    header = json.loads(Path(payload_path).read_text())
    meta = header.get("metadata", {})
    layout = meta.get("target")
    if not layout:
        raise ValueError(f"{payload_path} has no metadata.target (not a wedge payload)")
    return int(layout.get("dpi", meta.get("dpi"))), layout


# ---------------------------------------------------------------------------
# Scan loading and fiducial registration
# ---------------------------------------------------------------------------

def load_scan(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a scan as (rgb float32 HxWx3, luminance float32 HxW), 0..255."""
    img = Image.open(path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:  # grayscale
        rgb = np.dstack([arr, arr, arr])
        print("   NOTE: grayscale scan; C/M/Y selectivity reduced, tone still valid")
    else:
        rgb = arr[..., :3]
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return rgb, lum


def _corner_order(pts: np.ndarray) -> List[int]:
    """Indices of ``pts`` (x, y) picking TL, TR, BR, BL by geometric corner:
    TL = min(x+y), TR = max(x-y), BR = max(x+y), BL = min(x-y)."""
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    return [int(np.argmin(s)), int(np.argmax(d)), int(np.argmax(s)), int(np.argmin(d))]


def find_fiducials(lum: np.ndarray, layout: Dict) -> np.ndarray:
    """Locate the four corner fiducials in the scan, returned as (x, y)
    centroids ordered TL, TR, BR, BL by geometric corner.

    Each fiducial is the dark blob most extreme toward one scan corner (they
    sit in the corner margins, outside the patch grid); we take that blob's
    centroid via a fiducial-sized window (no connected-component labelling, so
    no scipy). Matching is by *position*, which assumes the sheet is scanned
    roughly upright (small skew is fine): the print's deterministic vertical
    flip is already baked into metadata.target (printed space), so an upright
    scan lines up. ``layout_fiducial_centres`` uses the same corner order, so
    the two correspond regardless of the fiducials' sizes.
    """
    H, W = lum.shape
    paper = float(np.percentile(lum, 99))
    dark = lum < (0.45 * paper)
    ys, xs = np.where(dark)
    if len(xs) == 0:
        raise ValueError("no dark regions found in scan (check exposure/threshold)")

    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    scale = 0.5 * (W / layout["image_width_px"] + H / layout["image_height_px"])
    max_fp = max(f["x1"] - f["x0"] for f in layout["fiducials"])
    win = max(3, int(round(max_fp * scale)))

    def centroid_near(px: int, py: int) -> Tuple[float, float]:
        x0, x1 = max(0, px - win), min(W, px + win + 1)
        y0, y1 = max(0, py - win), min(H, py + win + 1)
        sub = dark[y0:y1, x0:x1]
        sy, sx = np.where(sub)
        return (x0 + sx.mean(), y0 + sy.mean())

    # Most extreme dark pixel toward each geometric corner, then its centroid.
    keys = [np.argmin(xs + ys), np.argmax(xs - ys), np.argmax(xs + ys), np.argmin(xs - ys)]
    return np.array([centroid_near(int(xs[i]), int(ys[i])) for i in keys], dtype=np.float64)


def layout_fiducial_centres(layout: Dict) -> np.ndarray:
    """Fiducial centres from the layout, ordered TL, TR, BR, BL by geometric
    corner (matching find_fiducials)."""
    cs = np.array(
        [((f["x0"] + f["x1"]) / 2.0, (f["y0"] + f["y1"]) / 2.0) for f in layout["fiducials"]],
        dtype=np.float64,
    )
    return cs[_corner_order(cs)]


def solve_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares affine mapping src (x,y) -> dst (x,y). Returns 2x3."""
    n = len(src)
    m = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    for i, (x, y) in enumerate(src):
        m[2 * i] = [x, y, 1, 0, 0, 0]
        m[2 * i + 1] = [0, 0, 0, x, y, 1]
        b[2 * i] = dst[i, 0]
        b[2 * i + 1] = dst[i, 1]
    params, *_ = np.linalg.lstsq(m, b, rcond=None)
    return params.reshape(2, 3)


def apply_affine(a: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map (N,2) points through a 2x3 affine."""
    homo = np.hstack([pts, np.ones((len(pts), 1))])
    return homo @ a.T


# ---------------------------------------------------------------------------
# Patch sampling
# ---------------------------------------------------------------------------

def _robust_tone(region: np.ndarray) -> float:
    """Robust patch tone: the median of the per-scan-row means.

    A dead or weak nozzle prints a bright streak along one raster row of the
    patch (nozzle index runs along Y). Averaging every pixel would let those
    streaks bias the tone lighter: exactly what corrupts a wedge printed
    with clogged nozzles. Taking each row's mean (integrating the halftone
    along X) and then the *median* over rows ignores a minority of bright
    streak rows while staying unbiased on a clean patch (row means are
    symmetric around the true tone). Rows still resolve the tone, so unlike a
    median of pixels this does not collapse light patches to paper white.
    """
    if region.size == 0:
        return 0.0
    row_means = region.mean(axis=1)
    return float(np.median(row_means))


def sample_patch(rgb: np.ndarray, lum: np.ndarray, affine: np.ndarray,
                 patch: Dict, inset: float) -> Dict[str, float]:
    """Robust scanner response over the inner region of one patch, mapped to
    the scan through ``affine``. Returns {'R','G','B','L'} (see _robust_tone)."""
    w = patch["x1"] - patch["x0"]
    h = patch["y1"] - patch["y0"]
    ix0, iy0 = patch["x0"] + inset * w, patch["y0"] + inset * h
    ix1, iy1 = patch["x1"] - inset * w, patch["y1"] - inset * h
    corners = np.array([[ix0, iy0], [ix1, iy0], [ix1, iy1], [ix0, iy1]])
    mapped = apply_affine(affine, corners)

    H, W = lum.shape
    x0 = max(0, int(np.floor(mapped[:, 0].min())))
    x1 = min(W, int(np.ceil(mapped[:, 0].max())))
    y0 = max(0, int(np.floor(mapped[:, 1].min())))
    y1 = min(H, int(np.ceil(mapped[:, 1].max())))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("a patch mapped outside the scan (bad registration?)")

    region_rgb = rgb[y0:y1, x0:x1]
    region_lum = lum[y0:y1, x0:x1]
    return {
        "R": _robust_tone(region_rgb[..., 0]),
        "G": _robust_tone(region_rgb[..., 1]),
        "B": _robust_tone(region_rgb[..., 2]),
        "L": _robust_tone(region_lum),
    }


# ---------------------------------------------------------------------------
# Tone + LUT
# ---------------------------------------------------------------------------

def reflectance_to_lstar(r: np.ndarray) -> np.ndarray:
    """CIE L* from relative reflectance (Y/Yn), vectorised, 0..100."""
    r = np.clip(r, 0.0, 1.0)
    return np.where(r > 0.008856, 116.0 * np.cbrt(r) - 16.0, 903.3 * r)


def series_lstar(coverages: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(sorted_coverage, lstar)`` for a series: L* vs commanded
    coverage, with the 0 %-coverage patch as paper white."""
    order = np.argsort(coverages)
    c = coverages[order]
    v = values[order]
    paper = max(v[0], 1e-6)
    refl = np.clip(v / paper, 0.0, 1.0)
    return c, reflectance_to_lstar(refl)


def build_series_lut(coverages: np.ndarray, values: np.ndarray,
                     l_dark: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Invert a commanded-coverage -> measured-tone curve into a 256-entry LUT.

    ``values`` is the raw scanner response per patch (higher = lighter). The
    0 %-coverage patch defines paper white. ``l_dark`` overrides the dark
    anchor: pass a **common** value across C/M/Y for gray balance (each channel
    then reaches the same tone at input 255, so neutrals print neutral).
    Returns ``(lut, darkness)``.
    """
    c, lstar = series_lstar(coverages, values)

    # Anchor the tonal range on the 0 % patch (paper white) and the *darkest*
    # measured patch, not the last one: with clogs/noise the 100 % patch isn't
    # always the darkest, and dividing by a collapsed span would blow the
    # curve up. min() keeps the span the true measured range.
    l_white = float(lstar[0])
    if l_dark is None:
        l_dark = float(lstar.min())
    span = max(l_white - l_dark, 1e-6)
    darkness = np.clip((l_white - lstar) / span, 0.0, 1.0)   # 0 at paper, 1 at (common) dark
    darkness = np.maximum.accumulate(darkness)   # enforce monotonic non-decreasing
    # Make strictly increasing so np.interp is well-defined.
    darkness = darkness + np.linspace(0, 1e-6, len(darkness))

    t = np.linspace(0.0, 1.0, 256)               # desired darkness per input value
    coverage = np.interp(t, darkness, c)         # invert: darkness -> coverage
    lut = np.clip(np.rint(coverage * 255.0), 0, 255).astype(int)
    return lut, darkness


def degenerate_channels(luts: Dict[str, List[int]],
                        min_span: int = 40, max_low: int = 64) -> List[Tuple[str, str]]:
    """Flag channels whose measured curve looks degenerate (near-bilevel).

    Two symptoms of a corrupted measurement (typically clogged nozzles or a
    flooding/saturating substrate) show up in the LUT:

      * ``LUT[1]`` large: the faintest tone already needs a lot of coverage,
        i.e. the highlights printed white (a low-end dead zone);
      * ``LUT[255] - LUT[1]`` tiny: the whole tonal range is crammed into a
        narrow coverage window (a cliff).

    Returns ``[(channel, reason), ...]`` so the caller can warn before a
    broken LUT is shipped.
    """
    flags: List[Tuple[str, str]] = []
    for channel, lut in luts.items():
        arr = np.asarray(lut)
        span = int(arr[255] - arr[1])
        low = int(arr[1])
        reasons = []
        if span < min_span:
            reasons.append(f"tonal range crammed into {span}/255 coverage")
        if low > max_low:
            reasons.append(f"highlights dead until {low}/255 coverage")
        if reasons:
            flags.append((channel, "; ".join(reasons)))
    return flags


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def calibrate(scan_path: str, payload_path: str, inset: float = 0.3,
              gray_balance: bool = False) -> Tuple[int, Dict, Dict]:
    """Run the full extraction. Returns ``(dpi, luts, report)``.

    With ``gray_balance`` the C/M/Y LUTs share a common dark anchor (the
    weakest ink's maximum), so equal input drives equal tone and a neutral
    prints neutral, at the cost of some saturation in the strong inks (R2)."""
    dpi, layout = load_target_layout(payload_path)
    rgb, lum = load_scan(scan_path)

    scan_fids = find_fiducials(lum, layout)
    layout_fids = layout_fiducial_centres(layout)
    affine = solve_affine(layout_fids, scan_fids)

    # Sample every patch, grouped by series.
    by_series: Dict[str, List[Tuple[float, float]]] = {}
    for patch in layout["patches"]:
        series = patch["series"]
        chan = SERIES_SCAN_CHANNEL[series]
        resp = sample_patch(rgb, lum, affine, patch, inset)
        by_series.setdefault(series, []).append((patch["coverage"], resp[chan]))

    arrays = {}
    for series, pts in by_series.items():
        pts.sort()
        arrays[series] = (np.array([c for c, _ in pts]), np.array([v for _, v in pts]))

    # Gray balance: anchor C/M/Y on a common dark end (the least-dark maximum,
    # i.e. the weakest ink), so equal input -> equal tone -> neutral neutrals.
    common_l_dark = None
    if gray_balance:
        maxima = [float(series_lstar(*arrays[s])[1].min()) for s in ("C", "M", "Y") if s in arrays]
        if maxima:
            common_l_dark = max(maxima)

    luts: Dict[str, List[int]] = {}
    report: Dict[str, Dict] = {}
    for series, (cov, val) in arrays.items():
        ldark = common_l_dark if (gray_balance and series in ("C", "M", "Y")) else None
        lut, darkness = build_series_lut(cov, val, l_dark=ldark)
        if series in LUT_SERIES:
            luts[series] = lut.tolist()
        # Linearity metric: how far the measured tone is from a linear ramp in
        # commanded coverage. High on a raw wedge, low once the LUT is applied
        # and re-printed: a single number to track across bench iterations.
        cov_norm = cov / max(cov.max(), 1e-6)
        linearity_rmse = float(np.sqrt(np.mean((darkness - cov_norm) ** 2)))
        report[series] = {
            "coverages": cov.tolist(),
            "measured": val.tolist(),
            "darkness": darkness.tolist(),
            "linearity_rmse": linearity_rmse,
        }
    return dpi, luts, report


def write_overlay(scan_path: str, payload_path: str, out_path: str, inset: float = 0.3) -> None:
    """Draw detected fiducials and sampled patch windows onto the scan (QA)."""
    from PIL import ImageDraw

    dpi, layout = load_target_layout(payload_path)
    rgb, lum = load_scan(scan_path)
    scan_fids = find_fiducials(lum, layout)
    affine = solve_affine(layout_fiducial_centres(layout), scan_fids)

    img = Image.open(scan_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for (fx, fy) in scan_fids:
        draw.ellipse([fx - 6, fy - 6, fx + 6, fy + 6], outline=(255, 0, 0), width=2)
    for patch in layout["patches"]:
        w, h = patch["x1"] - patch["x0"], patch["y1"] - patch["y0"]
        corners = np.array([
            [patch["x0"] + inset * w, patch["y0"] + inset * h],
            [patch["x1"] - inset * w, patch["y0"] + inset * h],
            [patch["x1"] - inset * w, patch["y1"] - inset * h],
            [patch["x0"] + inset * w, patch["y1"] - inset * h],
        ])
        m = apply_affine(affine, corners)
        draw.polygon([tuple(p) for p in m], outline=(0, 255, 0))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def update_profile(profile_path: str, out_path: str, dpi: int, luts: Dict[str, List[int]]) -> None:
    """Load a calibration profile, set dpi_curves[<dpi>], write it back."""
    data = json.loads(Path(profile_path).read_text())
    data.setdefault("dpi_curves", {})
    data["dpi_curves"][str(dpi)] = luts
    Path(out_path).write_text(json.dumps(data, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build per-channel linearisation LUTs from a wedge scan.")
    parser.add_argument("scan", help="Scanned image of the printed wedge (PNG/TIFF).")
    parser.add_argument("--payload", required=True,
                        help="RIP payload (.json) that produced the wedge (carries the layout).")
    parser.add_argument("--profile", required=True,
                        help="Calibration profile JSON to update with the measured LUTs.")
    parser.add_argument("--out", default=None,
                        help="Where to write the updated profile. Default: overwrite --profile.")
    parser.add_argument("--inset", type=float, default=0.3,
                        help="Fraction of each patch trimmed on every side before averaging. Default: 0.3")
    parser.add_argument("--preview", default=None,
                        help="Write a QA overlay (fiducials + sample windows) to this PNG.")
    parser.add_argument("--force", action="store_true",
                        help="Write the LUTs even if the measured curve looks degenerate.")
    parser.add_argument("--gray-balance", action="store_true",
                        help="Anchor C/M/Y on a common dark end so neutrals print neutral (R2), "
                             "trading a little saturation for a cast-free grey.")
    args = parser.parse_args()

    print(f"Calibrating from scan: {args.scan}"
          + (" (gray-balanced)" if args.gray_balance else ""))
    try:
        dpi, luts, report = calibrate(args.scan, args.payload, inset=args.inset,
                                      gray_balance=args.gray_balance)
    except (OSError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    print(f"   DPI: {dpi}")
    for series in ("C", "M", "Y", "K", "NEUTRAL"):
        if series in report:
            dk = report[series]["darkness"]
            print(f"   {series:8s} darkness 0%->100%: {dk[0]:.2f} .. {dk[-1]:.2f}  "
                  f"(linearity RMSE {report[series]['linearity_rmse']:.3f})")
    if "NEUTRAL" in report:
        print(f"   Neutral-ramp linearity RMSE: {report['NEUTRAL']['linearity_rmse']:.3f} "
              f"(lower = more linear; track this across bench iterations)")

    # Write the QA overlay first: it is a diagnostic, and most useful exactly
    # when the guard below refuses (to see whether registration was the cause).
    if args.preview:
        write_overlay(args.scan, args.payload, args.preview, inset=args.inset)
        print(f"   QA overlay: {args.preview}")

    flags = degenerate_channels(luts)
    if flags:
        print("   WARNING: the measured curve looks degenerate on:")
        for channel, reason in flags:
            print(f"      {channel}: {reason}")
        print("   This usually means clogged nozzles or a flooding substrate.")
        print("   Purge/clean the head, run --target nozzle_check to confirm it is")
        print("   healthy, then re-print and re-scan the wedge.")
        if not args.force:
            print("   Refusing to write these LUTs (pass --force to override).")
            return 2

    out = args.out or args.profile
    update_profile(args.profile, out, dpi, luts)
    print(f"   Wrote dpi_curves[{dpi}] ({', '.join(luts)}) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
