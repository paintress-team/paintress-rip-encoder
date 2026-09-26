# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Print job viewer / debugging tool.

Reconstructs a human-viewable image from a RIP payload produced by
the RIP. It reverses the pass interleaving to rebuild each ink channel
(C, M, Y, K, LC, LM) as a full-resolution bitmap, and can optionally
merge them into a combined CMYK image plus an RGB preview.

This tool is for inspection only; it is not part of the print pipeline.
"""

import math
import numpy as np
from PIL import Image, ImageFilter
import head_layout
import rip_payload
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Soft-proof + ink estimation (R9)
# ---------------------------------------------------------------------------

def drop_volume_pl(drop_diameter_um: float) -> float:
    """Approximate drop volume (picolitres) from its diameter, as a sphere
    (1 pL = 1000 um^3). A rough estimate; override with a measured value."""
    return (math.pi / 6.0) * (drop_diameter_um ** 3) / 1000.0


def estimate_ink(channel_bitmaps: Dict[str, np.ndarray],
                 drop_pl: float) -> Dict[str, Tuple[int, float]]:
    """Per-channel (dot count, volume in microlitres) from the fired dots."""
    out: Dict[str, Tuple[int, float]] = {}
    for name, bmp in channel_bitmaps.items():
        dots = int(bmp.sum())
        out[name] = (dots, dots * drop_pl / 1e6)   # pL -> uL
    return out


def render_soft_proof(channel_bitmaps: Dict[str, np.ndarray], dpi: int,
                      drop_diameter_um: float, max_dim_px: int = 1600) -> Image.Image:
    """Simulate the printed sheet: spread each channel's dots by the drop size,
    let the eye integrate them (area downscale), and composite CMYK -> RGB.

    Shows tone, graininess, dead-nozzle streaks and band seams before a rig
    trip. Naive subtractive colour on a light-grey paper: a preview, not a
    colorimetric proof.
    """
    sample = next(iter(channel_bitmaps.values()))
    h, w = sample.shape
    blur = max(0.6, drop_diameter_um * dpi / 25400.0)   # drop footprint in px

    cov: Dict[str, np.ndarray] = {}
    for name in ("C", "M", "Y", "K"):
        if name in channel_bitmaps:
            img = Image.fromarray((channel_bitmaps[name] * 255).astype(np.uint8), "L")
            img = img.filter(ImageFilter.GaussianBlur(blur))
            cov[name] = np.asarray(img, dtype=np.float32) / 255.0
        else:
            cov[name] = np.zeros((h, w), dtype=np.float32)
    for light, base in (("LC", "C"), ("LM", "M")):     # blend light inks at ~50%
        if light in channel_bitmaps:
            img = Image.fromarray((channel_bitmaps[light] * 255).astype(np.uint8), "L")
            img = img.filter(ImageFilter.GaussianBlur(blur))
            cov[base] = np.clip(cov[base] + 0.5 * np.asarray(img, np.float32) / 255.0, 0, 1)

    c, m, y, k = cov["C"], cov["M"], cov["Y"], cov["K"]
    r = (255.0 * (1 - c) * (1 - k))
    g = (255.0 * (1 - m) * (1 - k))
    b = (255.0 * (1 - y) * (1 - k))
    rgb = np.clip(np.dstack([r, g, b]), 0, 255).astype(np.uint8)
    proof = Image.fromarray(rgb, "RGB")

    scale = min(1.0, max_dim_px / max(w, h))
    if scale < 1.0:                                    # box downscale ~ eye integrating dots
        proof = proof.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BOX)
    return proof


def load_print_job(input_path: str) -> Dict:
    """Load a RIP payload (JSON header + .bin sidecar) produced by rip.py."""
    return rip_payload.load_rip(input_path)


def channel_rows(metadata: Dict, channel_name: str) -> Optional[Tuple[int, int, int]]:
    """Where ``channel_name``'s nozzle rows land, from the payload header.

    Returns ``(nozzle_pitch_lines, first_nozzle, nozzle_count)``, or None for a
    plane no slot feeds (a blank light ink). The pitch comes from the head's
    nozzle pitch, not from ``dpi // nozzle_count``: the two only coincide on
    c6n90 (90 nozzles at 90 npi). A payload without a ``head_layout`` block is
    a c6n90 one (rip_payload format 0.1.x).
    """
    dpi = metadata["dpi"]
    layout = metadata.get("head_layout")
    if layout is None:
        pitch = head_layout.get_layout("c6n90").nozzle_pitch_npi
        return dpi // pitch, 0, metadata["nozzle_count"]
    for slot in layout["slots"]:
        if slot["ink"] == channel_name:
            return (dpi // layout["nozzle_pitch_npi"],
                    slot["first_nozzle"], slot["nozzle_count"])
    return None


def reconstruct_channel_from_passes(
    passes_data: List[np.ndarray],
    y_positions_mm: List[float],
    channel_index: int,
    dpi: int,
    nozzle_pitch_lines: int,
    nozzle_count: int,
    first_nozzle: int = 0,
    image_height_px: int = None,
) -> np.ndarray:
    """Reconstruct a single channel bitmap from print passes.

    Inverts the RIP's slicing: row ``n`` of a channel, with the bottom of the
    column at raster line ``head_line``, printed line
    ``head_line + (first_nozzle + n) * nozzle_pitch_lines``.

    Args:
        passes_data: List of arrays shaped (channels, nozzles, width).
        y_positions_mm: Absolute Y position of each pass in mm (the bottom of
            the column; negative during a lead-in).
        channel_index: Payload plane to extract.
        dpi: Print resolution.
        nozzle_pitch_lines: Raster lines between adjacent nozzles
            (dpi / nozzle pitch); see channel_rows().
        nozzle_count: Rows the channel's slot fires.
        first_nozzle: How far up its column the slot starts.
        image_height_px: Exact image height from the payload metadata.
            Without it the height is derived from the highest line any pass
            reaches, which overshoots when the final band is partial.

    Returns:
        Reconstructed bitmap of shape (height, width) with values 0/1.
    """
    if not passes_data:
        raise ValueError("Pass list is empty")

    line_spacing_mm = 25.4 / dpi
    image_width = passes_data[0].shape[2]
    nozzles = np.arange(nozzle_count)

    # Convert Y positions from mm to line indices
    head_lines = [
        int(round(y_mm / line_spacing_mm))
        for y_mm in y_positions_mm
    ]

    if image_height_px is not None:
        image_height = image_height_px
    else:
        top = (first_nozzle + nozzle_count - 1) * nozzle_pitch_lines
        image_height = max(1, max(head_lines) + top + 1)

    # Reconstruct the full-resolution bitmap
    reconstructed = np.zeros((image_height, image_width), dtype=np.uint8)

    for pass_data, head_line in zip(passes_data, head_lines):
        lines = head_line + (first_nozzle + nozzles) * nozzle_pitch_lines
        valid = (lines >= 0) & (lines < image_height)
        if not valid.any():
            continue
        # Use max so overlapping passes do not erase existing dots
        reconstructed[lines[valid], :] = np.maximum(
            reconstructed[lines[valid], :],
            pass_data[channel_index, nozzles[valid], :],
        )

    # Pass data is in machine order: line 0 is the lowest Y, i.e. the
    # BOTTOM of the image (the RIP slices bottom-up to match the machine's
    # bottom-left origin). Flip back to image orientation for viewing.
    return reconstructed[::-1]


def save_channel_image(
    bitmap: np.ndarray,
    output_path: str,
    invert: bool = True,
) -> None:
    """Save a 0/1 bitmap as a greyscale image.

    When invert is True the output uses 1 -> black, 0 -> white so that
    ink dots appear dark on a white background.
    """
    if invert:
        image_data = ((1 - bitmap) * 255).astype(np.uint8)
    else:
        image_data = (bitmap * 255).astype(np.uint8)

    image = Image.fromarray(image_data, mode="L")
    image.save(output_path)


def combine_cmyk_channels(
    channel_bitmaps: Dict[str, np.ndarray],
) -> Image.Image:
    """Merge individual channel bitmaps into a single CMYK image.

    If LC/LM channels are present they are blended into C/M at reduced
    intensity.
    """
    sample = next(iter(channel_bitmaps.values()))
    height, width = sample.shape

    cyan = np.zeros((height, width), dtype=np.uint8)
    magenta = np.zeros((height, width), dtype=np.uint8)
    yellow = np.zeros((height, width), dtype=np.uint8)
    black = np.zeros((height, width), dtype=np.uint8)

    # Cyan channel, with optional light-cyan contribution at ~50%
    if "C" in channel_bitmaps:
        cyan = channel_bitmaps["C"] * 255
    if "LC" in channel_bitmaps:
        cyan = np.clip(
            cyan.astype(np.int16) + channel_bitmaps["LC"] * 128, 0, 255
        ).astype(np.uint8)

    # Magenta channel, with optional light-magenta contribution at ~50%
    if "M" in channel_bitmaps:
        magenta = channel_bitmaps["M"] * 255
    if "LM" in channel_bitmaps:
        magenta = np.clip(
            magenta.astype(np.int16) + channel_bitmaps["LM"] * 128, 0, 255
        ).astype(np.uint8)

    if "Y" in channel_bitmaps:
        yellow = channel_bitmaps["Y"] * 255
    if "K" in channel_bitmaps:
        black = channel_bitmaps["K"] * 255

    cmyk_image = Image.merge("CMYK", [
        Image.fromarray(cyan, mode="L"),
        Image.fromarray(magenta, mode="L"),
        Image.fromarray(yellow, mode="L"),
        Image.fromarray(black, mode="L"),
    ])

    return cmyk_image


def save_combined_cmyk(
    channel_bitmaps: Dict[str, np.ndarray],
    output_path: str,
    also_save_rgb: bool = True,
) -> None:
    """Save a combined CMYK image (as TIFF) and optionally an RGB preview."""
    cmyk_image = combine_cmyk_channels(channel_bitmaps)

    cmyk_path = Path(output_path)
    if cmyk_path.suffix.lower() not in [".tiff", ".tif"]:
        cmyk_tiff_path = cmyk_path.with_suffix(".tiff")
        cmyk_image.save(cmyk_tiff_path)
        print(f"   CMYK saved as TIFF: {cmyk_tiff_path}")
    else:
        cmyk_image.save(output_path)

    if also_save_rgb:
        rgb_image = cmyk_image.convert("RGB")
        rgb_path = cmyk_path.with_name(f"{cmyk_path.stem}_rgb.png")
        rgb_image.save(rgb_path)
        print(f"   RGB preview: {rgb_path}")


def extract_all_channels(
    input_path: str,
    output_dir: str,
    output_format: str = "png",
    invert: bool = True,
    generate_combined: bool = True,
    soft_proof_path: Optional[str] = None,
    drop_diameter_um: Optional[float] = None,
    drop_volume_pl_override: Optional[float] = None,
) -> None:
    """Extract every channel from a print job and save as individual images.

    Args:
        input_path: Path to the RIP payload JSON header.
        output_dir: Directory for the output images.
        output_format: Image format (png, tiff, bmp).
        invert: Invert colours so ink dots appear dark.
        generate_combined: Also produce a merged CMYK/RGB image.
    """
    import time

    print(f"Loading: {input_path}")
    t0 = time.perf_counter()
    print_job = load_print_job(input_path)
    t1 = time.perf_counter()
    print(f"Loaded ({t1-t0:.3f}s)")

    metadata = print_job["metadata"]
    passes = print_job["passes"]

    dpi = metadata["dpi"]
    channel_order = metadata["channel_order"]
    total_passes = metadata["total_passes"]

    print(f"Metadata:")
    print(f"   DPI: {dpi}")
    print(f"   Dimensions: {metadata['image_width_px']}x{metadata['image_height_px']} px")
    print(f"   Passes: {total_passes}")
    print(f"   Channels: {channel_order}")

    passes_data = passes["data"]
    y_positions_mm = passes["y_positions_mm"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    channel_bitmaps: Dict[str, np.ndarray] = {}
    input_name = Path(input_path).stem

    print(f"\nExtracting individual channels...")
    for channel_index, channel_name in enumerate(channel_order):
        print(f"   [{channel_name}]", end=" ", flush=True)
        t0 = time.perf_counter()

        rows = channel_rows(metadata, channel_name)
        if rows is None:
            # No slot feeds this plane: it is blank by construction.
            channel_bitmap = np.zeros(
                (metadata["image_height_px"], passes_data[0].shape[2]),
                dtype=np.uint8)
        else:
            pitch_lines, first_nozzle, slot_nozzles = rows
            channel_bitmap = reconstruct_channel_from_passes(
                passes_data=passes_data,
                y_positions_mm=y_positions_mm,
                channel_index=channel_index,
                dpi=dpi,
                nozzle_pitch_lines=pitch_lines,
                nozzle_count=slot_nozzles,
                first_nozzle=first_nozzle,
                image_height_px=metadata["image_height_px"],
            )

        channel_bitmaps[channel_name] = channel_bitmap

        output_filename = f"{input_name}_{channel_name}.{output_format}"
        output_filepath = output_path / output_filename

        save_channel_image(
            bitmap=channel_bitmap,
            output_path=str(output_filepath),
            invert=invert,
        )

        t1 = time.perf_counter()
        print(f"OK {output_filepath.name} ({t1-t0:.3f}s)")

    if generate_combined:
        print(f"\nGenerating combined image...")
        t0 = time.perf_counter()

        combined_path = output_path / f"{input_name}_combined.tiff"
        save_combined_cmyk(
            channel_bitmaps=channel_bitmaps,
            output_path=str(combined_path),
            also_save_rgb=True,
        )

        t1 = time.perf_counter()
        print(f"   Done ({t1-t0:.3f}s)")

    # Ink estimation + soft-proof (R9). Prefer the drop diameter baked into the
    # payload's calibration profile; fall back to the CLI value / a default.
    profile_drop = None
    try:
        profile_drop = metadata["processing"]["calibration"].get("drop_diameter_um")
    except (KeyError, AttributeError, TypeError):
        pass
    drop_d = drop_diameter_um or profile_drop or 40.0
    drop_pl = drop_volume_pl_override if drop_volume_pl_override is not None else drop_volume_pl(drop_d)

    print(f"\nEstimated ink (drop {drop_d:.0f} um ~ {drop_pl:.1f} pL):")
    ink = estimate_ink(channel_bitmaps, drop_pl)
    total_uL = 0.0
    for name in channel_order:
        if name in ink:
            dots, uL = ink[name]
            total_uL += uL
            print(f"   {name:3s} {dots:>12,d} dots  ~{uL:8.3f} uL")
    print(f"   total ~{total_uL:.3f} uL")

    if soft_proof_path:
        print(f"\nRendering soft-proof...")
        proof = render_soft_proof(channel_bitmaps, dpi, drop_d)
        Path(soft_proof_path).parent.mkdir(parents=True, exist_ok=True)
        proof.save(soft_proof_path)
        print(f"   Soft-proof: {soft_proof_path} ({proof.width}x{proof.height} px)")

    print(f"\nComplete! Files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract channels from a print job and save as images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python viewer.py print_job.json -o ./channels/
  python viewer.py print_job.json -o ./channels/ --format tiff
  python viewer.py print_job.json -o ./channels/ --no-invert
  python viewer.py print_job.json -o ./channels/ --no-combined
        """
    )

    parser.add_argument("input", help="Input RIP payload JSON header produced by rip.py")
    parser.add_argument(
        "-o", "--output-dir",
        default="./output_channels",
        help="Output directory (default: ./output_channels)"
    )
    parser.add_argument(
        "--format",
        choices=["png", "tiff", "bmp"],
        default="png",
        help="Image format for individual channels (default: png)"
    )
    parser.add_argument(
        "--no-invert",
        action="store_true",
        help="Do not invert colours (default: invert so 1=black, 0=white)"
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Do not generate the combined CMYK image"
    )
    parser.add_argument(
        "--soft-proof",
        default=None,
        help="Render a simulated-print preview PNG to this path (R9): dots spread "
             "by the drop size and integrated, so it shows tone, graininess, dead "
             "nozzles and band seams before a rig trip."
    )
    parser.add_argument(
        "--drop-diameter-um",
        type=float,
        default=None,
        help="Drop diameter (um) for the soft-proof and ink estimate. Default: the "
             "payload's calibration drop_diameter_um, else 40."
    )
    parser.add_argument(
        "--drop-volume-pl",
        type=float,
        default=None,
        help="Measured drop volume (pL) for the ink estimate; default is a sphere "
             "estimate from the diameter."
    )

    args = parser.parse_args()

    extract_all_channels(
        input_path=args.input,
        output_dir=args.output_dir,
        output_format=args.format,
        invert=not args.no_invert,
        generate_combined=not args.no_combined,
        soft_proof_path=args.soft_proof,
        drop_diameter_um=args.drop_diameter_um,
        drop_volume_pl_override=args.drop_volume_pl,
    )


if __name__ == "__main__":
    main()