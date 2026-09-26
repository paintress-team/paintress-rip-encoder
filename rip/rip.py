# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Raster Image Processor (RIP) for piezo inkjet printing.

First stage of the Paintress pipeline: turns a regular raster image into
a halftoned, pass-organised print job that the encoder later converts
into the firmware wire format.

Stages performed here:
    1. Load the image and convert it to CMYK (UCR/GCR fallback when no
       ICC profile is available, so neutral shadows use the K channel
       instead of stacking three inks).
    2. Apply dot gain compensation (pre-lightening that counters the way
       ink dots spread on the substrate).
    3. Apply practical ink limits / colour balance (per-channel scaling,
       total-ink cap, DPI ink compensation).
    4. Halftone each channel into a 1-bit bitmap (dithering).
    5. Slice the bitmaps into print passes, following the interleaved
       nozzle layout of the printhead. The slicing runs bottom-up so the
       print lands the right way up on a machine whose origin is the
       bottom-left corner (pass 0 = lowest Y = bottom of the image).
    6. Append the light-ink channels (LC/LM).
    7. Compute the absolute Y position of every pass.
    8. Serialise the result to a JSON header + packed .bin sidecar.

Calibration: colour behaviour is driven by a calibration profile
(``--calibration profile.json``). The profile is the source of truth;
individual CLI flags (``--cyan-scale`` etc.) act as pointwise overrides.
See ``rip/profiles/default.json`` for the schema and the seed values.

DPI: the CLI accepts any multiple of the head's nozzle pitch (90 on c6n90,
180 on c4n180); --list-dpi shows them for the selected head.
A higher DPI simply uses more interleaved passes per band.

Dithering methods: Floyd-Steinberg, blue noise, ordered.
Colour: optional ICC profile support for RGB -> CMYK conversion.
"""

from typing import List, Dict, Tuple, Optional, Sequence
from dataclasses import dataclass, field
import numpy as np
from PIL import Image
import math
import json
import rip_payload
import head_layout
from head_layout import (  # noqa: F401  (re-exported: tools import them from rip)
    DEFAULT_HEAD,
    EMPTY_SLOT,
    HEAD_LAYOUTS,
    HeadGeometry,
    get_layout,
)
import argparse
from pathlib import Path
import io

# Attempt to import numba for JIT acceleration
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: Numba not found. Using pure-Python implementation (slower).")
    print("   Install with: pip install numba")

# Attempt to import PIL.ImageCms for ICC profile support
try:
    from PIL import ImageCms
    HAS_ICC = True
except ImportError:
    HAS_ICC = False
    print("Warning: ImageCms not available. ICC conversion disabled.")


# Shipped seed profile (versioned example + built-in fallback).
DEFAULT_PROFILE_PATH = Path(__file__).resolve().parent / "profiles" / "default.json"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PrintheadConfig:
    """Printhead configuration: which head, at which DPI, plumbed how.

    A thin front for ``head_layout.HeadGeometry``: the head's geometry and
    its plumbing (``ink_map``) resolved together, since on a head whose slots
    sit at different heights the plumbing decides which rows an ink reaches.
    DPI must be a multiple of the head's nozzle pitch.

    The ``nozzle_count`` / ``passes_per_band`` / ``lines_per_band`` properties
    are the single-slot names the rest of the RIP grew up with; they keep
    meaning exactly what they used to on ``c6n90``.
    """

    dpi: int = 630
    head: str = DEFAULT_HEAD
    ink_map: Optional[Tuple[str, ...]] = None

    def __post_init__(self):
        self.geometry = HeadGeometry.build(
            get_layout(self.head), self.dpi, self.ink_map)
        # Normalise to the plumbing actually in force (the layout's reference
        # wiring when none was given), so callers can read it back.
        self.ink_map = self.geometry.ink_map

    @property
    def layout(self) -> "head_layout.HeadLayout":
        return self.geometry.layout

    @property
    def channel_order(self) -> Tuple[str, ...]:
        return self.geometry.layout.channel_order

    @property
    def nozzle_count(self) -> int:
        """Nozzles per channel plane in the payload (the largest plumbed slot)."""
        return self.geometry.max_group_nozzles

    @property
    def passes_per_band(self) -> int:
        return self.geometry.nozzle_pitch_lines

    @property
    def lines_per_band(self) -> int:
        return self.geometry.band_step_lines

    @property
    def line_spacing_mm(self) -> float:
        return self.geometry.line_spacing_mm


@dataclass
class DotGainConfig:
    """Dot-gain compensation settings.

    Each gain value lies in [0.0, 1.0]. Higher values produce more
    compensation (lighter output).
    """

    enabled: bool = True
    cyan_gain: float = 0.50
    magenta_gain: float = 0.50
    yellow_gain: float = 0.375
    black_gain: float = 0.625

    def get_gain_for_channel(self, channel: str) -> float:
        """Return the gain factor for the given channel name."""
        gains = {
            "C":  self.cyan_gain,
            "M":  self.magenta_gain,
            "Y":  self.yellow_gain,
            "K":  self.black_gain,
            "LC": self.cyan_gain * 0.5,
            "LM": self.magenta_gain * 0.5,
        }
        return gains.get(channel, 0.20)


@dataclass
class InkLimitConfig:
    """Practical ink limiting before halftoning.

    This is intentionally simple and calibration-friendly. Channel scales fix
    per-ink strength / colour cast. max_total_ink limits CMYK overprint in a
    pixel. DPI compensation reduces coverage when the same droplet size is used
    at a higher raster DPI.

    Values are fractions, not 0..255.
    """

    enabled: bool = True
    channel_scales: Dict[str, float] = field(default_factory=lambda: {
        # Conservative starting point for an unprofiled CMYK head.
        # Green cast usually means too much C+Y or too little M.
        "C": 0.55,
        "M": 0.80,
        "Y": 0.45,
        "K": 0.55,
        "LC": 0.55,
        "LM": 0.80,
    })
    global_scale: float = 1.0
    max_total_ink: float = 0.90
    compensate_dpi: bool = True
    reference_dpi: int = 360
    dpi_power: float = 2.0


@dataclass
class DitherConfig:
    """Dithering configuration.

    Supported methods: "floyd_steinberg", "blue_noise", "ordered".
    """

    method: str = "floyd_steinberg"
    blue_noise_size: int = 64
    ordered_matrix_size: int = 8


# ---------------------------------------------------------------------------
# Calibration profile
# ---------------------------------------------------------------------------

# Ink map: which ink feeds each of the head's physical slots, in the order you
# read them facing the head (left to right by column, bottom to top within a
# column). This is machine-setup plumbing, stored in the calibration profile so
# that the RIP (slot geometry + dead-nozzle slot<->channel translation) and the
# encoder (slot routing) read one source. EMPTY_SLOT marks an unplumbed slot.
#
# On c6n90 every slot has the same geometry, so the plumbing only affects
# routing. On a head with slots at different heights (c4n180) it also decides
# which rows each ink can reach; see head_layout.py.
#
# The slot count is a property of the head layout; SLOT_COUNT is kept as the
# c6n90 value for callers that predate multiple layouts. A profile carrying no
# ink_map is read the legacy channel-keyed way (dead_nozzles keyed by ink).
SLOT_COUNT = head_layout.C6N90.slot_count


def dead_slots_to_channels(
    slot_dead: Dict, ink_map: Sequence[str]
) -> Dict[str, "List[int]"]:
    """Translate slot-keyed dead nozzles into channel-keyed, via the ink map.

    ``slot_dead`` is ``{slot_index: [nozzle indices]}`` (physical truth from the
    nozzle check). Each slot fires the ink the map plumbs into it, so slot s's
    dead nozzles become the dead nozzles of ink ``ink_map[s]``. Unplumbed slots
    (EMPTY_SLOT) carry no ink and drop out. The result is what the whole N2
    compensation machinery consumes: dead nozzles per channel.
    """
    channel_dead: Dict[str, List[int]] = {}
    for slot_key, indices in slot_dead.items():
        slot = int(slot_key)
        ink = ink_map[slot]
        if ink != EMPTY_SLOT:
            channel_dead[ink] = sorted(int(i) for i in indices)
    return channel_dead


def dead_channels_to_slots(
    channel_dead: Dict[str, "List[int]"], ink_map: Sequence[str]
) -> Dict[str, "List[int]"]:
    """Inverse of dead_slots_to_channels: channel-keyed back to slot-keyed.

    Used when writing the profile so a dead nozzle is stored against the
    physical slot it belongs to (plumbing-independent), not the ink that
    happened to feed it. An ink with no slot in the map is dropped.
    """
    slot_of_ink = {ink: slot for slot, ink in enumerate(ink_map)
                   if ink != EMPTY_SLOT}
    return {
        str(slot_of_ink[channel]): sorted(int(i) for i in indices)
        for channel, indices in channel_dead.items()
        if channel in slot_of_ink
    }


@dataclass
class CalibrationProfile:
    """All measured/tuned colour behaviour for one media, as one file.

    The profile is the single source of truth for colour: dot-gain
    compensation, ink limiting / channel balance, and black generation.
    CLI flags act only as pointwise overrides on a loaded profile.

    ``dpi_curves`` is reserved for R1 (measured per-channel x DPI
    linearisation LUTs) and is empty in the seed profile.
    """

    media: str = "default-unprofiled-cmyk"
    dot_gain: DotGainConfig = field(default_factory=DotGainConfig)
    ink_limit: InkLimitConfig = field(default_factory=InkLimitConfig)
    black_generation: float = 1.0
    # GCR start point (0..1): the neutral component below this stays CMY (no
    # K in the highlights, where black dots look grainy); above it K ramps in.
    black_start: float = 0.0
    dpi_curves: Dict[str, Dict] = field(default_factory=dict)
    # {channel: [nozzle indices]} that don't fire, from the nozzle check (N1).
    # Consumed by the N2 dead-nozzle compensation. Held channel-keyed in memory
    # (that is what compensation works in); on disk it is slot-keyed and
    # translated through ``ink_map`` at load/save so the data stays a physical
    # property of the head, valid across re-plumbing.
    dead_nozzles: Dict[str, "List[int]"] = field(default_factory=dict)
    # Head plumbing: the ink feeding each physical slot, left to right (see the
    # module-level ink-map note). None on a legacy profile that predates the
    # ink map; then dead_nozzles was stored channel-keyed and used verbatim.
    ink_map: Optional[Tuple[str, ...]] = None
    # The head this profile was measured on. dead_nozzles and ink_map are
    # physical facts about one head and do not carry over to another.
    head: Optional[str] = None
    # Estimated drop diameter (um) for the modelled Murray-Davies linearisation
    # used when a DPI has no measured LUT. One number per media; None disables it.
    drop_diameter_um: Optional[float] = None
    # Shingling for non-absorbent media (R8): print each swath in this many
    # sweeps, each firing every Nth column, so wet drops settle between
    # depositions. 1 = off (paper); 2 = even/odd (plastic/glass). Costs N x sweeps.
    shingle_passes: int = 1
    # Band-boundary feathering (R7): overlap consecutive bands by this many rows
    # and split the overlap stochastically, so a Y-advance error blurs into
    # noise instead of a hard seam repeating every band. 0 = off.
    band_overlap: int = 0

    @classmethod
    def from_dict(cls, data: Dict,
                  layout: Optional["head_layout.HeadLayout"] = None
                  ) -> "CalibrationProfile":
        """Build a profile from a parsed JSON dict (see profiles/default.json).

        ``layout`` is the head the profile is about to be used on; when given,
        the profile's plumbing and head name are checked against it."""
        dg = data.get("dot_gain", {})
        dot_gain = DotGainConfig(
            enabled=bool(dg.get("enabled", True)),
            cyan_gain=float(dg.get("C", 0.50)),
            magenta_gain=float(dg.get("M", 0.50)),
            yellow_gain=float(dg.get("Y", 0.375)),
            black_gain=float(dg.get("K", 0.625)),
        )

        raw_scales = data.get("channel_scales", {})
        # LC/LM mirror C/M when not stated explicitly.
        channel_scales = {
            "C": float(raw_scales.get("C", 0.55)),
            "M": float(raw_scales.get("M", 0.80)),
            "Y": float(raw_scales.get("Y", 0.45)),
            "K": float(raw_scales.get("K", 0.55)),
        }
        channel_scales["LC"] = float(raw_scales.get("LC", channel_scales["C"]))
        channel_scales["LM"] = float(raw_scales.get("LM", channel_scales["M"]))

        dpi_comp = data.get("dpi_compensation", {})
        ink_limit = InkLimitConfig(
            enabled=bool(data.get("ink_limit_enabled", True)),
            channel_scales=channel_scales,
            global_scale=float(data.get("global_scale", 1.0)),
            max_total_ink=float(data.get("max_total_ink", 0.90)),
            compensate_dpi=bool(dpi_comp.get("enabled", True)),
            reference_dpi=int(dpi_comp.get("reference_dpi", 360)),
            dpi_power=float(dpi_comp.get("power", 2.0)),
        )

        raw_ink_map = data.get("ink_map")
        ink_map = tuple(raw_ink_map) if raw_ink_map is not None else None

        # A profile's dead nozzles and plumbing are physical facts about one
        # head, so a profile may name the head it was measured on. Checked
        # against the head actually selected, when the caller knows it.
        head = data.get("head")
        if layout is not None:
            if head is not None and head != layout.name:
                raise ValueError(
                    f"profile was measured on head {head!r}, but head "
                    f"{layout.name!r} is selected; the dead-nozzle slots and "
                    f"ink map do not carry over"
                )
            if ink_map is not None:
                layout.validate_ink_map(ink_map)

        raw_dead = data.get("dead_nozzles", {}) or {}
        if ink_map is not None:
            # Slot-keyed on disk -> channel-keyed in memory, via the ink map.
            dead_nozzles = dead_slots_to_channels(raw_dead, ink_map)
        else:
            # Legacy channel-keyed profile: use verbatim.
            dead_nozzles = {ch: sorted(int(i) for i in idxs)
                            for ch, idxs in raw_dead.items()}

        return cls(
            media=str(data.get("media", "unnamed")),
            dot_gain=dot_gain,
            ink_limit=ink_limit,
            black_generation=float(data.get("black_generation", 1.0)),
            black_start=float(data.get("black_start", 0.0)),
            dpi_curves=dict(data.get("dpi_curves", {})),
            dead_nozzles=dead_nozzles,
            ink_map=ink_map,
            head=head,
            drop_diameter_um=(float(data["drop_diameter_um"])
                              if data.get("drop_diameter_um") is not None else None),
            shingle_passes=max(1, int(data.get("shingle_passes", 1))),
            band_overlap=max(0, int(data.get("band_overlap", 0))),
        )

    def to_dict(self) -> Dict:
        """Serialise back to the profile schema (used for provenance metadata)."""
        result = {
            "media": self.media,
            "dot_gain": {
                "enabled": self.dot_gain.enabled,
                "C": self.dot_gain.cyan_gain,
                "M": self.dot_gain.magenta_gain,
                "Y": self.dot_gain.yellow_gain,
                "K": self.dot_gain.black_gain,
            },
            "ink_limit_enabled": self.ink_limit.enabled,
            "channel_scales": dict(self.ink_limit.channel_scales),
            "global_scale": self.ink_limit.global_scale,
            "max_total_ink": self.ink_limit.max_total_ink,
            "dpi_compensation": {
                "enabled": self.ink_limit.compensate_dpi,
                "reference_dpi": self.ink_limit.reference_dpi,
                "power": self.ink_limit.dpi_power,
            },
            "black_generation": self.black_generation,
            "black_start": self.black_start,
            "dpi_curves": dict(self.dpi_curves),
            "drop_diameter_um": self.drop_diameter_um,
            "shingle_passes": self.shingle_passes,
            "band_overlap": self.band_overlap,
        }
        if self.head is not None:
            result["head"] = self.head
        if self.ink_map is not None:
            # Store dead nozzles against physical slots (plumbing-independent).
            result["ink_map"] = list(self.ink_map)
            result["dead_nozzles"] = dead_channels_to_slots(
                self.dead_nozzles, self.ink_map)
        else:
            result["dead_nozzles"] = {ch: list(v)
                                      for ch, v in self.dead_nozzles.items()}
        return result

    @classmethod
    def load(cls, path: str,
             layout: Optional["head_layout.HeadLayout"] = None
             ) -> "CalibrationProfile":
        """Load a profile from a JSON file, checked against ``layout``."""
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh), layout)

    @classmethod
    def default(cls, layout: Optional["head_layout.HeadLayout"] = None
                ) -> "CalibrationProfile":
        """Return the shipped seed profile, falling back to in-code defaults.

        The seed profile is written for c6n90; on any other head its plumbing
        does not apply, so only the colour half of it is kept."""
        try:
            profile = cls.load(str(DEFAULT_PROFILE_PATH))
        except (OSError, ValueError):
            profile = cls()
        if layout is not None and profile.ink_map is not None:
            try:
                layout.validate_ink_map(profile.ink_map)
            except ValueError:
                profile.ink_map = None
                profile.dead_nozzles = {}
        return profile

    def measured_luts(self, dpi: int) -> Optional[Dict[str, "List[int]"]]:
        """Return the measured per-channel linearisation LUTs for ``dpi``, or
        None if the profile has no complete set for it.

        A complete set has a 256-entry LUT for every C, M, Y, K channel. When
        present the RIP applies these in place of dot-gain + DPI-power (they
        subsume both, measured at that DPI); when absent it falls back to the
        heuristic colour stack.
        """
        curves = self.dpi_curves.get(str(dpi)) or self.dpi_curves.get(dpi)
        if not isinstance(curves, dict):
            return None
        luts = {}
        for ch in ("C", "M", "Y", "K"):
            lut = curves.get(ch)
            if not isinstance(lut, (list, tuple)) or len(lut) != 256:
                return None
            luts[ch] = list(lut)
        return luts


# ---------------------------------------------------------------------------
# Blue Noise Generator
# ---------------------------------------------------------------------------

def generate_blue_noise_texture(size: int = 64, seed: int = 42) -> np.ndarray:
    """Generate a blue-noise texture using the Void-and-Cluster algorithm.

    Returns an array of shape (size, size) with values in 0-255.
    """
    rng = np.random.default_rng(seed)

    initial_density = 0.1
    binary_pattern = rng.random((size, size)) < initial_density

    def create_gaussian_kernel(sigma: float = 1.5) -> np.ndarray:
        kernel_size = int(6 * sigma) | 1
        x = np.arange(kernel_size) - kernel_size // 2
        kernel_1d = np.exp(-x**2 / (2 * sigma**2))
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        return kernel_2d / kernel_2d.sum()

    def compute_energy(pattern: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Compute energy via convolution with wrap-around boundary."""
        from scipy.ndimage import convolve
        return convolve(pattern.astype(float), kernel, mode='wrap')

    try:
        from scipy.ndimage import convolve
        kernel = create_gaussian_kernel(1.5)

        # Phase 1: remove points from the tightest cluster
        pattern = binary_pattern.copy()
        rank = np.zeros((size, size), dtype=int)
        current_rank = pattern.sum() - 1

        while pattern.sum() > 0:
            energy = compute_energy(pattern, kernel)
            energy[~pattern] = -np.inf
            tightest = np.unravel_index(energy.argmax(), energy.shape)
            pattern[tightest] = False
            rank[tightest] = current_rank
            current_rank -= 1

        # Phase 2: add points into the largest void
        pattern = binary_pattern.copy()
        current_rank = binary_pattern.sum()

        while pattern.sum() < size * size:
            energy = compute_energy(pattern, kernel)
            energy[pattern] = np.inf
            largest_void = np.unravel_index(energy.argmin(), energy.shape)
            pattern[largest_void] = True
            rank[largest_void] = current_rank
            current_rank += 1

        # Normalise to 0-255
        texture = ((rank / (size * size - 1)) * 255).astype(np.uint8)

    except ImportError:
        # Fallback: filtered white noise (less ideal but functional)
        print("Warning: scipy not found. Using approximate blue noise.")
        noise = rng.random((size, size))
        from PIL import ImageFilter
        img = Image.fromarray((noise * 255).astype(np.uint8), mode='L')
        for _ in range(3):
            img = img.filter(ImageFilter.GaussianBlur(radius=1))
            img = img.filter(ImageFilter.SHARPEN)
        texture = np.array(img)

    return texture


def generate_ordered_dither_matrix(size: int = 8) -> np.ndarray:
    """Generate a Bayer matrix for ordered dithering.

    Size must be a power of 2 (2, 4, 8, 16, ...).
    Returns an array of shape (size, size) with values in 0-255.
    """
    if size == 2:
        base = np.array([[0, 2], [3, 1]])
    else:
        smaller = generate_ordered_dither_matrix(size // 2)
        base = np.zeros((size, size), dtype=int)
        base[0::2, 0::2] = 4 * smaller
        base[0::2, 1::2] = 4 * smaller + 2
        base[1::2, 0::2] = 4 * smaller + 3
        base[1::2, 1::2] = 4 * smaller + 1

    return ((base / (size * size)) * 255).astype(np.uint8)


# Global caches to avoid regenerating textures
_BLUE_NOISE_CACHE: Dict[Tuple[int, int], np.ndarray] = {}
_ORDERED_MATRIX_CACHE: Dict[int, np.ndarray] = {}

# Per-channel blue-noise seeds. Each channel gets its own texture so the dots
# of different inks do not land on top of each other (dot-on-dot causes
# graininess and colour shifts); a distinct seed decorrelates them.
CHANNEL_BLUE_NOISE_SEED = {
    "C": 42, "M": 1013, "Y": 2027, "K": 3041, "LC": 4051, "LM": 5077,
}


def get_blue_noise_texture(size: int = 64, seed: int = 42) -> np.ndarray:
    """Return a cached blue-noise texture, generating it on first access.

    ``seed`` selects an independent texture, used to decorrelate channels.
    """
    key = (size, seed)
    if key not in _BLUE_NOISE_CACHE:
        print(f"   Generating blue noise texture {size}x{size} (seed {seed})...")
        _BLUE_NOISE_CACHE[key] = generate_blue_noise_texture(size, seed=seed)
    return _BLUE_NOISE_CACHE[key]


def get_ordered_matrix(size: int = 8) -> np.ndarray:
    """Return a cached ordered-dither matrix, generating it on first access."""
    if size not in _ORDERED_MATRIX_CACHE:
        _ORDERED_MATRIX_CACHE[size] = generate_ordered_dither_matrix(size)
    return _ORDERED_MATRIX_CACHE[size]


# ---------------------------------------------------------------------------
# Dot Gain Compensation
# ---------------------------------------------------------------------------

def create_dot_gain_curve(gain: float) -> np.ndarray:
    """Create a 256-entry LUT that pre-compensates for dot gain.

    Channel values mean "amount of ink" (0 = no ink, 255 = max ink).
    To compensate dot gain we must REDUCE midtone ink before halftoning.

    A gamma > 1 is used: compensated = input ^ (1 + gain). Example: input
    128 with gain 0.20 maps to ~112, i.e. less ink and a lighter print.
    (A gamma < 1, as in an earlier version, would *increase* midtone ink
    and darken the print, the opposite of dot-gain compensation.)
    """
    x = np.arange(256, dtype=np.float32) / 255.0
    gamma = 1.0 + max(0.0, float(gain))
    y = np.power(x, gamma)
    return np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8)


def apply_dot_gain_compensation(
    cmyk_channels: Dict[str, np.ndarray],
    config: DotGainConfig,
) -> Dict[str, np.ndarray]:
    """Apply dot-gain compensation to every channel via per-channel LUTs."""
    if not config.enabled:
        return cmyk_channels

    compensated = {}
    for channel_name, channel_data in cmyk_channels.items():
        gain = config.get_gain_for_channel(channel_name)
        lut = create_dot_gain_curve(gain)
        compensated[channel_name] = lut[channel_data]

    return compensated


# ---------------------------------------------------------------------------
# ICC Profile Management
# ---------------------------------------------------------------------------

def load_icc_profile(profile_path: str) -> Optional["ImageCms.ImageCmsProfile"]:
    """Load an ICC profile from a file path."""
    if not HAS_ICC:
        return None

    try:
        return ImageCms.getOpenProfile(profile_path)
    except Exception as e:
        print(f"Warning: Failed to load ICC profile: {e}")
        return None


def get_default_srgb_profile() -> Optional["ImageCms.ImageCmsProfile"]:
    """Return the built-in sRGB profile."""
    if not HAS_ICC:
        return None
    return ImageCms.createProfile('sRGB')


def convert_image_with_icc(
    image: Image.Image,
    input_profile: Optional["ImageCms.ImageCmsProfile"] = None,
    output_profile: Optional["ImageCms.ImageCmsProfile"] = None,
    rendering_intent: int = None,
) -> Image.Image:
    """Convert an image to CMYK using ICC profiles.

    If no input profile is supplied the embedded profile is used, falling
    back to sRGB. If no output profile is available a naive conversion is
    performed instead.
    """
    if not HAS_ICC or output_profile is None:
        return image.convert("CMYK")

    if rendering_intent is None:
        rendering_intent = ImageCms.Intent.PERCEPTUAL

    # Use the embedded profile when available
    if input_profile is None:
        if 'icc_profile' in image.info:
            try:
                input_profile = ImageCms.ImageCmsProfile(
                    io.BytesIO(image.info['icc_profile'])
                )
            except Exception:
                input_profile = get_default_srgb_profile()
        else:
            input_profile = get_default_srgb_profile()

    if image.mode != 'RGB':
        image = image.convert('RGB')

    try:
        transform = ImageCms.buildTransform(
            input_profile,
            output_profile,
            'RGB',
            'CMYK',
            renderingIntent=rendering_intent,
        )
        return ImageCms.applyTransform(image, transform)
    except Exception as e:
        print(f"Warning: ICC conversion failed: {e}. Using naive conversion.")
        return image.convert("CMYK")


def convert_rgb_to_cmyk_ucr_channels(
    image: Image.Image,
    black_generation: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Fallback RGB -> CMYK conversion with basic UCR/GCR.

    PIL's plain image.convert("CMYK") is a device-independent inversion:
    black RGB becomes C=M=Y=255, K=0. For an actual inkjet this can dump
    three inks for neutral shadows and easily cause dark/green casts.

    This fallback generates K from the neutral component and removes that
    amount from CMY, so neutral dark areas primarily use the black channel.
    A proper printer ICC profile is still better when available.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    c = 1.0 - rgb[..., 0]
    m = 1.0 - rgb[..., 1]
    y = 1.0 - rgb[..., 2]

    neutral = np.minimum(np.minimum(c, m), y)
    k = np.clip(neutral * black_generation, 0.0, 1.0)

    c = np.clip(c - k, 0.0, 1.0)
    m = np.clip(m - k, 0.0, 1.0)
    y = np.clip(y - k, 0.0, 1.0)

    return {
        "C": np.clip(np.rint(c * 255.0), 0, 255).astype(np.uint8),
        "M": np.clip(np.rint(m * 255.0), 0, 255).astype(np.uint8),
        "Y": np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8),
        "K": np.clip(np.rint(k * 255.0), 0, 255).astype(np.uint8),
    }


def apply_ink_limits(
    cmyk_channels: Dict[str, np.ndarray],
    config: InkLimitConfig,
    dpi: int,
) -> Dict[str, np.ndarray]:
    """Apply per-channel scaling, total ink limiting, and DPI compensation."""
    if not config.enabled:
        return cmyk_channels

    dpi_scale = 1.0
    if config.compensate_dpi and dpi > 0 and config.reference_dpi > 0:
        # With the same drop size, doubling both X and Y DPI gives 4x droplets
        # per area at the same 100% bitmap coverage.
        dpi_scale = min(1.0, (config.reference_dpi / dpi) ** config.dpi_power)

    scaled_float: Dict[str, np.ndarray] = {}
    for channel_name, channel_data in cmyk_channels.items():
        ch_scale = config.channel_scales.get(channel_name, 1.0)
        scale = config.global_scale * dpi_scale * ch_scale
        scaled_float[channel_name] = (channel_data.astype(np.float32) / 255.0) * scale

    if config.max_total_ink > 0:
        total = None
        for arr in scaled_float.values():
            total = arr if total is None else total + arr

        if total is not None:
            ratio = np.minimum(1.0, config.max_total_ink / np.maximum(total, 1e-6))
            for channel_name in scaled_float:
                scaled_float[channel_name] *= ratio

    return {
        channel_name: np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
        for channel_name, arr in scaled_float.items()
    }


def apply_total_ink_limit(
    cmyk_channels: Dict[str, np.ndarray],
    max_total_ink: float,
) -> Dict[str, np.ndarray]:
    """Cap the summed CMYK ink per pixel, scaling all channels down in a pixel
    that exceeds ``max_total_ink`` (a fraction). Used after the linearisation
    LUT, where per-channel scaling no longer applies."""
    if max_total_ink <= 0:
        return cmyk_channels

    floats = {name: arr.astype(np.float32) / 255.0 for name, arr in cmyk_channels.items()}
    total = None
    for arr in floats.values():
        total = arr if total is None else total + arr
    if total is None:
        return cmyk_channels
    ratio = np.minimum(1.0, max_total_ink / np.maximum(total, 1e-6))
    return {
        name: np.clip(np.rint(arr * ratio * 255.0), 0, 255).astype(np.uint8)
        for name, arr in floats.items()
    }


def _reflectance_to_lstar(r: np.ndarray) -> np.ndarray:
    """CIE L* from relative reflectance (0..1)."""
    r = np.clip(r, 0.0, 1.0)
    return np.where(r > 0.008856, 116.0 * np.cbrt(r) - 16.0, 903.3 * r)


def murray_davies_luts(dpi: int, drop_diameter_um: float,
                       r_solid: float = 0.05) -> Dict[str, "List[int]"]:
    """Modelled per-channel linearisation LUTs for an unmeasured DPI.

    A drop of diameter ``drop_diameter_um`` covers ``k`` cell-areas at the
    raster pitch; random placement gives covered area ``1 - exp(-k*c)``
    (Murray-Davies reflectance), which is inverted for a perceptually linear
    (L*) ramp, the same shape a measurement would produce, but from one
    per-media number. Interpolates far better than a fixed gamma when a DPI
    hasn't been measured; the same curve is used for all channels.
    """
    cell_um = 25400.0 / dpi
    k = math.pi * (drop_diameter_um / 2.0) ** 2 / (cell_um ** 2)
    c = np.linspace(0.0, 1.0, 256)
    covered = 1.0 - np.exp(-k * c)
    lstar = _reflectance_to_lstar(1.0 - covered * (1.0 - r_solid))
    span = max(float(lstar[0] - lstar[-1]), 1e-6)
    darkness = np.clip((lstar[0] - lstar) / span, 0.0, 1.0)
    darkness = np.maximum.accumulate(darkness) + np.linspace(0.0, 1e-6, 256)
    coverage = np.interp(np.linspace(0.0, 1.0, 256), darkness, c)
    lut = np.clip(np.rint(coverage * 255.0), 0, 255).astype(int).tolist()
    return {ch: list(lut) for ch in ("C", "M", "Y", "K")}


def apply_linearization_luts(
    cmyk_channels: Dict[str, np.ndarray],
    luts: Dict[str, "List[int]"],
) -> Dict[str, np.ndarray]:
    """Map each channel value through its measured linearisation LUT.

    ``luts`` maps a channel name to a 256-entry list: ``lut[v]`` is the
    coverage (0..255) to command for desired channel value ``v``. Channels
    without a LUT pass through unchanged.
    """
    out = {}
    for name, channel in cmyk_channels.items():
        lut = luts.get(name)
        if lut is None or len(lut) != 256:
            out[name] = channel
        else:
            table = np.asarray(lut, dtype=np.uint8)
            out[name] = table[channel]
    return out


def load_image_as_cmyk(
    path: str,
    icc_profile_path: Optional[str] = None,
    black_generation: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Load an image and convert it to CMYK channels.

    Returns a dict mapping "C", "M", "Y", "K" to uint8 arrays of shape
    (height, width) with values in 0..255.
    """
    image = Image.open(path)

    output_profile = None
    if icc_profile_path:
        output_profile = load_icc_profile(icc_profile_path)
        if output_profile:
            print(f"   Using ICC profile: {icc_profile_path}")

    if output_profile or (HAS_ICC and 'icc_profile' in image.info):
        cmyk_image = convert_image_with_icc(image, output_profile=output_profile)
    elif image.mode == "CMYK":
        cmyk_image = image
    else:
        return convert_rgb_to_cmyk_ucr_channels(image, black_generation=black_generation)

    cyan, magenta, yellow, black = cmyk_image.split()

    return {
        "C": np.array(cyan, dtype=np.uint8),
        "M": np.array(magenta, dtype=np.uint8),
        "Y": np.array(yellow, dtype=np.uint8),
        "K": np.array(black, dtype=np.uint8),
    }


def resize_cmyk_channels(
    channels: Dict[str, np.ndarray],
    target_width_px: int,
    target_height_px: int,
) -> Dict[str, np.ndarray]:
    """Resample continuous-tone CMYK channels to a target pixel size.

    Done before dithering, so each channel is resampled as a greyscale image
    (Lanczos) rather than as a 1-bit halftone.
    """
    resized = {}
    for name, channel in channels.items():
        image = Image.fromarray(channel, mode="L").resize(
            (target_width_px, target_height_px), Image.LANCZOS
        )
        resized[name] = np.array(image, dtype=np.uint8)
    return resized


# ---------------------------------------------------------------------------
# Photographic front-end (R6): 16-bit input, float pipeline, linear-light resample
# ---------------------------------------------------------------------------

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB (0..1) -> linear light."""
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, np.power((x + a) / (1 + a), 2.4))


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Linear light -> sRGB (0..1)."""
    a = 0.055
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * np.power(x, 1 / 2.4) - a)


def load_image_rgb_float(path: str) -> Tuple[Optional[np.ndarray], Optional[bytes], Optional["Image.Image"]]:
    """Load an image as float32 RGB in 0..1, preserving 16-bit precision.

    PIL's ``convert("RGB")`` truncates 16-bit input to 8 bits, quantising
    shadows before the halftone (banding in gradients); this keeps the full
    precision as float. Returns ``(rgb, icc_bytes, cmyk_image)``. ``rgb`` is
    None and ``cmyk_image`` is set for a CMYK-mode input (handled the old way).
    """
    image = Image.open(path)
    icc = image.info.get("icc_profile")
    if image.mode == "CMYK":
        return None, icc, image
    if image.mode in ("RGBA", "LA", "PA"):
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(bg, rgba).convert("RGB")

    arr = np.asarray(image)
    if arr.dtype == np.uint16:
        rgb = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        rgb = arr.astype(np.float32) / 255.0
    else:
        rgb = arr.astype(np.float32)
        if rgb.size and rgb.max() > 1.0:
            rgb = rgb / 255.0
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    return np.ascontiguousarray(rgb[..., :3]), icc, None


def resample_rgb_linear(rgb: np.ndarray, target_width_px: int, target_height_px: int) -> np.ndarray:
    """Lanczos resample RGB (0..1) in **linear light**, returning sRGB 0..1.

    Resampling gamma-encoded values darkens fine detail; decoding to linear
    first averages physically-correct luminances.
    """
    lin = srgb_to_linear(rgb)
    out = np.empty((target_height_px, target_width_px, 3), dtype=np.float32)
    for c in range(3):
        plane = Image.fromarray(np.ascontiguousarray(lin[..., c], dtype=np.float32), mode="F")
        plane = plane.resize((target_width_px, target_height_px), Image.LANCZOS)
        out[..., c] = np.asarray(plane, dtype=np.float32)
    return linear_to_srgb(np.clip(out, 0.0, 1.0)).astype(np.float32)


def gcr_black(neutral: np.ndarray, black_generation: float, black_start: float) -> np.ndarray:
    """Black generation from the neutral component with an adjustable start point
    (R2): below ``black_start`` no K is generated (highlights stay CMY), above it
    K ramps to ``black_generation`` of the neutral. ``black_start=0`` reproduces
    the plain ``neutral * black_generation``."""
    if black_start <= 0.0:
        ramp = neutral
    else:
        ramp = np.clip((neutral - black_start) / max(1.0 - black_start, 1e-6), 0.0, 1.0)
    return np.clip(ramp * black_generation, 0.0, 1.0)


def rgb_float_to_cmyk_ucr(rgb: np.ndarray, black_generation: float = 1.0,
                          black_start: float = 0.0) -> Dict[str, np.ndarray]:
    """Float RGB (0..1) -> CMYK uint8 with UCR/GCR (see convert_rgb_to_cmyk_ucr_channels)."""
    c = 1.0 - rgb[..., 0]
    m = 1.0 - rgb[..., 1]
    y = 1.0 - rgb[..., 2]
    k = gcr_black(np.minimum(np.minimum(c, m), y), black_generation, black_start)
    c, m, y = np.clip(c - k, 0.0, 1.0), np.clip(m - k, 0.0, 1.0), np.clip(y - k, 0.0, 1.0)
    return {name: np.clip(np.rint(v * 255.0), 0, 255).astype(np.uint8)
            for name, v in (("C", c), ("M", m), ("Y", y), ("K", k))}


def _split_cmyk_image(cmyk_image: "Image.Image") -> Dict[str, np.ndarray]:
    cyan, magenta, yellow, black = cmyk_image.split()
    return {"C": np.array(cyan, dtype=np.uint8), "M": np.array(magenta, dtype=np.uint8),
            "Y": np.array(yellow, dtype=np.uint8), "K": np.array(black, dtype=np.uint8)}


def prepare_cmyk_channels(
    input_path: str,
    dpi: int,
    print_width_mm: Optional[float],
    print_height_mm: Optional[float],
    icc_profile_path: Optional[str],
    black_generation: float,
    black_start: float = 0.0,
) -> Tuple[Dict[str, np.ndarray], Tuple[int, int], Tuple[int, int], bool]:
    """R6 front-end: load (16-bit aware), resample in linear light, then convert
    RGB->CMYK in float. Returns ``(cmyk, (w,h), (native_w,native_h), resampled)``."""
    rgb, _icc_bytes, cmyk_image = load_image_rgb_float(input_path)

    if cmyk_image is not None:  # CMYK-mode input: keep the legacy path
        cmyk = _split_cmyk_image(cmyk_image)
        native_h, native_w = cmyk["C"].shape
        w, h = resolve_target_size_px(native_w, native_h, dpi, print_width_mm, print_height_mm)
        resampled = (w, h) != (native_w, native_h)
        if resampled:
            cmyk = resize_cmyk_channels(cmyk, w, h)
        return cmyk, (w, h), (native_w, native_h), resampled

    native_h, native_w = rgb.shape[:2]
    w, h = resolve_target_size_px(native_w, native_h, dpi, print_width_mm, print_height_mm)
    resampled = (w, h) != (native_w, native_h)
    if resampled:
        rgb = resample_rgb_linear(rgb, w, h)

    output_profile = load_icc_profile(icc_profile_path) if icc_profile_path else None
    if output_profile:
        print(f"   Using ICC profile: {icc_profile_path}")
        pil = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), "RGB")
        cmyk = _split_cmyk_image(convert_image_with_icc(pil, output_profile=output_profile))
    else:
        cmyk = rgb_float_to_cmyk_ucr(rgb, black_generation, black_start)
    return cmyk, (w, h), (native_w, native_h), resampled


def resolve_target_size_px(
    native_width_px: int,
    native_height_px: int,
    dpi: int,
    print_width_mm: Optional[float],
    print_height_mm: Optional[float],
) -> Tuple[int, int]:
    """Resolve the target raster size (px) from a requested physical size.

    A requested width/height (mm) is converted to pixels at the given DPI. If
    only one is given, the other is scaled to preserve the aspect ratio. If
    neither is given, the native size is kept.
    """
    target_width = (
        max(1, round(print_width_mm / 25.4 * dpi)) if print_width_mm else None
    )
    target_height = (
        max(1, round(print_height_mm / 25.4 * dpi)) if print_height_mm else None
    )

    if target_width is None and target_height is None:
        return native_width_px, native_height_px
    if target_width is None:
        target_width = max(1, round(native_width_px * target_height / native_height_px))
    if target_height is None:
        target_height = max(1, round(native_height_px * target_width / native_width_px))

    return target_width, target_height


# ---------------------------------------------------------------------------
# Dithering Algorithms
# ---------------------------------------------------------------------------

if HAS_NUMBA:
    @njit(cache=True)
    def _floyd_steinberg_numba(pixel_buffer: np.ndarray, output_bitmap: np.ndarray,
                               dead_rows: np.ndarray) -> None:
        """Numba-accelerated serpentine Floyd-Steinberg error diffusion (in-place).

        Alternating the scan direction each row (boustrophedon) breaks up the
        directional "worm" artifacts a fixed left-to-right raster produces.
        The error weights are mirrored on right-to-left rows so ``d`` is always
        the forward direction.

        A row flagged in ``dead_rows`` (printed by a dead nozzle) never
        deposits a dot; its whole value becomes quantisation error and diffuses
        to the neighbouring printable rows (N2a reroute).
        """
        height, width = pixel_buffer.shape

        for y in range(height):
            d = 1 if (y % 2 == 0) else -1
            x = 0 if d == 1 else width - 1
            dead = dead_rows[y] != 0
            for _ in range(width):
                old_value = pixel_buffer[y, x]
                new_value = 0.0 if dead else (255.0 if old_value >= 128.0 else 0.0)
                output_bitmap[y, x] = 1 if new_value > 0 else 0
                quant_error = old_value - new_value

                if 0 <= x + d < width:
                    pixel_buffer[y, x + d] += quant_error * 0.4375
                if y + 1 < height:
                    if 0 <= x - d < width:
                        pixel_buffer[y + 1, x - d] += quant_error * 0.1875
                    pixel_buffer[y + 1, x] += quant_error * 0.3125
                    if 0 <= x + d < width:
                        pixel_buffer[y + 1, x + d] += quant_error * 0.0625
                x += d


def apply_floyd_steinberg(grayscale_channel: np.ndarray,
                          dead_rows: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply serpentine Floyd-Steinberg dithering to a 0..255 channel, returning a 0/1 bitmap.

    ``dead_rows`` (bool/uint8, length = height) marks rows a dead nozzle would
    print: those never get a dot and their ink diffuses to neighbours (N2a)."""
    height, width = grayscale_channel.shape
    pixel_buffer = grayscale_channel.astype(np.float32)
    output_bitmap = np.zeros((height, width), dtype=np.uint8)
    if dead_rows is None:
        dead = np.zeros(height, dtype=np.uint8)
    else:
        dead = np.ascontiguousarray(dead_rows).astype(np.uint8)

    if HAS_NUMBA:
        _floyd_steinberg_numba(pixel_buffer, output_bitmap, dead)
    else:
        for y in range(height):
            d = 1 if (y % 2 == 0) else -1
            xs = range(width) if d == 1 else range(width - 1, -1, -1)
            row_dead = dead[y] != 0
            for x in xs:
                old_value = pixel_buffer[y, x]
                new_value = 0.0 if row_dead else (255.0 if old_value >= 128.0 else 0.0)
                output_bitmap[y, x] = 1 if new_value > 0 else 0
                quant_error = old_value - new_value

                if 0 <= x + d < width:
                    pixel_buffer[y, x + d] += quant_error * 0.4375
                if y + 1 < height:
                    if 0 <= x - d < width:
                        pixel_buffer[y + 1, x - d] += quant_error * 0.1875
                    pixel_buffer[y + 1, x] += quant_error * 0.3125
                    if 0 <= x + d < width:
                        pixel_buffer[y + 1, x + d] += quant_error * 0.0625

    return output_bitmap


def apply_blue_noise_dither(
    grayscale_channel: np.ndarray,
    noise_texture: np.ndarray,
) -> np.ndarray:
    """Apply threshold dithering using a tiled blue-noise texture.

    Returns a 0/1 bitmap.
    """
    height, width = grayscale_channel.shape
    noise_h, noise_w = noise_texture.shape

    tiles_y = (height + noise_h - 1) // noise_h
    tiles_x = (width + noise_w - 1) // noise_w
    tiled_noise = np.tile(noise_texture, (tiles_y, tiles_x))[:height, :width]

    output_bitmap = (grayscale_channel > tiled_noise).astype(np.uint8)
    return output_bitmap


def apply_ordered_dither(
    grayscale_channel: np.ndarray,
    dither_matrix: np.ndarray,
) -> np.ndarray:
    """Apply ordered dithering using a tiled Bayer matrix.

    Returns a 0/1 bitmap.
    """
    height, width = grayscale_channel.shape
    matrix_h, matrix_w = dither_matrix.shape

    tiles_y = (height + matrix_h - 1) // matrix_h
    tiles_x = (width + matrix_w - 1) // matrix_w
    tiled_matrix = np.tile(dither_matrix, (tiles_y, tiles_x))[:height, :width]

    output_bitmap = (grayscale_channel > tiled_matrix).astype(np.uint8)
    return output_bitmap


def apply_dithering(
    grayscale_channel: np.ndarray,
    config: DitherConfig,
    blue_noise_seed: int = 42,
    dead_rows: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Dispatch to the configured dithering method.

    ``blue_noise_seed`` selects the channel's independent blue-noise texture
    (ignored by the other methods). ``dead_rows`` (N2a reroute) is only
    honoured by Floyd-Steinberg: error diffusion is what carries a dead
    row's ink to its neighbours. Returns a 0/1 bitmap.
    """
    if config.method == "floyd_steinberg":
        return apply_floyd_steinberg(grayscale_channel, dead_rows=dead_rows)

    elif config.method == "blue_noise":
        texture = get_blue_noise_texture(config.blue_noise_size, seed=blue_noise_seed)
        return apply_blue_noise_dither(grayscale_channel, texture)

    elif config.method == "ordered":
        matrix = get_ordered_matrix(config.ordered_matrix_size)
        return apply_ordered_dither(grayscale_channel, matrix)

    else:
        raise ValueError(f"Unknown dithering method: {config.method}")


def convert_to_halftone_bitmaps(
    cmyk_channels: Dict[str, np.ndarray],
    dither_config: DitherConfig,
    dead_masks: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Dither every CMYK channel into a 0/1 bitmap.

    Each channel uses its own blue-noise texture (see CHANNEL_BLUE_NOISE_SEED)
    so different inks do not print dot-on-dot. ``dead_masks`` (N2a reroute)
    maps a channel to a per-row dead-nozzle mask.
    """
    halftone_bitmaps: Dict[str, np.ndarray] = {}
    for channel_name, channel_data in cmyk_channels.items():
        seed = CHANNEL_BLUE_NOISE_SEED.get(channel_name, 42)
        dead = dead_masks.get(channel_name) if dead_masks else None
        halftone_bitmaps[channel_name] = apply_dithering(channel_data, dither_config, seed, dead)
    return halftone_bitmaps


# ---------------------------------------------------------------------------
# Print Pass Generation
# ---------------------------------------------------------------------------

def generate_print_passes(
    halftone_bitmaps: Dict[str, np.ndarray],
    config: PrintheadConfig,
    input_channel_order: Optional[Tuple[str, ...]] = None,
    band_overlap: int = 0,
    feather_seed: int = 12345,
) -> List[np.ndarray]:
    """Convert halftone bitmaps into a list of print passes.

    Each pass is an array of shape (channels, nozzle_count, image_width), where
    nozzle_count is the largest plumbed slot's; a channel fed by a shorter slot
    fills only the first rows and leaves the rest zeroed (the payload header
    carries the per-slot counts).

    The passes are ordered bottom-up in image terms: the machine origin is
    the bottom-left corner and every pass steps +Y, so pass 0 (the lowest
    Y) must carry the bottom of the image. Machine line L holds image row
    H-1-L; nozzle n of a slot sits passes_per_band lines above nozzle n-1, and
    the slot itself starts ``first_nozzle`` nozzles up its column, which is
    what makes an ink's rows depend on the plumbing (see head_layout.py).

    ``band_overlap`` > 0 (R7) advances only ``lines_per_band - band_overlap``
    per band, so consecutive bands share ``band_overlap`` lines; in that
    overlap a stochastic per-pixel mask splits the lines between the two bands
    (probability ramped across the zone), turning the hard band seam (which
    a small Y-advance error repeats every band) into dithered noise. Each
    slot meets that seam at its own height, so the mask is applied against the
    slot's own band edges.
    """
    if not halftone_bitmaps:
        raise ValueError("halftone_bitmaps is empty")

    geometry = config.geometry
    if input_channel_order is None:
        input_channel_order = geometry.imaging_channels

    sample_bitmap = next(iter(halftone_bitmaps.values()))
    image_height, image_width = sample_bitmap.shape

    slots = []
    for channel_name in input_channel_order:
        if channel_name not in halftone_bitmaps:
            raise ValueError(f"Channel '{channel_name}' not found")
        if halftone_bitmaps[channel_name].shape != (image_height, image_width):
            raise ValueError(f"Channel '{channel_name}' has inconsistent shape")
        slot = geometry.slot_for(channel_name)
        if slot is None:
            raise ValueError(
                f"Channel '{channel_name}' is not plumbed into head "
                f"{geometry.layout.name} (ink map: {', '.join(geometry.ink_map)})"
            )
        slots.append(slot)

    stacked_bitmaps = np.stack(
        [halftone_bitmaps[ch] for ch in input_channel_order],
        axis=0
    )

    # The image's row axis points the other way (row 0 is the top), so the
    # rows are flipped here; the partial last band, if any, ends up at the
    # highest Y with the image's top edge. Without this flip the print
    # comes out vertically mirrored on the bed.
    stacked_bitmaps = stacked_bitmaps[:, ::-1, :]

    array_nozzles = max(s.nozzle_count for s in slots)
    pitch = geometry.nozzle_pitch_lines
    band_overlap = geometry.clamp_band_overlap(band_overlap)
    step = geometry.band_step_lines - band_overlap
    lead_in = geometry.lead_in_bands(band_overlap)
    band_count = max(1, -(-image_height // step))

    # Per-boundary feather masks: earlier_owns[bnd][i, x] True -> the earlier
    # band prints line i of that overlap. Probability ramps 1 -> 0 across the
    # zone so ownership crosses smoothly from the earlier to the later band.
    # The mask is indexed by position within the overlap zone, so slots at
    # different heights share it without their seams lining up.
    earlier_owns: Dict[int, np.ndarray] = {}
    if band_overlap > 0:
        rng = np.random.default_rng(feather_seed)
        ramp = 1.0 - (np.arange(band_overlap) + 0.5) / band_overlap
        for bnd in range(-lead_in, band_count - 1):
            earlier_owns[bnd] = rng.random((band_overlap, image_width)) < ramp[:, None]

    print_passes: List[np.ndarray] = []
    for band_index in range(-lead_in, band_count):
        band_start_line = band_index * step

        for pass_offset in range(geometry.interleave_passes):
            head_line = band_start_line + pass_offset
            if head_line >= image_height:
                break

            pass_data = np.zeros(
                (len(input_channel_order), array_nozzles, image_width),
                dtype=np.uint8
            )

            for ci, slot in enumerate(slots):
                nozzles = np.arange(slot.nozzle_count)
                lines = head_line + (slot.first_nozzle + nozzles) * pitch
                valid = (lines >= 0) & (lines < image_height)
                if not valid.any():
                    continue
                pass_data[ci, nozzles[valid], :] = stacked_bitmaps[ci, lines[valid], :]

                if band_overlap <= 0:
                    continue
                # This slot's own band edges: the zone it shares with the band
                # before it, and the one it shares with the band after.
                top_lo = band_start_line + slot.first_nozzle * pitch
                bot_lo = top_lo + slot.nozzle_count * pitch - band_overlap
                for n in nozzles[valid]:
                    fr = int(lines[n])
                    if (band_index - 1) in earlier_owns and top_lo <= fr < top_lo + band_overlap:
                        # This band is the LATER one here: keep where NOT earlier-owned.
                        keep = ~earlier_owns[band_index - 1][fr - top_lo]
                        pass_data[ci, n, :] *= keep.astype(np.uint8)
                    elif band_index in earlier_owns and bot_lo <= fr < bot_lo + band_overlap:
                        # This band is the EARLIER one here: keep where earlier-owned.
                        keep = earlier_owns[band_index][fr - bot_lo]
                        pass_data[ci, n, :] *= keep.astype(np.uint8)

            print_passes.append(pass_data)

    return print_passes


# ---------------------------------------------------------------------------
# Dead-nozzle compensation (N2)
# ---------------------------------------------------------------------------

def dead_nozzle_row_masks(
    dead_nozzles: Dict[str, "List[int]"],
    image_height: int,
    config: PrintheadConfig,
) -> Dict[str, np.ndarray]:
    """Per-channel boolean mask (length image_height) of rows a dead nozzle prints.

    Machine line ``L = H-1-R`` is printed by the nozzle whose position up the
    column matches: ``L // passes_per_band`` counts nozzle pitches from the
    bottom of the column, the slot's own ``first_nozzle`` is subtracted, and
    the band step folds that onto a nozzle index. A dead nozzle therefore owns
    ``passes_per_band`` consecutive rows per band, and *which* rows depends on
    the slot the ink is plumbed into, so re-plumbing moves the streaks.

    A slot longer than the band step sees each row on several bands; a dead
    nozzle in one of them is masked for the whole row (conservative: better a
    row compensated too often than a streak left through).
    """
    geometry = config.geometry
    pitch = geometry.nozzle_pitch_lines
    band_step = geometry.band_step_nozzles
    lines = image_height - 1 - np.arange(image_height)

    masks: Dict[str, np.ndarray] = {}
    for channel, indices in dead_nozzles.items():
        if not indices:
            continue
        slot = geometry.slot_for(channel)
        if slot is None:
            continue
        dead = sorted({int(i) % band_step for i in indices
                       if 0 <= int(i) < slot.nozzle_count})
        if not dead:
            continue
        position = ((lines // pitch) - slot.first_nozzle) % band_step
        masks[channel] = np.isin(position, np.asarray(dead))
    return masks


def limit_reroute_ink(
    channel: np.ndarray,
    dead_rows: np.ndarray,
    reach: int,
) -> np.ndarray:
    """Zero a channel on dead rows with no healthy row within ``reach`` rows (N2a).

    Reroute diffuses a dead row's ink to its neighbours during Floyd-Steinberg.
    In a large dead cluster there is no healthy row nearby, so the error would
    cascade far down and dump the ink well away from the gap. Dropping the ink
    of unreachable dead rows keeps reroute a *local* smear (near the cluster
    edges) instead: the deep interior stays honestly blank, nothing lands far.
    Dead rows still get the mask in dithering, so they never fire regardless.
    """
    dead = np.ascontiguousarray(dead_rows).astype(bool)
    if not dead.any():
        return channel
    n = dead.shape[0]
    far = n + 1
    dist = np.full(n, far, dtype=np.int64)
    last = -far                                  # nearest healthy row above
    for i in range(n):
        if not dead[i]:
            last = i
        dist[i] = i - last
    last = 2 * far                               # nearest healthy row below
    for i in range(n - 1, -1, -1):
        if not dead[i]:
            last = i
        dist[i] = min(int(dist[i]), last - i)
    unreachable = dead & (dist > reach)
    if not unreachable.any():
        return channel
    out = channel.copy()
    out[unreachable, :] = 0
    return out


def slot_nozzle_counts(config: PrintheadConfig) -> Dict[str, int]:
    """Nozzles per channel, from the slot each ink is plumbed into."""
    geometry = config.geometry
    return {ink: slot.nozzle_count
            for ink, slot in zip(geometry.ink_map, geometry.layout.slots)
            if ink != EMPTY_SLOT}


def classify_retouch_donors(
    dead_nozzles: Dict[str, "List[int]"],
    nozzle_count,
) -> Tuple[Dict[str, "List[int]"], Dict[str, "List[int]"], Dict[str, "List[int]"]]:
    """Assign each dead nozzle a *healthy* donor neighbour for retouch (N2b).

    Prefer ``n-1``; if it's also dead (or absent), fall back to ``n+1``; if both
    are dead, the nozzle is in the interior of a cluster and cannot be
    compensated. Returns ``(up, down, skipped)`` per-channel nozzle lists:
    ``up`` served by ``n-1``, ``down`` by ``n+1``, ``skipped`` by neither.

    ``nozzle_count`` is either one count for every channel or a per-channel
    mapping, since slots need not be the same length."""
    counts = (nozzle_count if isinstance(nozzle_count, dict)
              else {ch: nozzle_count for ch in dead_nozzles})
    up: Dict[str, List[int]] = {}
    down: Dict[str, List[int]] = {}
    skipped: Dict[str, List[int]] = {}
    for channel, indices in dead_nozzles.items():
        limit = counts.get(channel, 0)
        dead = {int(i) for i in indices}
        for n in sorted(dead):
            if n - 1 >= 0 and (n - 1) not in dead:
                up.setdefault(channel, []).append(n)
            elif n + 1 < limit and (n + 1) not in dead:
                down.setdefault(channel, []).append(n)
            else:
                skipped.setdefault(channel, []).append(n)
    return up, down, skipped


def generate_retouch_passes(
    halftone_bitmaps: Dict[str, np.ndarray],
    dead_nozzles: Dict[str, "List[int]"],
    config: PrintheadConfig,
    y_positions_mm: List[float],
    input_channel_order: Optional[Tuple[str, ...]] = None,
    band_overlap: int = 0,
) -> Tuple[List[np.ndarray], List[float]]:
    """Build retouch passes that reprint dead-nozzle rows with the nearest
    *healthy* neighbour (N2b).

    Each dead nozzle borrows from ``n-1`` if healthy, else ``n+1`` if healthy,
    else it is skipped (interior of a cluster: no healthy neighbour to lend a
    jet). ``n-1`` donors shift the sliced data down one nozzle (j <- j+1) and
    move the head up one pitch; ``n+1`` donors shift up (j <- j-1) and move down
    one pitch, so in both cases the healthy jet lands on the dead rows. All-empty
    passes are dropped. Returns ``(passes, y_positions_mm)``.
    """
    if not dead_nozzles:
        return [], []

    if input_channel_order is None:
        input_channel_order = config.geometry.imaging_channels
    image_height = next(iter(halftone_bitmaps.values())).shape[0]
    up, down, _skipped = classify_retouch_donors(
        dead_nozzles, slot_nozzle_counts(config))

    def retouch_bitmaps(subset: Dict[str, "List[int]"]) -> Dict[str, np.ndarray]:
        masks = dead_nozzle_row_masks(subset, image_height, config)
        out: Dict[str, np.ndarray] = {}
        for channel in input_channel_order:
            bm = halftone_bitmaps[channel]
            mask = masks.get(channel)
            out[channel] = (bm * mask[:, None].astype(bm.dtype)) if mask is not None \
                else np.zeros_like(bm)
        return out

    y_offset_mm = config.passes_per_band * config.line_spacing_mm  # one nozzle pitch
    kept_passes: List[np.ndarray] = []
    kept_y: List[float] = []

    def emit(subset: Dict[str, "List[int]"], shift_down: bool, dy: float) -> None:
        if not subset:
            return
        passes = generate_print_passes(retouch_bitmaps(subset), config,
                                       input_channel_order, band_overlap=band_overlap)
        for p, y in zip(passes, y_positions_mm):
            sp = np.zeros_like(p)
            if shift_down:
                sp[:, :-1, :] = p[:, 1:, :]     # j <- j+1: donor n-1, head up one pitch
            else:
                sp[:, 1:, :] = p[:, :-1, :]      # j <- j-1: donor n+1, head down one pitch
            if sp.any():
                kept_passes.append(sp)
                kept_y.append(y + dy)

    emit(up, shift_down=True, dy=y_offset_mm)
    emit(down, shift_down=False, dy=-y_offset_mm)
    return kept_passes, kept_y


def apply_shingling(
    print_passes: List[np.ndarray],
    y_positions_mm: List[float],
    shingle_passes: int,
) -> Tuple[List[np.ndarray], List[float]]:
    """Split each swath into ``shingle_passes`` sweeps, each firing every Nth
    column (R8), so wet drops settle between depositions on non-absorbent media.

    The sub-sweeps share the swath's Y (the head sweeps it N times with
    complementary column masks); empty sub-sweeps are dropped. Costs N x
    sweeps, a per-media option (paper off, plastic/glass on), not a default.
    """
    if shingle_passes <= 1:
        return print_passes, list(y_positions_mm)

    out_passes: List[np.ndarray] = []
    out_y: List[float] = []
    for pass_data, y in zip(print_passes, y_positions_mm):
        for k in range(shingle_passes):
            sub = np.zeros_like(pass_data)
            sub[:, :, k::shingle_passes] = pass_data[:, :, k::shingle_passes]
            if sub.any():
                out_passes.append(sub)
                out_y.append(y)
    return out_passes, out_y


def add_light_ink_channels(
    print_passes: List[np.ndarray],
    light_channel_count: int = 2
) -> List[np.ndarray]:
    """Append zeroed planes for the channels the colour path does not render.

    The payload always carries one plane per channel the head declares, in the
    layout's channel order; the light inks (LC/LM on c6n90) have no colour path
    yet and travel blank. A head that declares none (c4n180) adds nothing.
    """
    if not print_passes or light_channel_count <= 0:
        return print_passes

    expanded_passes = []

    for pass_data in print_passes:
        current_channels, nozzle_count, width = pass_data.shape
        expanded = np.zeros(
            (current_channels + light_channel_count, nozzle_count, width),
            dtype=pass_data.dtype
        )
        expanded[:current_channels] = pass_data
        expanded_passes.append(expanded)

    return expanded_passes


# ---------------------------------------------------------------------------
# Y-Position Calculation
# ---------------------------------------------------------------------------

def compute_pass_y_positions_mm(
    print_height_mm: float,
    config: PrintheadConfig,
    band_overlap: int = 0,
) -> List[float]:
    """Compute the absolute Y position in mm for every pass.

    Y = 0 is the bottom of the image. A head whose slots sit at different
    heights needs a lead-in *below* the image before its highest slot can reach
    the bottom rows, so the first positions can be negative; see
    https://paintress.dev/concepts/swaths-and-passes/

    ``band_overlap`` must match generate_print_passes: bands then advance by
    ``lines_per_band - band_overlap`` (R7 feathering). Both read the same
    schedule, so the two cannot drift apart."""
    print_height_lines = int(round((print_height_mm / 25.4) * config.dpi))
    line_spacing = config.line_spacing_mm
    return [head_line * line_spacing
            for head_line in config.geometry.pass_schedule(
                print_height_lines, band_overlap)]


def convert_positions_to_deltas(y_positions_mm: List[float]) -> List[float]:
    """Convert absolute Y positions to relative movements (deltas)."""
    if not y_positions_mm:
        return []

    deltas = [
        y_positions_mm[i + 1] - y_positions_mm[i]
        for i in range(len(y_positions_mm) - 1)
    ]
    deltas.append(0.0)
    return deltas


# ---------------------------------------------------------------------------
# Calibration targets
# ---------------------------------------------------------------------------

# Channel ramps printed by the wedge target, one per column. The neutral
# series drives C=M=Y together (K held at 0) and feeds gray-balance work (R2).
WEDGE_SERIES: Tuple[str, ...] = ("C", "M", "Y", "K", "NEUTRAL")


# Four corner fiducials of *distinct* sizes. Distinct sizes let the scan tool
# identify each corner by area ranking, so registration is invariant to how the
# sheet is placed on the scanner (rotation / flip), which matters because the
# print itself is vertically flipped on the bed (see generate_print_passes).
FIDUCIAL_CORNERS: Tuple[str, ...] = ("TL", "TR", "BL", "BR")
FIDUCIAL_SIZE_FACTORS = {"TL": 1.0, "TR": 0.82, "BL": 0.66, "BR": 0.50}


def add_corner_fiducials(
    k_plane: np.ndarray,
    width: int,
    height: int,
    margin_px: int,
    fiducial_px: int,
) -> List[Dict]:
    """Paint four distinct-size solid-K registration squares into the corner
    margins of the K plane and return their pixel bounds + corner label
    (shared by both calibration targets). ``fiducial_px`` is the largest
    (TL) square; the others are scaled down so each corner is identifiable."""
    fiducials: List[Dict] = []
    for corner in FIDUCIAL_CORNERS:
        fp = max(2, round(fiducial_px * FIDUCIAL_SIZE_FACTORS[corner]))
        inset = max(0, (margin_px - fp) // 2)
        fx = inset if corner in ("TL", "BL") else width - inset - fp
        fy = inset if corner in ("TL", "TR") else height - inset - fp
        k_plane[fy:fy + fp, fx:fx + fp] = 255
        fiducials.append({"corner": corner, "x0": fx, "y0": fy, "x1": fx + fp, "y1": fy + fp})
    return fiducials


def _fit_wedge_grid(
    steps: int,
    n_series: int,
    gap_mm: float,
    margin_mm: float,
    max_x_mm: float,
    max_y_mm: float,
) -> Tuple[int, int, float]:
    """Choose the panel count and patch size for a roughly-square target.

    The ``steps`` rows are split across ``wrap`` panels placed side by side,
    each panel a mini ``n_series``-column wedge. We pick the ``wrap`` that
    maximises the (square) patch size while fitting inside the ``max_x_mm`` x
    ``max_y_mm`` box: bigger patches scan more reliably, and maximising the
    patch under a square box naturally lands on a roughly-square layout.

    Returns ``(wrap, rows_per_panel, patch_mm)``.
    """
    usable_x = max_x_mm - 2 * margin_mm
    usable_y = max_y_mm - 2 * margin_mm
    if usable_x <= 0 or usable_y <= 0:
        raise ValueError("margins do not fit inside the target box")

    best = None  # (patch_mm, wrap, rows)
    for wrap in range(1, steps + 1):
        rows = math.ceil(steps / wrap)
        cols = wrap * n_series
        patch_x = (usable_x - (cols - 1) * gap_mm) / cols
        patch_y = (usable_y - (rows - 1) * gap_mm) / rows
        patch = min(patch_x, patch_y)
        if patch <= 0:
            continue
        if best is None or patch > best[0]:
            best = (patch, wrap, rows)

    if best is None:
        raise ValueError("target box too small for any wedge layout")
    patch_mm, wrap, rows = best
    return wrap, rows, patch_mm


def build_wedge_channels(
    dpi: int,
    steps: int = 33,
    gap_mm: float = 2.0,
    margin_mm: float = 10.0,
    fiducial_mm: float = 6.0,
    max_x_mm: float = 200.0,
    max_y_mm: float = 200.0,
    min_patch_mm: float = 4.0,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Synthesise the continuous-tone CMYK channels of a step-wedge target.

    The target is a roughly-square grid that fits inside a ``max_x_mm`` x
    ``max_y_mm`` box. The ``steps`` coverage levels are split across ``wrap``
    panels placed side by side; each panel is a mini wedge with one column per
    entry in ``WEDGE_SERIES`` (C, M, Y, K, and a neutral C=M=Y ramp) and rows
    of increasing commanded coverage. The patch size is chosen as large as the
    box allows (better for scanning). Four solid-K fiducial squares sit in the
    corner margins so the scan tool can register and de-skew the print.

    Returns the (C, M, Y, K) coverage arrays (uint8, 0..255) plus a layout
    descriptor giving every patch's series, coverage and pixel bounds and the
    fiducial positions: everything the scan tool needs to locate the
    patches without hard-coding the geometry.
    """
    if steps < 2:
        raise ValueError("steps must be >= 2")

    n_series = len(WEDGE_SERIES)
    wrap, rows_per_panel, patch_mm = _fit_wedge_grid(
        steps, n_series, gap_mm, margin_mm, max_x_mm, max_y_mm
    )
    if patch_mm < min_patch_mm:
        print(f"   WARNING: patch size {patch_mm:.1f} mm < {min_patch_mm:.1f} mm "
              f"(box {max_x_mm:.0f}x{max_y_mm:.0f} mm too small or too many steps)")

    px_per_mm = dpi / 25.4
    patch_px = max(1, round(patch_mm * px_per_mm))
    gap_px = max(0, round(gap_mm * px_per_mm))
    margin_px = max(1, round(margin_mm * px_per_mm))
    fiducial_px = max(1, round(fiducial_mm * px_per_mm))

    cols = wrap * n_series
    width = 2 * margin_px + cols * patch_px + (cols - 1) * gap_px
    height = 2 * margin_px + rows_per_panel * patch_px + (rows_per_panel - 1) * gap_px

    channels = {name: np.zeros((height, width), dtype=np.uint8) for name in ("C", "M", "Y", "K")}
    patches: List[Dict] = []

    for panel in range(wrap):
        for local_row in range(rows_per_panel):
            step_index = panel * rows_per_panel + local_row
            if step_index >= steps:
                break
            coverage = step_index / (steps - 1)
            value = int(round(coverage * 255.0))
            y0 = margin_px + local_row * (patch_px + gap_px)
            y1 = y0 + patch_px
            for series_index, series_name in enumerate(WEDGE_SERIES):
                col = panel * n_series + series_index
                x0 = margin_px + col * (patch_px + gap_px)
                x1 = x0 + patch_px

                if series_name == "NEUTRAL":
                    for ch in ("C", "M", "Y"):
                        channels[ch][y0:y1, x0:x1] = value
                else:
                    channels[series_name][y0:y1, x0:x1] = value

                patches.append({
                    "series": series_name,
                    "step": step_index,
                    "panel": panel,
                    "local_row": local_row,
                    "coverage": coverage,
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                })

    fiducials = add_corner_fiducials(channels["K"], width, height, margin_px, fiducial_px)

    layout = {
        "kind": "wedge",
        "dpi": dpi,
        "steps": steps,
        "series": list(WEDGE_SERIES),
        "wrap": wrap,
        "rows_per_panel": rows_per_panel,
        "cols": cols,
        "patch_mm": patch_mm,
        "image_width_px": width,
        "image_height_px": height,
        "patch_px": patch_px,
        "gap_px": gap_px,
        "margin_px": margin_px,
        "fiducial_px": fiducial_px,
        "patches": patches,
        "fiducials": fiducials,
    }
    return channels, layout


# Channels that carry data (LC/LM are unused today, so the nozzle check omits
# them). One diagonal staircase of dashes per channel, stacked down Y.
NOZZLE_CHECK_CHANNELS: Tuple[str, ...] = ("C", "M", "Y", "K", "LC", "LM")


def blit_text(plane: np.ndarray, text: str, cx: int, top: int, height_px: int,
              anchor: str = "mt", value: int = 255) -> None:
    """Render ``text`` into a uint8 ``plane`` (as ink=value) ~``height_px`` tall.

    Uses PIL's built-in bitmap font upscaled with nearest-neighbour, so the
    nozzle-check target can print its own index numbers, no external font.
    ``anchor`` places the box: 'mt' = top-centred at (cx, top), 'lt' =
    left-top at (cx, top).
    """
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    bbox = probe.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if tw <= 0 or th <= 0:
        return
    glyph = Image.new("L", (tw, th), 0)
    ImageDraw.Draw(glyph).text((-bbox[0], -bbox[1]), text, fill=255, font=font)
    scale = height_px / th
    glyph = glyph.resize((max(1, round(tw * scale)), max(1, round(th * scale))), Image.NEAREST)
    arr = np.asarray(glyph)
    h, w = arr.shape
    x0 = cx - w // 2 if anchor == "mt" else cx
    y0 = top
    H, W = plane.shape
    xs, xe = max(0, x0), min(W, x0 + w)
    ys, ye = max(0, y0), min(H, y0 + h)
    if xs >= xe or ys >= ye:
        return
    sub = arr[ys - y0:ye - y0, xs - x0:xe - x0]
    plane[ys:ye, xs:xe][sub > 127] = value


def build_nozzle_check_channels(
    dpi: int,
    dash_mm: float = 1.5,
    dx_mm: float = 1.8,
    margin_mm: float = 10.0,
    fiducial_mm: float = 6.0,
    channels: Optional[Tuple[str, ...]] = None,
    config: Optional[PrintheadConfig] = None,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Synthesise the channels of a per-nozzle check target, self-labelled so a
    human reads the failed nozzles by eye (no scan needed).

    Every nozzle fires one isolated dash in its ink; because the RIP maps image
    rows to nozzles, a dash on the ``passes_per_band`` rows a nozzle owns is
    printed entirely by that one nozzle. The dashes are staggered in X by
    ``dx_mm`` per nozzle so they form a diagonal comb; a dead nozzle is a gap.

    One comb per plumbed ink: each ink has its own band, so inks the eye cannot
    separate (LC/LM against C/M) are told apart by position. Only the nozzles a
    print actually fires are checked: a slot's idle nozzles cannot affect the
    output and have no index in the dead-nozzle records, so a gap there would be
    noise in the profile. Above the combs a printed **index ruler** carries
    a tick per nozzle and a number every 10; **guide lines** every 10 nozzles
    run the full height, and each comb is labelled with its ink at the left. To
    read: find the gap, follow the guide/column up to the ruler, read the
    number. Enter the failures with
    ``tools/record_dead_nozzles.py --dead "C:3,17;M:42"``.
    """
    cfg = config or PrintheadConfig(dpi=dpi)
    counts = slot_nozzle_counts(cfg)
    if channels is None:
        channels = tuple(ch for ch in cfg.channel_order if ch in counts)
    missing = [ch for ch in channels if ch not in counts]
    if missing:
        raise ValueError(
            f"nozzle check asked for {', '.join(missing)}, which head "
            f"{cfg.layout.name} has no slot plumbed with")
    # The ruler and the labels are drawn in one ink; keep it K where there is
    # one, so an existing c6n90 target is unchanged.
    ruler_ink = "K" if "K" in channels else channels[0]
    nozzle_count = max(counts[ch] for ch in channels)
    ppb = cfg.passes_per_band
    lpb = cfg.lines_per_band

    px_per_mm = dpi / 25.4
    dash_px = max(1, round(dash_mm * px_per_mm))
    dx_px = max(1, round(dx_mm * px_per_mm))
    margin_px = max(1, round(margin_mm * px_per_mm))
    fiducial_px = max(1, round(fiducial_mm * px_per_mm))

    number_h = max(6, round(4.5 * px_per_mm))     # ~4.5 mm ruler numbers
    label_h = max(8, round(9.0 * px_per_mm))       # ~9 mm channel labels
    tick10, tick5, tick1 = (round(5.0 * px_per_mm), round(3.5 * px_per_mm),
                            round(2.0 * px_per_mm))
    guide_w = max(2, round(0.3 * px_per_mm))
    strip_px = number_h + tick10 + round(3 * px_per_mm)   # ruler band at the top

    n_channels = len(channels)
    stair_w = (nozzle_count - 1) * dx_px + dash_px
    width = 2 * margin_px + stair_w
    height = strip_px + n_channels * lpb           # ruler strip + one band/channel

    planes = {name: np.zeros((height, width), dtype=np.uint8) for name in channels}
    nozzles: List[Dict] = []

    def dash_centre_x(n: int) -> int:
        return margin_px + n * dx_px + dash_px // 2

    # Channel combs. Reverse the band assignment (band = last..first) so that,
    # after the pass slicer's flip, the channels read C, M, Y, K top-to-bottom.
    for channel_index, channel_name in enumerate(channels):
        band_index = n_channels - 1 - channel_index
        band_start = band_index * lpb
        for n in range(counts[channel_name]):
            printed_y0 = band_start + n * ppb
            y1 = height - printed_y0
            y0 = height - (printed_y0 + ppb)
            x0 = margin_px + n * dx_px
            x1 = x0 + dash_px
            planes[channel_name][y0:y1, x0:x1] = 255
            nozzles.append({"channel": channel_name, "nozzle": n,
                            "x0": x0, "y0": y0, "x1": x1, "y1": y1})

    # Guide lines (below the ruler numbers, through the combs) + the top index
    # ruler (both in K).
    k = planes[ruler_ink]
    for n in range(0, nozzle_count + 1, 10):
        gx = dash_centre_x(min(n, nozzle_count - 1)) if n < nozzle_count else \
            margin_px + n * dx_px
        k[number_h:, max(0, gx - guide_w // 2):gx + guide_w // 2 + 1] = 255
    for n in range(nozzle_count):
        gx = dash_centre_x(n)
        t = tick10 if n % 10 == 0 else (tick5 if n % 5 == 0 else tick1)
        k[strip_px - t:strip_px, max(0, gx - 1):gx + 2] = 255
        if n % 10 == 0:
            blit_text(k, str(n), gx, 0, number_h, anchor="mt")

    # Channel labels down the left margin, at each comb's centre.
    for channel_index, channel_name in enumerate(channels):
        band_index = n_channels - 1 - channel_index
        centre_y = height - (band_index * lpb + lpb // 2)
        blit_text(k, channel_name, margin_px // 2, centre_y - label_h // 2, label_h, anchor="mt")

    fiducials = add_corner_fiducials(k, width, height, margin_px, fiducial_px)

    layout = {
        "kind": "nozzle_check",
        "dpi": dpi,
        "channels": list(channels),
        "nozzle_count": nozzle_count,
        # Per channel, since slots need not be the same length. Each entry is
        # the nozzles a print fires from that slot, which is what the comb
        # checks and what dead_nozzles indexes.
        "nozzle_counts": {ch: counts[ch] for ch in channels},
        "head": cfg.layout.name,
        "ruler_ink": ruler_ink,
        "passes_per_band": ppb,
        "image_width_px": width,
        "image_height_px": height,
        "dash_px": dash_px,
        "dx_px": dx_px,
        "margin_px": margin_px,
        "fiducial_px": fiducial_px,
        "strip_px": strip_px,
        "nozzles": nozzles,
        "fiducials": fiducials,
    }
    return planes, layout


# Column-distance (colour-to-colour registration) target.
#
# What is measured is a distance between *columns*, so there is one figure per
# column, not per ink: inks sharing a column (C/M/Y on c4n180) are at the same
# X by construction and nothing separates them. The reference is the leading
# column (the highest column index, which the encoder gives offset 0), so
# every residual reads directly as that column's distance behind it.
#
# On c6n90 this resolves to M as the reference and C, Y, K as the tests, which
# is what the target has always printed; c4n180 reduces to one measurement (K's
# column against C's). Only inked columns are swept: LC/LM carry no ink today.
COL_ALIGN_REF = "M"
COL_ALIGN_CHANNELS: Tuple[str, ...] = ("C", "Y", "K")


def resolve_col_align_channels(config: PrintheadConfig) -> Tuple[str, Tuple[str, ...]]:
    """Pick the reference ink and the inks to sweep, one per other column.

    The sandwich only cancels trigger jitter if reference and test print on the
    same sweeps, so every ink chosen here starts at the same height up its
    column as the reference does (``first_nozzle``). On a head where the
    columns are split in Y that rules out most candidates: on c4n180 the
    reference has to be C, not M, because C is the ink sharing K's band.
    """
    geometry = config.geometry
    inked = [(geometry.slot_for(ch), ch) for ch in geometry.imaging_channels]
    if len(inked) < 2:
        raise ValueError(
            f"col_align needs at least two inked columns; head "
            f"{geometry.layout.name} has {len(inked)} inked slot(s)")

    # The leading column is the highest index; among its inks prefer the one at
    # the bottom of the column, so the tests can line up with it.
    ref_column = max(slot.column for slot, _ in inked)
    candidates = sorted((s.first_nozzle, ch) for s, ch in inked
                        if s.column == ref_column)
    ref_offset, reference = candidates[0]

    # One test ink per other column, at the reference's height, columns
    # descending (which on c6n90 gives C, Y, K, the historical order).
    tests: Dict[int, str] = {}
    for slot, ch in inked:
        if slot.column == ref_column or slot.first_nozzle != ref_offset:
            continue
        tests.setdefault(slot.column, ch)
    if not tests:
        raise ValueError(
            f"col_align found no ink sharing the reference's band on head "
            f"{geometry.layout.name}: the columns cannot be compared without "
            f"the sweeps being common mode")
    return reference, tuple(tests[c] for c in sorted(tests, reverse=True))


def build_col_align_channels(
    dpi: int,
    span_mm: float = 0.8,
    cell_pitch_mm: float = 2.5,
    mark_mm: float = 0.5,
    repeats: int = 2,
    margin_mm: float = 10.0,
    fiducial_mm: float = 6.0,
    channels: Optional[Tuple[str, ...]] = None,
    config: Optional[PrintheadConfig] = None,
    reference: Optional[str] = None,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Synthesise the channels of a column-distance calibration target.

    The encoder registers the colour columns by delaying each channel
    ``ceil(distance_behind_M * dpi / 25.4)`` data columns (see
    PrintheadLayout in the encoder), so any error in the configured
    ``column_gap_mm`` / ``group_gap_mm`` prints every channel displaced in X
    by the same *residual* on every job. This target measures that residual
    per channel, through the production encode path (print it with the
    offsets you want to calibrate, i.e. the encoder defaults).

    Geometry. For each test channel, a **collinear sandwich**: a reference
    segment (M), the test segment, and a second reference segment stacked in
    Y at the same commanded X, repeated across a sweep of commanded offsets
    ``k`` = -span..+span px (one cell per k, labelled by the ruler strip at
    the top). If the configured gaps were exact, the k=0 cell would be a
    perfectly straight line; in general the straight cell reads ``k_best``
    and the residual is ``-k_best`` px. ``tools/col_align_gaps.py`` turns the
    per-channel readings into corrected ``column_gap_mm`` / ``group_gap_mm``.

    The sandwich is engineered so the comparison is immune to everything but
    column geometry: all three segments sit inside **one band** (so ref and
    test share the same sweeps: trigger jitter is common mode), each
    segment spans whole nozzles (all ``passes_per_band`` interleaved sweeps
    contribute equally to ref and test), and the test segment is vertically
    centred between the two reference segments (linear head yaw cancels in
    the two-ref average).

    A final **yaw band** prints one full-band vertical line per channel
    (reference included): its slope is the head yaw and its wiggle the
    column straightness, context for interpreting the residuals, not part
    of the fit.
    """
    cfg = config or PrintheadConfig(dpi=dpi)
    resolved_ref, resolved_tests = resolve_col_align_channels(cfg)
    if reference is None:
        reference = resolved_ref
    if channels is None:
        channels = resolved_tests
    nozzle_count = min(slot_nozzle_counts(cfg)[ch]
                       for ch in (reference,) + tuple(channels))
    ppb = cfg.passes_per_band
    lpb = cfg.lines_per_band

    px_per_mm = dpi / 25.4
    span_px = max(1, round(span_mm * px_per_mm))
    pitch_px = max(3, round(cell_pitch_mm * px_per_mm))
    mark_px = max(1, round(mark_mm * px_per_mm))
    margin_px = max(1, round(margin_mm * px_per_mm))
    fiducial_px = max(1, round(fiducial_mm * px_per_mm))
    if mark_px + 2 > pitch_px:
        raise ValueError("cell pitch too small for the mark width")
    if margin_px < span_px + mark_px:
        raise ValueError("margin too small for the commanded-offset span")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    # One sandwich per test channel per repeat band, stacked as equal
    # nozzle-range thirds of the band; segments span whole nozzles.
    third = nozzle_count // len(channels)
    gap_nz = 1
    ref_nz = max(3, (third * 3) // 10)
    test_nz = third - 2 * ref_nz - 2 * gap_nz
    if test_nz < 2:
        raise ValueError(f"too many test channels for {nozzle_count} nozzles")

    n_cells = 2 * span_px + 1
    width = 2 * margin_px + n_cells * pitch_px

    number_h = max(6, round(2.5 * px_per_mm))
    tick_big, tick_small = round(1.5 * px_per_mm), round(0.8 * px_per_mm)
    strip_px = max(margin_px, number_h + tick_big + round(1.0 * px_per_mm))

    # Machine band 0 (image bottom, printed first) is the yaw band; the
    # sandwich repeats sit above it, right under the ruler strip. Anchoring
    # the bands at the image bottom keeps them on the pass slicer's band
    # grid, which starts at machine line 0.
    total_bands = repeats + 1
    height = strip_px + total_bands * lpb

    def nz_rows(band: int, n0: int, n1: int) -> Tuple[int, int]:
        """Image-row span [y0, y1) of nozzles [n0, n1) of machine band."""
        lo = band * lpb + n0 * ppb
        hi = band * lpb + n1 * ppb
        return height - hi, height - lo

    def cell_x(i: int) -> int:
        return margin_px + pitch_px // 2 + i * pitch_px

    planes = {name: np.zeros((height, width), dtype=np.uint8)
              for name in cfg.geometry.imaging_channels}
    # Rulers, labels and fiducials all ride one plane; K where the head has it.
    ink_for_marks = "K" if "K" in planes else reference
    k_plane = planes[ink_for_marks]
    cells: List[Dict] = []

    for rep in range(repeats):
        band = 1 + rep
        for channel_index, channel_name in enumerate(channels):
            # Reverse the thirds so channels read top-to-bottom on paper.
            base = (len(channels) - 1 - channel_index) * third
            ref_a = nz_rows(band, base, base + ref_nz)
            test = nz_rows(band, base + ref_nz + gap_nz,
                           base + ref_nz + gap_nz + test_nz)
            ref_b = nz_rows(band, base + third - ref_nz, base + third)

            for i in range(n_cells):
                k = i - span_px
                rx0 = cell_x(i) - mark_px // 2
                tx0 = rx0 + k
                for (y0, y1) in (ref_a, ref_b):
                    planes[reference][y0:y1, rx0:rx0 + mark_px] = 255
                planes[channel_name][test[0]:test[1], tx0:tx0 + mark_px] = 255
                cells.append({
                    "channel": channel_name, "repeat": rep, "k": k,
                    "x_px": cell_x(i),
                    "test": {"x0": tx0, "y0": test[0], "x1": tx0 + mark_px, "y1": test[1]},
                    "refs": [
                        {"x0": rx0, "y0": ref_a[0], "x1": rx0 + mark_px, "y1": ref_a[1]},
                        {"x0": rx0, "y0": ref_b[0], "x1": rx0 + mark_px, "y1": ref_b[1]},
                    ],
                })

            # Channel letter in the left margin, centred on the test segment.
            label_h = max(8, round(2.5 * px_per_mm))
            blit_text(k_plane, channel_name, margin_px // 2,
                      (test[0] + test[1]) // 2 - label_h // 2, label_h, anchor="mt")

    # Yaw band: one full-band line per channel (reference first), K-labelled.
    yaw_lines: List[Dict] = []
    yaw_channels = (reference,) + tuple(channels)
    yaw_pitch_px = min(round(8.0 * px_per_mm),
                       (width - 2 * margin_px) // (len(yaw_channels) + 1))
    y0, y1 = nz_rows(0, 1, nozzle_count - 1)
    for j, channel_name in enumerate(yaw_channels):
        lx0 = margin_px + (j + 1) * yaw_pitch_px - mark_px // 2
        planes[channel_name][y0:y1, lx0:lx0 + mark_px] = 255
        blit_text(k_plane, channel_name, lx0 + mark_px + max(2, round(0.5 * px_per_mm)),
                  y0 + round(1.0 * px_per_mm), max(8, round(2.5 * px_per_mm)), anchor="lt")
        yaw_lines.append({"channel": channel_name,
                          "x0": lx0, "y0": y0, "x1": lx0 + mark_px, "y1": y1})

    # Ruler strip: a tick per cell, a k label every few cells.
    label_step = 5 if span_px >= 10 else (2 if span_px >= 4 else 1)
    for i in range(n_cells):
        k = i - span_px
        cx = cell_x(i)
        t = tick_big if k % label_step == 0 else tick_small
        k_plane[strip_px - t:strip_px, max(0, cx - 1):cx + 1] = 255
        if k % label_step == 0:
            blit_text(k_plane, str(k) if k <= 0 else f"+{k}", cx, 0, number_h, anchor="mt")

    fiducials = add_corner_fiducials(k_plane, width, height, margin_px, fiducial_px)

    layout = {
        "kind": "col_align",
        "dpi": dpi,
        "head": cfg.layout.name,
        "ref": reference,
        "channels": list(channels),
        "span_px": span_px,
        "pitch_px": pitch_px,
        "mark_px": mark_px,
        "repeats": repeats,
        "passes_per_band": ppb,
        "lines_per_band": lpb,
        "image_width_px": width,
        "image_height_px": height,
        "margin_px": margin_px,
        "strip_px": strip_px,
        "fiducial_px": fiducial_px,
        "cells": cells,
        "yaw_lines": yaw_lines,
        "fiducials": fiducials,
    }
    return planes, layout


def render_target_preview(
    cmyk_channels: Dict[str, np.ndarray],
    layout: Dict,
    max_dim_px: int = 1400,
) -> "Image.Image":
    """Render a labelled RGB preview of a calibration target (continuous tone).

    Naive subtractive CMYK -> RGB on a light-grey paper, with the series names
    across the top and coverage labels down the side: a human sanity check
    before committing a print. Not used for printing. The output is scaled to
    fit ``max_dim_px`` on its longest side so the preview stays readable
    regardless of the target DPI.
    """
    from PIL import ImageDraw

    c = cmyk_channels["C"].astype(np.float32) / 255.0
    m = cmyk_channels["M"].astype(np.float32) / 255.0
    y = cmyk_channels["Y"].astype(np.float32) / 255.0
    k = cmyk_channels["K"].astype(np.float32) / 255.0
    # Light inks (nozzle-check target) contribute to their sibling's band at
    # ~half strength, so their combs show up in the preview. Absent for the
    # CMYK-only targets (wedge, col_align).
    if "LC" in cmyk_channels:
        c = np.clip(c + 0.5 * cmyk_channels["LC"].astype(np.float32) / 255.0, 0, 1)
    if "LM" in cmyk_channels:
        m = np.clip(m + 0.5 * cmyk_channels["LM"].astype(np.float32) / 255.0, 0, 1)
    r = (255.0 * (1 - c) * (1 - k)).astype(np.uint8)
    g = (255.0 * (1 - m) * (1 - k)).astype(np.uint8)
    b = (255.0 * (1 - y) * (1 - k)).astype(np.uint8)
    rgb = np.dstack([r, g, b])
    paper = (rgb == 255).all(axis=2)
    rgb[paper] = (245, 245, 245)

    img = Image.fromarray(rgb, mode="RGB")
    scale = min(3.0, max_dim_px / max(img.width, img.height))
    # BOX (area averaging) when shrinking: NEAREST drops whole rows, so
    # equal-height marks (e.g. nozzle-check dashes) come out 2 px vs 3 px.
    img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                     Image.BOX if scale < 1.0 else Image.NEAREST)
    draw = ImageDraw.Draw(img)

    series = layout.get("series", [])
    for p in layout.get("patches", []):
        # Series letter above the top row of every panel column.
        if p.get("local_row", p.get("step")) == 0:
            cx = (p["x0"] + p["x1"]) / 2 * scale
            draw.text((cx - 6, 2), p["series"], fill=(0, 0, 0))
        # Coverage % inside each panel's first (cyan) column.
        if series and p["series"] == series[0]:
            draw.text((p["x0"] * scale + 2, p["y0"] * scale + 2),
                      f"{int(round(p['coverage'] * 100))}", fill=(0, 0, 0))

    # (Nozzle-check channel labels and the index ruler are printed into the K
    # plane by build_nozzle_check_channels, so they already show above.)

    return img


def process_target_for_printing(
    output_path: str,
    kind: str = "wedge",
    dpi: int = 630,
    dither_method: str = "floyd_steinberg",
    blue_noise_size: int = 64,
    steps: int = 33,
    gap_mm: float = 2.0,
    max_x_mm: float = 200.0,
    max_y_mm: float = 200.0,
    dash_mm: float = 1.5,
    dx_mm: float = 1.8,
    span_mm: float = 0.8,
    cell_pitch_mm: float = 2.5,
    mark_mm: float = 0.5,
    repeats: int = 2,
    preview_path: Optional[str] = None,
    ink_map: Optional[Sequence[str]] = None,
    head: str = DEFAULT_HEAD,
) -> Dict:
    """Generate a calibration target payload directly (no input image).

    ``kind`` is ``wedge`` (R1 linearisation step wedge), ``nozzle_check``
    (N1 per-nozzle staircase) or ``col_align`` (colour-column distance
    calibration). The target is halftoned with the *production*
    dither so it prints through the same pipeline a real job does. The colour
    stages (dot gain, ink limiting, DPI compensation) are deliberately
    bypassed: the target measures the raw device, and the nozzle check must
    reflect exactly which nozzles fire.
    """
    printhead_config = PrintheadConfig(dpi=dpi, head=head, ink_map=ink_map)
    ink_map = printhead_config.ink_map

    dither_config = DitherConfig(method=dither_method, blue_noise_size=blue_noise_size)

    print(f"Generating calibration target: {kind}")
    print(f"   DPI: {dpi} ({printhead_config.passes_per_band} passes/band)")

    if kind == "wedge":
        print(f"   Steps: {steps}, dither: {dither_method}, box {max_x_mm:.0f}x{max_y_mm:.0f} mm")
        cmyk_channels, layout = build_wedge_channels(
            dpi=dpi, steps=steps, gap_mm=gap_mm, max_x_mm=max_x_mm, max_y_mm=max_y_mm
        )
        marks = "patches"
    elif kind == "nozzle_check":
        print(f"   Dither: {dither_method}, dash {dash_mm:.1f} mm, step {dx_mm:.1f} mm")
        cmyk_channels, layout = build_nozzle_check_channels(
            dpi=dpi, dash_mm=dash_mm, dx_mm=dx_mm, config=printhead_config
        )
        marks = "nozzles"
    elif kind == "col_align":
        print(f"   Dither: {dither_method}, sweep ±{span_mm:.2f} mm, "
              f"pitch {cell_pitch_mm:.1f} mm, {repeats} repeat band(s)")
        cmyk_channels, layout = build_col_align_channels(
            dpi=dpi, span_mm=span_mm, cell_pitch_mm=cell_pitch_mm,
            mark_mm=mark_mm, repeats=repeats, config=printhead_config,
        )
        marks = "cells"
    else:
        raise ValueError(f"Unknown target kind: {kind}")

    image_height_px, image_width_px = cmyk_channels["C"].shape
    print_width_mm = (image_width_px / dpi) * 25.4
    print_height_mm = (image_height_px / dpi) * 25.4
    if kind == "wedge":
        print(f"   Layout: {layout['wrap']} panels, {layout['cols']}x{layout['rows_per_panel']} grid, "
              f"patch {layout['patch_mm']:.1f} mm")
    elif kind == "nozzle_check":
        print(f"   Layout: {len(layout['channels'])} channels x {layout['nozzle_count']} nozzles")
    else:
        print(f"   Layout: {len(layout['channels'])} channels vs {layout['ref']}, "
              f"{2 * layout['span_px'] + 1} cells of ±{layout['span_px']} px, "
              f"{layout['repeats']} repeat band(s) + yaw band")
    print(f"   Target: {image_width_px}x{image_height_px} px "
          f"(X {print_width_mm:.1f} mm x Y {print_height_mm:.1f} mm; X is the head sweep)")

    # Halftone (production dither), then slice into passes; no colour stages.
    # The nozzle check exercises every plumbed slot, so it halftones and slices
    # every ink the head carries (LC/LM included on c6n90); the other targets
    # are CMYK and get any remaining planes zero-filled to the payload shape.
    halftone_bitmaps = convert_to_halftone_bitmaps(cmyk_channels, dither_config)
    if kind == "nozzle_check":
        pass_channel_order = tuple(layout["channels"])
        print_passes = generate_print_passes(
            halftone_bitmaps, printhead_config, input_channel_order=pass_channel_order)
        print_passes_all = print_passes
    else:
        print_passes = generate_print_passes(halftone_bitmaps, printhead_config)
        print_passes_all = add_light_ink_channels(
            print_passes,
            light_channel_count=len(printhead_config.geometry.blank_channels))
    y_positions = compute_pass_y_positions_mm(print_height_mm, printhead_config)

    # The pass slicer flips the image onto the bed specifically so the print
    # comes out NON-mirrored (the mirrored-prints fix), i.e. the printed sheet
    # matches the image orientation. So metadata.target is the image-space
    # layout as-is: a scan of the sheet (roughly upright) lines up with it.
    print_job = {
        "metadata": {
            "dpi": dpi,
            "image_width_px": image_width_px,
            "image_height_px": image_height_px,
            "print_width_mm": print_width_mm,
            "print_height_mm": print_height_mm,
            "total_passes": len(print_passes_all),
            "passes_per_band": printhead_config.passes_per_band,
            "nozzle_count": printhead_config.nozzle_count,
            "channel_order": printhead_config.channel_order,
            # The full head geometry + plumbing: per-slot column, height up the
            # column and nozzle count. On a head whose slots sit at different
            # heights this is what says which rows each ink reached, so a reader
            # cannot reconstruct the print without it.
            "head_layout": printhead_config.geometry.to_metadata(),
            # Carry the head plumbing so the encoder routes each ink comb to the
            # slot it will actually print from; otherwise the nozzle check
            # would land on the wrong slots and the dead-nozzle capture (which
            # keys by slot through the same map) would be silently mis-attributed.
            "ink_map": list(printhead_config.ink_map),
            "processing": {
                "dither_method": dither_method,
                "icc_profile": None,
                "calibration": None,
            },
            "target": layout,
        },
        "passes": {
            "y_positions_mm": y_positions,
            "y_deltas_mm": convert_positions_to_deltas(y_positions),
            "data": print_passes_all,
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bin_path = rip_payload.save_rip(
        output_path,
        metadata=print_job["metadata"],
        y_positions_mm=print_job["passes"]["y_positions_mm"],
        y_deltas_mm=print_job["passes"]["y_deltas_mm"],
        passes=print_job["passes"]["data"],
    )
    print(f"   Saved {Path(output_path).name} + {bin_path.name} "
          f"({len(layout[marks])} {marks}, {len(layout['fiducials'])} fiducials)")

    if preview_path:
        preview = render_target_preview(cmyk_channels, layout)
        Path(preview_path).parent.mkdir(parents=True, exist_ok=True)
        preview.save(preview_path)
        print(f"   Preview: {Path(preview_path).name} ({preview.width}x{preview.height} px)")

    return print_job


# ---------------------------------------------------------------------------
# Main Processing Pipeline
# ---------------------------------------------------------------------------

def process_image_for_printing(
    input_path: str,
    output_path: str,
    dpi: int = 630,
    print_width_mm: Optional[float] = None,
    print_height_mm: Optional[float] = None,
    icc_profile_path: Optional[str] = None,
    dither_method: str = "floyd_steinberg",
    blue_noise_size: int = 64,
    calibration: Optional[CalibrationProfile] = None,
    nozzle_comp: str = "none",
    head: str = DEFAULT_HEAD,
) -> Dict:
    """Process an image through the full RIP pipeline.

    Colour behaviour is entirely driven by ``calibration`` (a
    CalibrationProfile); when omitted the shipped seed profile is used.

    Steps:
        1. Load image and convert to CMYK (UCR/GCR fallback)
        2. Apply dot-gain compensation
        3. Apply ink limits / colour balance
        4. Dither each channel to a 1-bit bitmap
        5. Generate print passes
        6. Add light-ink (LC/LM) channels
        7. Compute Y positions
        8. Serialise to JSON header + .bin sidecar
    """
    import time

    layout = get_layout(head)
    if calibration is None:
        calibration = CalibrationProfile.default(layout)

    printhead_config = PrintheadConfig(dpi=dpi, head=head,
                                       ink_map=calibration.ink_map)
    dither_config = DitherConfig(method=dither_method, blue_noise_size=blue_noise_size)
    dot_gain_config = calibration.dot_gain
    ink_limit_config = calibration.ink_limit

    # A LUT for this DPI replaces dot gain + DPI ink power (it subsumes both):
    # a measured one if available, else a modelled Murray-Davies one from the
    # profile's drop diameter; failing both, the heuristic colour stack.
    measured_luts = calibration.measured_luts(dpi)
    lut_source = "measured"
    if measured_luts is None and calibration.drop_diameter_um:
        measured_luts = murray_davies_luts(dpi, calibration.drop_diameter_um)
        lut_source = "modelled"
    colour_mode = f"linearised ({lut_source})" if measured_luts else "heuristic"

    geometry = printhead_config.geometry
    overlap = max(0, calibration.band_overlap)
    print(f"Configuration:")
    print(f"   Head: {layout.name} ({geometry.describe()})")
    print(f"   DPI: {dpi} ({geometry.interleave_passes} passes/band, "
          f"band step {geometry.band_step_mm:.2f} mm"
          + (f", lead-in {geometry.lead_in_mm(overlap):.2f} mm below the image"
             if geometry.lead_in_bands(overlap) else "") + ")")
    print(f"   Dithering: {dither_method}")
    print(f"   Calibration profile: {calibration.media}")
    print(f"   Colour: {colour_mode}")
    if measured_luts:
        print(f"   Total ink cap: {ink_limit_config.max_total_ink:.2f} (post-linearisation)")
    else:
        print(
            f"   Dot Gain: {'enabled' if dot_gain_config.enabled else 'disabled'} "
            f"(C={dot_gain_config.cyan_gain:.2f}, M={dot_gain_config.magenta_gain:.2f}, "
            f"Y={dot_gain_config.yellow_gain:.2f}, K={dot_gain_config.black_gain:.2f})"
        )
        scales = ink_limit_config.channel_scales
        print(
            f"   Ink limits: {'enabled' if ink_limit_config.enabled else 'disabled'} "
            f"(scale={ink_limit_config.global_scale:.2f}, C={scales.get('C', 1.0):.2f}, "
            f"M={scales.get('M', 1.0):.2f}, Y={scales.get('Y', 1.0):.2f}, "
            f"K={scales.get('K', 1.0):.2f}, total={ink_limit_config.max_total_ink:.2f})"
        )
        if ink_limit_config.compensate_dpi:
            print(f"   DPI ink compensation: reference {ink_limit_config.reference_dpi} DPI")
    if icc_profile_path:
        print(f"   ICC Profile: {icc_profile_path}")

    # Steps 1-2: load (16-bit aware), resample in linear light, RGB->CMYK in
    # float (R6 photographic front-end). The image is actually resized here, so
    # metadata reflects the real raster.
    print(f"\nLoading image (16-bit float, linear-light resample)...")
    t0 = time.perf_counter()
    cmyk_channels, (image_width_px, image_height_px), (native_width_px, native_height_px), resampled = \
        prepare_cmyk_channels(input_path, dpi, print_width_mm, print_height_mm,
                              icc_profile_path, calibration.black_generation, calibration.black_start)
    t1 = time.perf_counter()
    print(f"   {native_width_px}x{native_height_px} px ({t1-t0:.3f}s)")
    if resampled:
        if abs((image_width_px / image_height_px) - (native_width_px / native_height_px)) > 1e-3:
            print("   WARNING: requested size changes the aspect ratio (image will be distorted)")
        print(f"   Resampled (linear light): {native_width_px}x{native_height_px} "
              f"-> {image_width_px}x{image_height_px} px")

    # Physical size always derives from the actual raster, so metadata can't
    # disagree with the data.
    print_width_mm = (image_width_px / dpi) * 25.4
    print_height_mm = (image_height_px / dpi) * 25.4

    print(f"   Dimensions: {print_width_mm:.1f} x {print_height_mm:.1f} mm")

    # Steps 2-3: colour correction. A measured LUT linearises each channel and
    # subsumes dot gain + the DPI ink power, so we apply ONLY the LUT (then the
    # total-ink cap, per the plan's "TAC after linearisation"); the per-channel
    # scales are held back for measured gray balance (R2). Without a LUT we fall
    # back to the heuristic dot-gain + ink-limit stack.
    if measured_luts:
        print(f"\nApplying measured linearisation LUTs...")
        t0 = time.perf_counter()
        cmyk_channels = apply_linearization_luts(cmyk_channels, measured_luts)
        if ink_limit_config.max_total_ink > 0:
            cmyk_channels = apply_total_ink_limit(cmyk_channels, ink_limit_config.max_total_ink)
        t1 = time.perf_counter()
        print(f"   Done ({t1-t0:.3f}s)")
    else:
        # Step 2: Dot-gain compensation
        if dot_gain_config.enabled:
            print(f"\nApplying dot gain compensation...")
            t0 = time.perf_counter()
            cmyk_channels = apply_dot_gain_compensation(cmyk_channels, dot_gain_config)
            t1 = time.perf_counter()
            print(f"   Done ({t1-t0:.3f}s)")

        # Step 3: practical ink limiting / colour balance
        if ink_limit_config.enabled:
            print(f"\nApplying ink limits and colour balance...")
            t0 = time.perf_counter()
            cmyk_channels = apply_ink_limits(cmyk_channels, ink_limit_config, dpi)
            t1 = time.perf_counter()
            print(f"   Done ({t1-t0:.3f}s)")

    # Dead-nozzle compensation (N2), driven by the profile's dead_nozzles.
    dead_masks: Dict[str, np.ndarray] = {}
    if nozzle_comp != "none" and calibration.dead_nozzles:
        dead_masks = dead_nozzle_row_masks(
            calibration.dead_nozzles, image_height_px, printhead_config)
        n_dead = sum(len(v) for v in calibration.dead_nozzles.values())
        print(f"\nDead-nozzle compensation: {nozzle_comp} ({n_dead} dead nozzles)")
        if nozzle_comp == "reroute" and dither_method != "floyd_steinberg":
            print("   WARNING: reroute needs floyd_steinberg dithering; "
                  "dead rows will be blanked without redistribution.")
    elif nozzle_comp != "none":
        print(f"\nDead-nozzle compensation: {nozzle_comp}, but the profile lists no dead nozzles.")

    # Step 4: Halftoning (reroute masks dead rows during error diffusion, N2a)
    print(f"\nApplying dithering ({dither_method})...")
    t0 = time.perf_counter()
    reroute_masks = dead_masks if nozzle_comp == "reroute" else None
    if reroute_masks:
        # Confine reroute to a local smear: drop the ink of dead rows with no
        # healthy row within one nozzle pitch, so a large cluster's ink can't
        # cascade far down (it would otherwise dump well away from the gap).
        reach = printhead_config.passes_per_band
        cmyk_channels = {
            ch: (limit_reroute_ink(data, reroute_masks[ch], reach)
                 if ch in reroute_masks else data)
            for ch, data in cmyk_channels.items()
        }
    halftone_bitmaps = convert_to_halftone_bitmaps(cmyk_channels, dither_config, reroute_masks)
    t1 = time.perf_counter()
    print(f"   Done ({t1-t0:.3f}s)")

    # Step 5: Generate passes
    band_overlap = max(0, calibration.band_overlap)
    print(f"\nGenerating print passes"
          + (f" (band feathering {band_overlap} rows)" if band_overlap else "") + "...")
    t0 = time.perf_counter()
    print_passes = generate_print_passes(halftone_bitmaps, printhead_config, band_overlap=band_overlap)
    y_positions = compute_pass_y_positions_mm(print_height_mm, printhead_config, band_overlap)

    # Retouch passes (N2b): reprint dead rows with a healthy neighbour.
    if nozzle_comp == "retouch" and dead_masks:
        retouch_passes, retouch_y = generate_retouch_passes(
            halftone_bitmaps, calibration.dead_nozzles, printhead_config, y_positions,
            band_overlap=band_overlap)
        print_passes.extend(retouch_passes)
        y_positions = list(y_positions) + retouch_y
        _, _, skipped = classify_retouch_donors(
            calibration.dead_nozzles, slot_nozzle_counts(printhead_config))
        n_skip = sum(len(v) for v in skipped.values())
        print(f"   + {len(retouch_passes)} retouch passes appended"
              + (f" ({n_skip} nozzles in clusters had no healthy neighbour)" if n_skip else ""))
    t1 = time.perf_counter()
    print(f"   {len(print_passes)} passes generated ({t1-t0:.3f}s)")

    # Shingling for non-absorbent media (R8): split each swath into N
    # column-interleaved sweeps so wet drops settle between depositions.
    shingle_passes = max(1, calibration.shingle_passes)
    if shingle_passes > 1:
        before = len(print_passes)
        print_passes, y_positions = apply_shingling(print_passes, y_positions, shingle_passes)
        print(f"   Shingling x{shingle_passes}: {before} -> {len(print_passes)} sweeps")

    # Step 6: Add LC/LM channels
    t0 = time.perf_counter()
    print_passes_all = add_light_ink_channels(
        print_passes, light_channel_count=len(printhead_config.geometry.blank_channels))
    t1 = time.perf_counter()
    blank = printhead_config.geometry.blank_channels
    if blank:
        print(f"   {'/'.join(blank)} channels added ({t1-t0:.3f}s)")

    # Step 7: Y positions already computed above (+ retouch / shingling).

    # Step 8: Build output structure
    print_job = {
        "metadata": {
            "dpi": dpi,
            "image_width_px": image_width_px,
            "image_height_px": image_height_px,
            "print_width_mm": print_width_mm,
            "print_height_mm": print_height_mm,
            "total_passes": len(print_passes_all),
            "passes_per_band": printhead_config.passes_per_band,
            "nozzle_count": printhead_config.nozzle_count,
            "channel_order": printhead_config.channel_order,
            # The full head geometry + plumbing: per-slot column, height up the
            # column and nozzle count. On a head whose slots sit at different
            # heights this is what says which rows each ink reached, so a reader
            # cannot reconstruct the print without it.
            "head_layout": printhead_config.geometry.to_metadata(band_overlap),
            # Head plumbing carried downstream so the encoder routes inks to the
            # same slots the dead-nozzle compensation assumed (one source). This
            # is the plumbing in force: the profile's when it gave one, else
            # the head's reference wiring.
            "ink_map": list(printhead_config.ink_map),
            "processing": {
                "dither_method": dither_method,
                "icc_profile": icc_profile_path,
                "colour_mode": "linearised" if measured_luts else "heuristic",
                "lut_source": lut_source if measured_luts else None,
                "nozzle_comp": nozzle_comp if dead_masks else "none",
                "shingle_passes": shingle_passes,
                "band_overlap": band_overlap,
                "calibration": calibration.to_dict(),
            },
        },
        "passes": {
            "y_positions_mm": y_positions,
            "y_deltas_mm": convert_positions_to_deltas(y_positions),
            "data": print_passes_all,
        },
    }

    # Step 9: Save (JSON header + packed binary sidecar)
    print(f"\nSaving output...")
    t0 = time.perf_counter()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bin_path = rip_payload.save_rip(
        output_path,
        metadata=print_job["metadata"],
        y_positions_mm=print_job["passes"]["y_positions_mm"],
        y_deltas_mm=print_job["passes"]["y_deltas_mm"],
        passes=print_job["passes"]["data"],
    )
    t1 = time.perf_counter()
    print(f"   Saved {Path(output_path).name} + {bin_path.name} ({t1-t0:.3f}s)")

    return print_job


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def list_supported_dpis(head: str = DEFAULT_HEAD, max_dpi: int = 1800) -> None:
    """Print the DPIs a head supports: the multiples of its nozzle pitch."""
    layout = get_layout(head)
    pitch = layout.nozzle_pitch_npi
    print(f"Supported DPIs for head {layout.name} (multiples of {pitch}):")
    for dpi in range(pitch, max_dpi + 1, pitch):
        geometry = HeadGeometry.build(layout, dpi)
        print(f"  {dpi:4d} DPI - {geometry.interleave_passes} passes/band, "
              f"band step {geometry.band_step_mm:.2f} mm")


def resolve_calibration(args, layout=None) -> CalibrationProfile:
    """Load a calibration profile and apply CLI flag overrides.

    The profile is the source of truth. Any flag left at its sentinel
    (None / False) does not touch the profile; a supplied flag overrides
    the corresponding field.
    """
    if args.calibration:
        profile = CalibrationProfile.load(args.calibration, layout)
    else:
        profile = CalibrationProfile.default(layout)

    # Dot gain: a single --dot-gain value fans out to the channels the same
    # way the seed profile does (C=M=v, Y=0.75v, K=1.25v).
    if args.dot_gain is not None:
        v = args.dot_gain
        profile.dot_gain.cyan_gain = v
        profile.dot_gain.magenta_gain = v
        profile.dot_gain.yellow_gain = v * 0.75
        profile.dot_gain.black_gain = v * 1.25
    if args.no_dot_gain:
        profile.dot_gain.enabled = False

    # Ink limiting / channel balance overrides.
    if args.no_ink_limit:
        profile.ink_limit.enabled = False
    if args.ink_scale is not None:
        profile.ink_limit.global_scale = args.ink_scale
    if args.cyan_scale is not None:
        profile.ink_limit.channel_scales["C"] = args.cyan_scale
        profile.ink_limit.channel_scales["LC"] = args.cyan_scale
    if args.magenta_scale is not None:
        profile.ink_limit.channel_scales["M"] = args.magenta_scale
        profile.ink_limit.channel_scales["LM"] = args.magenta_scale
    if args.yellow_scale is not None:
        profile.ink_limit.channel_scales["Y"] = args.yellow_scale
    if args.black_scale is not None:
        profile.ink_limit.channel_scales["K"] = args.black_scale
    if args.max_total_ink is not None:
        profile.ink_limit.max_total_ink = args.max_total_ink
    if args.reference_dpi is not None:
        profile.ink_limit.reference_dpi = args.reference_dpi
    if args.no_dpi_compensation:
        profile.ink_limit.compensate_dpi = False
    if args.black_generation is not None:
        profile.black_generation = args.black_generation
    if args.black_start is not None:
        profile.black_start = args.black_start
    if args.shingle is not None:
        profile.shingle_passes = args.shingle
    if args.band_overlap is not None:
        profile.band_overlap = args.band_overlap

    return profile


def main():
    parser = argparse.ArgumentParser(
        description="RIP - Process an image for inkjet printing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic (uses the seed calibration profile, Floyd-Steinberg, 630 DPI)
  python rip.py image.png -o output.json

  # A specific media calibration
  python rip.py image.png -o output.json --calibration profiles/glossy.json

  # Blue Noise dithering
  python rip.py image.png -o output.json --dither blue_noise

  # With ICC profile
  python rip.py image.png -o output.json --icc printer_profile.icc

  # Override one field of the loaded profile
  python rip.py image.png -o output.json --magenta-scale 0.9
        """
    )

    parser.add_argument("input", nargs="?", help="Input image (PNG, JPG, etc.)")
    parser.add_argument("-o", "--output", help="Output header file (.json; a .bin sidecar is written alongside)")

    parser.add_argument(
        "--head",
        choices=sorted(HEAD_LAYOUTS),
        default=DEFAULT_HEAD,
        help=f"Printhead layout. Default: {DEFAULT_HEAD}"
    )

    parser.add_argument(
        "--ink-map",
        type=str,
        default=None,
        help="Head plumbing: the ink in each slot, comma-separated, in the "
             "order you read the head facing it (left to right by column, "
             f"bottom to top within a column). Use {EMPTY_SLOT!r} for an "
             "unplumbed slot. Overrides the calibration profile; defaults to "
             "the head's reference wiring."
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="DPI (must be a multiple of the head's nozzle pitch: 90 for "
             "c6n90, 180 for c4n180). Default: the head's own (630 / 720)."
    )

    parser.add_argument(
        "--width",
        type=float,
        default=None,
        help="Target print width in mm; resamples the image to this size at the "
             "chosen DPI. If only one of width/height is given, the other keeps "
             "the aspect ratio. Default: the image's native size."
    )
    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help="Target print height in mm; resamples the image (see --width)."
    )

    parser.add_argument(
        "--dither",
        choices=["floyd_steinberg", "blue_noise", "ordered"],
        default="floyd_steinberg",
        help="Dithering method. Default: floyd_steinberg"
    )
    parser.add_argument(
        "--blue-noise-size",
        type=int,
        default=64,
        choices=[32, 64, 128, 256],
        help="Blue noise texture size. Default: 64"
    )

    parser.add_argument(
        "--icc",
        type=str,
        default=None,
        help="Path to a CMYK ICC profile"
    )

    # Calibration profile is the source of truth; the flags below are
    # pointwise overrides (each defaults to a sentinel meaning "do not
    # touch the profile").
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to a calibration profile JSON (see rip/profiles/default.json). "
             "Default: the shipped seed profile."
    )

    parser.add_argument(
        "--dot-gain",
        type=float,
        default=None,
        help="Override dot gain (0.0 to 1.0). Higher = lighter midtones. "
             "Fans out to C=M=v, Y=0.75v, K=1.25v."
    )
    parser.add_argument(
        "--no-dot-gain",
        action="store_true",
        help="Disable dot gain compensation"
    )

    parser.add_argument("--ink-scale", type=float, default=None,
                        help="Override the global ink multiplier applied before dithering.")
    parser.add_argument("--cyan-scale", type=float, default=None,
                        help="Override the cyan channel multiplier. Lower if prints are green/cyan.")
    parser.add_argument("--magenta-scale", type=float, default=None,
                        help="Override the magenta channel multiplier. Raise if prints are green.")
    parser.add_argument("--yellow-scale", type=float, default=None,
                        help="Override the yellow channel multiplier. Lower if prints are green/yellow.")
    parser.add_argument("--black-scale", type=float, default=None,
                        help="Override the black channel multiplier.")
    parser.add_argument("--max-total-ink", type=float, default=None,
                        help="Override the maximum total CMYK ink per pixel, as a fraction.")
    parser.add_argument("--black-generation", type=float, default=None,
                        help="Override UCR/GCR black generation (0..1) for the non-ICC path.")
    parser.add_argument("--black-start", type=float, default=None,
                        help="Override the GCR start point (0..1): neutral below this stays CMY "
                             "(no K in highlights), above it K ramps in.")
    parser.add_argument("--shingle", type=int, default=None,
                        help="Shingling for non-absorbent media (R8): print each swath in N "
                             "column-interleaved sweeps so wet drops settle. 1=off, 2=even/odd. "
                             "Costs N x sweeps.")
    parser.add_argument("--band-overlap", type=int, default=None,
                        help="Band-boundary feathering (R7): overlap consecutive bands by this many "
                             "rows and split the seam stochastically to hide Y-advance error. 0=off.")
    parser.add_argument("--reference-dpi", type=int, default=None,
                        help="Override the DPI baseline for ink compensation.")
    parser.add_argument("--no-dpi-compensation", action="store_true",
                        help="Disable automatic ink reduction when DPI is above reference DPI")
    parser.add_argument("--no-ink-limit", action="store_true",
                        help="Disable practical ink limiting / channel scaling")

    parser.add_argument(
        "--nozzle-comp",
        choices=["none", "reroute", "retouch"],
        default="none",
        help="Compensate the profile's dead_nozzles (from the nozzle check, N2). "
             "'reroute' diffuses their ink to neighbours during Floyd-Steinberg "
             "(free, partial); 'retouch' adds passes so the nearest healthy "
             "neighbour (n-1 or n+1) reprints the dead rows (exact, ~2x sweeps; "
             "cluster interiors with no healthy neighbour are left uncompensated). "
             "Default: none."
    )

    # Calibration-target generation (no input image; see --target).
    parser.add_argument(
        "--target",
        choices=["wedge", "nozzle_check", "col_align"],
        default=None,
        help="Generate a calibration target payload instead of processing an "
             "image. 'wedge' is a per-channel step wedge for R1 linearisation; "
             "'nozzle_check' fires every nozzle once for the N1 nozzle check; "
             "'col_align' measures the colour-column distances (encode it with "
             "the column/group gaps you want to calibrate, then read the print "
             "by eye and pass the readings to tools/col_align_gaps.py). One "
             "target per DPI."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=33,
        help="Number of coverage steps per channel in the wedge target. Default: 33"
    )
    parser.add_argument(
        "--dash-mm",
        type=float,
        default=1.5,
        help="Nozzle-check dash length in mm (X). Default: 1.5"
    )
    parser.add_argument(
        "--dx-mm",
        type=float,
        default=1.8,
        help="Nozzle-check X stagger per nozzle in mm. Default: 1.8"
    )
    parser.add_argument(
        "--max-x-mm",
        type=float,
        default=200.0,
        help="Max wedge width in mm (head-sweep axis). The target fits this box "
             "and picks the largest patch that fits. Default: 200"
    )
    parser.add_argument(
        "--max-y-mm",
        type=float,
        default=200.0,
        help="Max wedge height in mm (advance axis). Default: 200"
    )
    parser.add_argument(
        "--gap-mm",
        type=float,
        default=2.0,
        help="Gap between wedge patches in mm. Default: 2.0"
    )
    parser.add_argument(
        "--span-mm",
        type=float,
        default=0.8,
        help="col_align: commanded-offset sweep, ±this many mm in whole pixels "
             "at the target DPI. Default: 0.8"
    )
    parser.add_argument(
        "--cell-pitch-mm",
        type=float,
        default=2.5,
        help="col_align: X pitch between sweep cells in mm. Default: 2.5"
    )
    parser.add_argument(
        "--mark-mm",
        type=float,
        default=0.5,
        help="col_align: width of the printed marks in mm. Default: 0.5"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="col_align: number of repeat measurement bands. Default: 2"
    )
    parser.add_argument(
        "--preview",
        type=str,
        default=None,
        help="Also write a labelled RGB preview PNG of the target to this path."
    )

    parser.add_argument(
        "--list-dpi",
        action="store_true",
        help="List supported DPIs and exit"
    )

    args = parser.parse_args()

    if args.list_dpi:
        list_supported_dpis(args.head)
        return 0

    layout = get_layout(args.head)
    if args.dpi is None:
        args.dpi = layout.default_dpi
    if not layout.supports_dpi(args.dpi):
        print(f"Error: DPI must be a multiple of {layout.nozzle_pitch_npi} for "
              f"head {layout.name} (got: {args.dpi})")
        print("   Use --list-dpi to see valid values")
        return 1

    cli_ink_map = None
    if args.ink_map:
        cli_ink_map = tuple(part.strip() for part in args.ink_map.split(","))
        try:
            layout.validate_ink_map(cli_ink_map)
        except ValueError as e:
            print(f"Error: --ink-map: {e}")
            return 1

    # Calibration-target mode: synthesise the target, no input image needed.
    if args.target:
        if not args.output:
            parser.error("'-o/--output' argument is required")
        if args.target == "wedge" and args.steps < 2:
            print(f"Error: --steps must be >= 2 (got: {args.steps})")
            return 1
        # Carry the profile's head plumbing into the target so the encoder
        # routes each ink to the right slot (the nozzle check must fire from the
        # slots the head is actually plumbed with).
        try:
            target_profile = (CalibrationProfile.load(args.calibration, layout)
                              if args.calibration
                              else CalibrationProfile.default(layout))
        except (OSError, ValueError) as e:
            print(f"Error: could not load calibration profile: {e}")
            return 1
        try:
            process_target_for_printing(
                output_path=args.output,
                kind=args.target,
                dpi=args.dpi,
                dither_method=args.dither,
                blue_noise_size=args.blue_noise_size,
                steps=args.steps,
                gap_mm=args.gap_mm,
                max_x_mm=args.max_x_mm,
                max_y_mm=args.max_y_mm,
                dash_mm=args.dash_mm,
                dx_mm=args.dx_mm,
                span_mm=args.span_mm,
                cell_pitch_mm=args.cell_pitch_mm,
                mark_mm=args.mark_mm,
                repeats=args.repeats,
                preview_path=args.preview,
                ink_map=cli_ink_map or target_profile.ink_map,
                head=args.head,
            )
        except ValueError as e:
            print(f"Error: {e}")
            return 1
        return 0

    if not args.input:
        parser.error("'input' argument is required (use --list-dpi to list supported DPIs)")

    if not args.output:
        parser.error("'-o/--output' argument is required")

    if args.dot_gain is not None and not 0.0 <= args.dot_gain <= 1.0:
        print(f"Error: --dot-gain must be between 0.0 and 1.0 (got: {args.dot_gain})")
        return 1

    for name in ("ink_scale", "cyan_scale", "magenta_scale", "yellow_scale",
                 "black_scale", "max_total_ink", "black_generation"):
        value = getattr(args, name)
        if value is not None and value < 0.0:
            print(f"Error: --{name.replace('_', '-')} must be >= 0.0 (got: {value})")
            return 1

    try:
        calibration = resolve_calibration(args, layout)
        if cli_ink_map is not None:
            calibration.ink_map = cli_ink_map
    except (OSError, ValueError) as e:
        print(f"Error: could not load calibration profile: {e}")
        return 1

    process_image_for_printing(
        input_path=args.input,
        output_path=args.output,
        dpi=args.dpi,
        print_width_mm=args.width,
        print_height_mm=args.height,
        icc_profile_path=args.icc,
        dither_method=args.dither,
        blue_noise_size=args.blue_noise_size,
        calibration=calibration,
        nozzle_comp=args.nozzle_comp,
        head=args.head,
    )

    return 0


if __name__ == "__main__":
    exit(main())
