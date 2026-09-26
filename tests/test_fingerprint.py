# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pin the two identities a job carries, and keep them apart.

A job says two different things, to two different authorities:

* the **frame** fingerprint: the line size, clocks, edges and packing the
  firmware shifts out. The firmware reports it at IDENTIFY and the daemon
  refuses a job that disagrees. Every head sharing the frame shares it, which is
  what lets one firmware build serve them all.
* the **head** fingerprint: which printhead the job was packed for. Checked
  host-side against the machine's configuration, never against the firmware:
  the board has no way of knowing which head is bolted on.

Both are generated once, in paintress-protocol, from ``profiles/*.yaml``. This
pins the shipped values: if either changes, the test fails and flags what has to
be re-done (re-flash for the frame, re-RIP for a head).

Runs under pytest or as a plain script.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "encoder"))
sys.path.insert(0, str(REPO / "rip"))  # encoder imports rip_payload

import encoder as enc  # noqa: E402
import head_profiles  # noqa: E402

# Must equal paintress-firmware config/head_config.gen.h FRAME_HASH.
FRAME_FINGERPRINT = 0x108865B9

# One per head, from paintress-protocol profiles/*.yaml.
HEAD_FINGERPRINTS = {
    "c6n90": 0x9C244CD5,
    "c4n180": 0x93255450,
}

NOZZLE_COUNTS = {"c6n90": 90, "c4n180": 60}


def test_frame_fingerprint_is_pinned():
    assert head_profiles.FRAME_FINGERPRINT == FRAME_FINGERPRINT, (
        f"FRAME_FINGERPRINT 0x{head_profiles.FRAME_FINGERPRINT:08X} != pinned "
        f"0x{FRAME_FINGERPRINT:08X}. If the wire frame changed on purpose, "
        f"re-flash the firmware and re-encode jobs, then update this pin."
    )


def test_every_head_shares_the_frame():
    """The premise of a universal firmware: the heads differ only in things the
    device never reads."""
    for name in HEAD_FINGERPRINTS:
        assert name in head_profiles.HEADS, f"{name} is not a generated head"
    # There is one frame constant, not one per head; the generated module
    # could not express a per-head frame even if a profile tried to.
    assert isinstance(head_profiles.FRAME_FINGERPRINT, int)
    assert head_profiles.BYTES_PER_LINE == 147
    assert head_profiles.CLOCK_COUNT == 196
    assert head_profiles.BITS_PER_CLOCK == 6


def test_head_fingerprints_are_pinned_and_distinct():
    for name, pinned in HEAD_FINGERPRINTS.items():
        profile = head_profiles.head(name)
        assert profile.head_fingerprint == pinned, (
            f"{name} head fingerprint 0x{profile.head_fingerprint:08X} != "
            f"pinned 0x{pinned:08X}. If the head changed on purpose, re-RIP "
            f"and re-encode jobs for it, then update this pin."
        )
        assert profile.nozzle_count == NOZZLE_COUNTS[name]
    values = [p.head_fingerprint for p in head_profiles.HEADS.values()]
    assert len(set(values)) == len(values), (
        "two heads share a head fingerprint: that check is the only thing "
        "standing between a job and the wrong printhead"
    )


def test_a_head_fingerprint_is_never_a_frame_fingerprint():
    """They answer different questions; confusing them would make the daemon
    check the wrong authority."""
    for profile in head_profiles.HEADS.values():
        assert profile.head_fingerprint != head_profiles.FRAME_FINGERPRINT


def test_encoder_stamps_the_generated_values():
    # The encoder stamps the generated frame fingerprint, not a local recompute.
    assert enc.FRAME_FINGERPRINT == FRAME_FINGERPRINT
    assert enc.PACKED_LINE_BYTES == head_profiles.BYTES_PER_LINE


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failures else 0)
