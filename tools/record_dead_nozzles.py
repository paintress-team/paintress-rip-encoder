# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Record dead nozzles, read by eye off a printed nozzle check, in a profile.

Print ``rip.py --target nozzle_check``: every nozzle fires one dash under a
numbered ruler. Purge first; a dash still missing after that is a dead nozzle.
Follow the guide line from the gap up to the ruler, read its number, and pass
the list here:

    python tools/record_dead_nozzles.py --profile rip/profiles/mymedia.json \\
        --dead "Y:10,11,50;K:30"

The nozzles are named per ink but stored against the physical slot each ink
is plumbed into (the profile's ``ink_map``), so the record belongs to the head
and survives a re-plumbing. The RIP reads it back for ``--nozzle-comp``.

Reading is manual on purpose: the dashes are about 0.3 mm tall, and a person
reading a numbered chart is more reliable than detecting them in a scan.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "rip"))
sys.path.insert(0, str(REPO / "encoder"))
import rip  # noqa: E402  (dead_channels_to_slots, EMPTY_SLOT: one translation source)
from head_profiles import head  # noqa: E402

SLOT_INKS = head("c6n90").slot_inks   # reference plumbing default


def parse_dead(spec: str) -> Dict[str, List[int]]:
    """Parse a dead-nozzle spec like 'C:3,17;M:42' -> {'C': [3, 17], 'M': [42]}."""
    out: Dict[str, List[int]] = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        channel, _, idxs = part.partition(":")
        out[channel.strip()] = sorted(int(i) for i in idxs.split(",") if i.strip())
    return out


def update_profile(profile_path: str, out_path: str, dead: Dict[str, List[int]]) -> None:
    """Write ``dead_nozzles`` into a calibration profile, keyed by slot.

    A profile that carries no ink_map is stamped with the head's reference
    plumbing. An ink the map leaves unplumbed (no slot) is dropped: there is no
    physical column to attribute its dashes to.
    """
    data = json.loads(Path(profile_path).read_text())
    ink_map = data.get("ink_map")
    if ink_map is None:
        ink_map = list(SLOT_INKS)
        data["ink_map"] = ink_map

    data["dead_nozzles"] = rip.dead_channels_to_slots(
        {ch: v for ch, v in dead.items() if v}, ink_map)
    Path(out_path).write_text(json.dumps(data, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record dead nozzles read off a printed nozzle check.")
    parser.add_argument("--dead", required=True,
                        help="Dead nozzles per ink, e.g. 'C:3,17;M:42'. "
                             "Pass an empty string to clear the record.")
    parser.add_argument("--profile", required=True, help="Calibration profile JSON to update.")
    parser.add_argument("--out", default=None, help="Output profile path. Default: overwrite --profile.")
    args = parser.parse_args()

    dead = parse_dead(args.dead)
    total = sum(len(v) for v in dead.values())
    out = args.out or args.profile
    update_profile(args.profile, out, dead)
    print(f"Recorded {total} dead nozzle(s) across {len(dead)} ink(s) in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
