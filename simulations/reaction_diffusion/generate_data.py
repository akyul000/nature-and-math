"""
Batch data generator for Chapter 02: Nature's Patterns.

Generates 6 Gray-Scott preset JSON files for the Turing pattern simulator.
"""

from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gray_scott import solve_gray_scott

OUT = Path(__file__).parent.parent.parent / "docs/chapters/02-nature-patterns/data"
OUT.mkdir(parents=True, exist_ok=True)

SAVE_AT = (0, 500, 1000, 3000, 8000)

# Presets from Pearson (1993) parameter space — each produces a distinct pattern type
PRESETS = [
    {"name": "spots",     "F": 0.035,  "k": 0.065,  "desc": "Dark spots on light background"},
    {"name": "stripes",   "F": 0.060,  "k": 0.064,  "desc": "Parallel stripe pattern"},
    {"name": "labyrinth", "F": 0.040,  "k": 0.059,  "desc": "Winding maze-like structure"},
    {"name": "worms",     "F": 0.050,  "k": 0.063,  "desc": "Moving worm-like features"},
    {"name": "coral",     "F": 0.0545, "k": 0.0630, "desc": "Coral-like branching spots"},
    {"name": "mitosis",   "F": 0.028,  "k": 0.053,  "desc": "Self-replicating spots"},
]

for preset in PRESETS:
    name = preset["name"]
    F, k = preset["F"], preset["k"]
    print(f"Computing {name} (F={F}, k={k}) …", end=" ", flush=True)

    snapshots = solve_gray_scott(
        F=F, k=k,
        Du=0.2, Dv=0.1,
        dt=1.0,
        n_steps=max(SAVE_AT),
        nx=80,
        save_at=SAVE_AT,
        seed=42,
    )

    payload = {
        "preset": name,
        "desc":   preset["desc"],
        "params": {"F": F, "k": k, "Du": 0.2, "Dv": 0.1},
        "nx":     80,
        "steps":  list(SAVE_AT),
        "u": [snap.round(4).flatten().tolist() for (_step, snap) in snapshots],
    }

    fname = f"gray_scott_{name}.json"
    (OUT / fname).write_text(json.dumps(payload, separators=(",", ":")))
    size_kb = (OUT / fname).stat().st_size // 1024
    print(f"→ {fname}  ({size_kb} KB)")

print(f"\nDone — {len(PRESETS)} files in {OUT}")
