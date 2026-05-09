"""
Generate pre-computed wave equation data for the browser simulation.

Writes JSON files to docs/chapters/08-hyperbolic/data/
Run from the repo root:
    python simulations/wave/generate_data.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from solver import solve_wave

OUT_DIR = Path(__file__).parent.parent.parent / "docs/chapters/08-hyperbolic/data"

C_VALUES  = np.round(np.arange(0.1, 2.1, 0.1), 1)
NX        = 200
NT        = 400
KEEP_EVERY = 4


def generate(c: float) -> None:
    x, t, history = solve_wave(c=c, nx=NX, nt=NT)

    frames = [
        {"t_idx": i, "u": history[i].round(6).tolist()}
        for i in range(0, NT, KEEP_EVERY)
    ]

    payload = {
        "params": {"c": c, "nx": NX, "nt": NT, "L": 1.0},
        "x": x.round(6).tolist(),
        "t": t[::KEEP_EVERY].round(6).tolist(),
        "frames": frames,
    }

    fname = OUT_DIR / f"wave_c_{c:.1f}.json"
    with open(fname, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  wrote {fname.name}  ({len(frames)} frames)")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(C_VALUES)} wave equation datasets → {OUT_DIR}")
    for c in C_VALUES:
        generate(float(c))
    print("Done.")
