"""
Generate pre-computed heat equation data for the browser simulation.

Writes JSON files to docs/chapters/04-heat-equation/data/
Run from the repo root:
    python simulations/heat/generate_data.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from solver import solve_heat

OUT_DIR = Path(__file__).parent.parent.parent / "docs/chapters/04-heat-equation/data"

ALPHA_VALUES = np.round(np.arange(0.01, 1.01, 0.01), 2)
NX         = 60
L          = 1.0
T_SCALE    = 0.05   # T = T_SCALE / alpha  →  comparable diffusion progress across alphas
CFL        = 0.45   # stability: r = alpha * dt / dx^2 <= 0.5; use 0.45 for safety
KEEP_EVERY = 3      # thin frames to reduce file size


def generate(alpha: float) -> None:
    dx = L / (NX - 1)
    T  = T_SCALE / alpha
    dt = CFL * dx ** 2 / alpha
    nt = max(50, int(T / dt) + 1)
    x, t, history = solve_heat(alpha=alpha, nx=NX, nt=nt, L=L, T=T)

    frames = [
        {"t_idx": i, "u": history[i].round(6).tolist()}
        for i in range(0, nt, KEEP_EVERY)
    ]

    payload = {
        "params": {"alpha": alpha, "nx": NX, "nt": nt, "L": L, "T": round(T, 6)},
        "x": x.round(6).tolist(),
        "t": t[::KEEP_EVERY].round(6).tolist(),
        "frames": frames,
    }

    fname = OUT_DIR / f"heat_alpha_{alpha:.2f}.json"
    with open(fname, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  wrote {fname.name}  ({len(frames)} frames, nt={nt})")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(ALPHA_VALUES)} heat equation datasets → {OUT_DIR}")
    for alpha in ALPHA_VALUES:
        generate(float(alpha))
    print("Done.")
