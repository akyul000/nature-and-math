"""
Generate diffusion data for chapter 07 (parabolic PDEs).
The diffusion equation is identical to the heat equation — same solver, different label.

Writes to docs/chapters/07-parabolic/data/
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "heat"))
from solver import solve_heat

OUT_DIR = Path(__file__).parent.parent.parent / "docs/chapters/07-parabolic/data"

D_VALUES   = np.round(np.arange(0.01, 1.01, 0.01), 2)
NX         = 60
L          = 1.0
T_SCALE    = 0.05
CFL        = 0.45
KEEP_EVERY = 3


def generate(D: float) -> None:
    dx = L / (NX - 1)
    T  = T_SCALE / D
    dt = CFL * dx ** 2 / D
    nt = max(50, int(T / dt) + 1)
    x, t, history = solve_heat(alpha=D, nx=NX, nt=nt, L=L, T=T)

    frames = [
        {"t_idx": i, "u": history[i].round(6).tolist()}
        for i in range(0, nt, KEEP_EVERY)
    ]

    payload = {
        "params": {"D": D, "nx": NX, "nt": nt, "L": L, "T": round(T, 6)},
        "x": x.round(6).tolist(),
        "t": t[::KEEP_EVERY].round(6).tolist(),
        "frames": frames,
    }

    fname = OUT_DIR / f"diffusion_D_{D:.2f}.json"
    with open(fname, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  wrote {fname.name}  ({len(frames)} frames)")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(D_VALUES)} diffusion datasets → {OUT_DIR}")
    for D in D_VALUES:
        generate(float(D))
    print("Done.")
