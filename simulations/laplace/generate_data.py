"""
Generate pre-computed Laplace equation data for the browser simulation.

Writes JSON files to docs/chapters/06-elliptic/data/
Run from the repo root:
    python simulations/laplace/generate_data.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from solver import solve_laplace

OUT_DIR = Path(__file__).parent.parent.parent / "docs/chapters/06-elliptic/data"

BC_TOP_VALUES = np.round(np.arange(0.0, 1.1, 0.1), 1)
N = 50


def generate(bc_top: float) -> None:
    x, y, u = solve_laplace(bc_top=bc_top, n=N)

    # Laplace is steady-state, no frames needed — just the 2D grid
    # Store as a single "frame" for consistency with sim-viewer.js
    payload = {
        "params": {"bc_top": bc_top, "n": N},
        "x": x.round(6).tolist(),
        "y": y.round(6).tolist(),
        # sim-viewer uses 'frames' list; heatmap mode reads frames[0].u as a 2D array
        "frames": [
            {"t_idx": 0, "u": u.round(6).tolist()}
        ],
    }

    fname = OUT_DIR / f"laplace_bc_top_{bc_top:.1f}.json"
    with open(fname, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  wrote {fname.name}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(BC_TOP_VALUES)} Laplace datasets → {OUT_DIR}")
    for bc_top in BC_TOP_VALUES:
        generate(float(bc_top))
    print("Done.")
