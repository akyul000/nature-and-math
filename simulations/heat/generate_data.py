"""
Batch data generator for Chapter 2: Building Intuition.

Generates 9 JSON files (ic × alpha) with u, dudx, d2udx2 histories
for the 1-D temperature bar explorer simulation.
"""

from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from solver import solve_heat

OUT = Path(__file__).parent.parent.parent / "docs/chapters/02-intuition/data"
OUT.mkdir(parents=True, exist_ok=True)

N_SNAPSHOTS = 100
NX = 100
L, T = 1.0, 0.5

# (ic, alpha, nt) — nt chosen to satisfy stability r = alpha*dt/dx^2 <= 0.5
COMBOS = [
    ("pulse",    0.01, 200),
    ("pulse",    0.05, 600),
    ("pulse",    0.1,  1100),
    ("gaussian", 0.01, 200),
    ("gaussian", 0.05, 600),
    ("gaussian", 0.1,  1100),
    ("sine",     0.01, 200),
    ("sine",     0.05, 600),
    ("sine",     0.1,  1100),
]

for ic, alpha, nt in COMBOS:
    x, t, u_hist = solve_heat(alpha=alpha, nx=NX, nt=nt, L=L, T=T, ic=ic)

    # Subsample time axis to N_SNAPSHOTS frames
    idx = np.linspace(0, len(t) - 1, N_SNAPSHOTS, dtype=int)
    u_sub = u_hist[idx]   # (N_SNAPSHOTS, NX)
    t_sub = t[idx]        # (N_SNAPSHOTS,)

    # Compute spatial derivatives using central differences (same formula taught in Ch2)
    dx = x[1] - x[0]
    dudx   = np.gradient(u_sub,  dx, axis=1)   # first derivative
    d2udx2 = np.gradient(dudx,   dx, axis=1)   # second derivative (Laplacian)

    payload = {
        "params": {"ic": ic, "alpha": alpha, "nx": NX, "T": T},
        "x":              x.round(4).tolist(),
        "t":              t_sub.round(4).tolist(),
        "u_history":      u_sub.round(4).tolist(),
        "dudx_history":   dudx.round(4).tolist(),
        "d2udx2_history": d2udx2.round(4).tolist(),
    }

    fname = f"heat_ic_{ic}_alpha_{alpha}.json"
    (OUT / fname).write_text(json.dumps(payload, separators=(",", ":")))
    size_kb = (OUT / fname).stat().st_size // 1024
    print(f"wrote {fname}  ({size_kb} KB)")

print(f"\nDone — {len(COMBOS)} files in {OUT}")
