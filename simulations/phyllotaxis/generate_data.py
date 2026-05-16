"""
generate_data.py

Runs phyllotaxis training from multiple starting angles and saves:

  1. phyllotaxis_training_{start_deg}.json  — convergence trace per run
  2. phyllotaxis_spiral_{label}.json        — seed (x, y) at key angles

Output: output/ (relative to this script).
When the golden-ratio chapter is created, update OUT to point to
../../docs/chapters/0X-golden-ratio/data/ instead.

Usage:
    cd simulations/phyllotaxis
    python generate_data.py
"""

import json
import math
from pathlib import Path

import jax.numpy as jnp

from phyllotaxis import (
    PhyllotaxisModel,
    seed_positions,
    train,
    GOLDEN_ANGLE_DEG,
)

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# ── Training runs ─────────────────────────────────────────────────────────────

START_ANGLES = [90.0, 120.0, 45.0]

for start_deg in START_ANGLES:
    print(f"Training from {start_deg}°…")
    model, history = train(initial_alpha_deg=start_deg, n_steps=4000, lr=3e-4, N=100)
    final_deg = float(model.alpha) * 180.0 / jnp.pi % 360.0
    print(f"  Converged to {final_deg:.3f}°  (golden ≈ {GOLDEN_ANGLE_DEG:.3f}°)")

    payload = {
        "start_deg": start_deg,
        "golden_angle_deg": round(GOLDEN_ANGLE_DEG, 4),
        "final_alpha_deg": round(final_deg, 4),
        **history,
    }
    out_path = OUT / f"phyllotaxis_training_{int(start_deg)}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  → {out_path}\n")

# ── Spiral snapshots at fixed angles ─────────────────────────────────────────

SNAPSHOT_ANGLES = {
    "90":     90.0,
    "120":    120.0,
    "180":    180.0,
    "golden": GOLDEN_ANGLE_DEG,
}

N_SPIRAL = 200

for label, angle_deg in SNAPSHOT_ANGLES.items():
    alpha_rad = jnp.array(angle_deg * math.pi / 180.0)
    snap = PhyllotaxisModel(alpha=alpha_rad)
    x, y = seed_positions(snap.alpha, N=N_SPIRAL)

    payload = {
        "alpha_deg": round(angle_deg, 4),
        "label": label,
        "N": N_SPIRAL,
        "x": [round(float(v), 4) for v in x],
        "y": [round(float(v), 4) for v in y],
    }
    out_path = OUT / f"phyllotaxis_spiral_{label}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved spiral snapshot ({label}): {out_path}")

print("\nDone.")
