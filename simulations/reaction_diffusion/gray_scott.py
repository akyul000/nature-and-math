"""
Gray-Scott reaction-diffusion solver.

    ∂u/∂t = Du ∇²u − u·v² + F(1−u)
    ∂v/∂t = Dv ∇²v + u·v² − (F+k)·v

Explicit Euler, periodic boundary conditions, unit grid spacing.

Stability: r = Du * dt / dx² = Du * 1.0 / 1² = Du = 0.2 ≤ 0.5  ✓
"""

import numpy as np


def _laplacian(a: np.ndarray) -> np.ndarray:
    """Five-point Laplacian with periodic BCs (dx = 1)."""
    return (
        np.roll(a, 1,  axis=0) + np.roll(a, -1, axis=0) +
        np.roll(a, 1,  axis=1) + np.roll(a, -1, axis=1) -
        4 * a
    )


def solve_gray_scott(
    F: float,
    k: float,
    Du: float = 0.2,
    Dv: float = 0.1,
    dt: float = 1.0,
    n_steps: int = 8000,
    nx: int = 80,
    save_at: tuple[int, ...] = (0, 500, 1000, 3000, 8000),
    seed: int = 42,
) -> list[tuple[int, np.ndarray]]:
    """
    Run the Gray-Scott model and return snapshots of the u-field.

    Returns:
        List of (step, u_2d) tuples at each step in save_at.
        u_2d has shape (nx, nx), values in [0, 1].
    """
    rng = np.random.default_rng(seed)

    u = np.ones((nx, nx), dtype=np.float64)
    v = np.zeros((nx, nx), dtype=np.float64)

    # Seed: noisy square in the centre
    r = nx // 8
    cx, cy = nx // 2, nx // 2
    u[cx-r:cx+r, cy-r:cy+r] = 0.5 + rng.uniform(-0.05, 0.05, (2*r, 2*r))
    v[cx-r:cx+r, cy-r:cy+r] = 0.25 + rng.uniform(-0.05, 0.05, (2*r, 2*r))

    snapshots: list[tuple[int, np.ndarray]] = []
    save_set = set(save_at)

    for step in range(n_steps + 1):
        if step in save_set:
            snapshots.append((step, u.copy()))

        if step == n_steps:
            break

        uv2 = u * v * v
        u += dt * (Du * _laplacian(u) - uv2 + F * (1.0 - u))
        v += dt * (Dv * _laplacian(v) + uv2 - (F + k) * v)

        # Clamp to [0, 1] to prevent numerical blow-up
        np.clip(u, 0.0, 1.0, out=u)
        np.clip(v, 0.0, 1.0, out=v)

    return snapshots
