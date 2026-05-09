"""
1-D wave equation solver via explicit finite differences (Leapfrog scheme).

    u_tt = c^2 * u_xx,  x in [0, L],  t in [0, T]
    u(0, t) = u(L, t) = 0   (fixed ends)
    u(x, 0) = initial displacement, u_t(x, 0) = 0 (released from rest)

Stability requires CFL number nu = c * dt / dx <= 1.
"""

import numpy as np


def solve_wave(
    c: float = 1.0,
    nx: int = 200,
    nt: int = 400,
    L: float = 1.0,
    T: float = 1.0,
    ic: str = "pulse",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (x, t_array, u_history) where u_history has shape (nt, nx).

    ic: 'pulse'   — rectangular pulse in the middle
        'gaussian' — narrow Gaussian
        'pluck'   — triangular pluck at x=0.3
    """
    dx = L / (nx - 1)
    dt = T / nt
    nu = c * dt / dx
    if nu > 1.0:
        raise ValueError(
            f"Unstable: CFL = {nu:.3f} > 1. Reduce dt or c."
        )

    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)

    u_prev = _initial_condition(x, ic)
    u_curr = u_prev.copy()
    u_prev[0] = u_prev[-1] = 0.0
    u_curr[0] = u_curr[-1] = 0.0

    history = np.empty((nt, nx))
    history[0] = u_prev.copy()

    # first step: use Taylor expansion (zero initial velocity)
    laplacian = np.roll(u_curr, -1) - 2 * u_curr + np.roll(u_curr, 1)
    laplacian[0] = laplacian[-1] = 0.0
    u_next = u_curr + 0.5 * nu ** 2 * laplacian
    u_next[0] = u_next[-1] = 0.0
    history[1] = u_next.copy()

    for i in range(2, nt):
        laplacian = np.roll(u_curr, -1) - 2 * u_curr + np.roll(u_curr, 1)
        laplacian[0] = laplacian[-1] = 0.0
        u_new = 2 * u_curr - u_prev + nu ** 2 * laplacian
        u_new[0] = u_new[-1] = 0.0
        u_prev, u_curr = u_curr, u_new
        history[i] = u_curr

    return x, t, history


def _initial_condition(x: np.ndarray, ic: str) -> np.ndarray:
    n = len(x)
    u = np.zeros(n)
    if ic == "pulse":
        u[n // 3 : 2 * n // 3] = 0.5
    elif ic == "gaussian":
        u = np.exp(-((x - 0.5) ** 2) / (2 * 0.04 ** 2))
    elif ic == "pluck":
        idx = int(0.3 * n)
        u[:idx] = x[:idx] / x[idx]
        u[idx:] = (x[-1] - x[idx:]) / (x[-1] - x[idx])
    else:
        raise ValueError(f"Unknown ic: {ic!r}")
    return u
