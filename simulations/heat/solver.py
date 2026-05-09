"""
1-D heat equation solver via explicit finite differences.

    u_t = alpha * u_xx,  x in [0, L],  t in [0, T]
    u(0, t) = u(L, t) = 0   (Dirichlet BCs)
    u(x, 0) = initial_condition(x)

Stability requires r = alpha * dt / dx^2 <= 0.5.
"""

import numpy as np


def solve_heat(
    alpha: float = 0.1,
    nx: int = 100,
    nt: int = 300,
    L: float = 1.0,
    T: float = 0.5,
    ic: str = "pulse",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (x, t_array, u_history) where u_history has shape (nt, nx).

    ic: 'pulse'   — rectangular pulse in the middle third
        'gaussian' — Gaussian centered at x=0.5
        'sine'    — single sine mode
    """
    dx = L / (nx - 1)
    dt = T / nt
    r = alpha * dt / dx ** 2
    if r > 0.5:
        raise ValueError(
            f"Unstable: r = {r:.3f} > 0.5. Reduce dt or increase nx."
        )

    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)

    u = _initial_condition(x, ic)
    u[0] = u[-1] = 0.0

    history = np.empty((nt, nx))
    history[0] = u.copy()

    for i in range(1, nt):
        laplacian = np.roll(u, -1) - 2 * u + np.roll(u, 1)
        laplacian[0] = laplacian[-1] = 0.0
        u = u + r * laplacian
        u[0] = u[-1] = 0.0
        history[i] = u

    return x, t, history


def _initial_condition(x: np.ndarray, ic: str) -> np.ndarray:
    n = len(x)
    u = np.zeros(n)
    if ic == "pulse":
        u[n // 3 : 2 * n // 3] = 1.0
    elif ic == "gaussian":
        u = np.exp(-((x - 0.5) ** 2) / (2 * 0.05 ** 2))
    elif ic == "sine":
        u = np.sin(np.pi * x)
    else:
        raise ValueError(f"Unknown ic: {ic!r}")
    return u
