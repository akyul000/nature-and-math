"""
2-D Laplace equation solver via Gauss-Seidel iteration.

    ∇²u = 0,  (x,y) in [0,1]²

Boundary conditions (Dirichlet):
    u(x, 1)  = bc_top      (top edge)
    u(x, 0)  = 0           (bottom)
    u(0, y)  = 0           (left)
    u(1, y)  = 0           (right)
"""

import numpy as np


def solve_laplace(
    bc_top: float = 1.0,
    n: int = 50,
    max_iter: int = 5000,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (x, y, u) where u has shape (n, n).
    x and y are 1-D coordinate arrays.
    """
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    u = np.zeros((n, n))

    u[-1, :] = bc_top   # top row
    u[0, :]  = 0.0      # bottom
    u[:, 0]  = 0.0      # left
    u[:, -1] = 0.0      # right

    for _ in range(max_iter):
        u_old = u.copy()
        u[1:-1, 1:-1] = 0.25 * (
            u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]
        )
        # re-enforce BCs
        u[-1, :] = bc_top
        u[0, :]  = 0.0
        u[:, 0]  = 0.0
        u[:, -1] = 0.0

        if np.max(np.abs(u - u_old)) < tol:
            break

    return x, y, u
