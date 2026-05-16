"""
phyllotaxis.py

An Equinox model with a single learnable divergence angle α.
Training minimizes a seed crowding loss — the sum of pairwise 1/d²
between seeds placed via the phyllotaxis rule:

    r_n = sqrt(n),   θ_n = n · α

The golden angle (≈ 137.508°) is the global minimizer because it
is the "most irrational" number: no two seeds ever land close together
for as long as possible before the spiral pattern repeats.

Run directly to verify convergence:
    python phyllotaxis.py
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

# φ = (1 + √5) / 2,  golden angle = 360° × (2 − φ) ≈ 137.508°
PHI = (1.0 + 5.0 ** 0.5) / 2.0
GOLDEN_ANGLE_DEG = 360.0 * (2.0 - PHI)
GOLDEN_ANGLE_RAD = GOLDEN_ANGLE_DEG * jnp.pi / 180.0


class PhyllotaxisModel(eqx.Module):
    """Single learnable divergence angle for phyllotaxis spiral."""
    alpha: jnp.ndarray   # shape (), radians


def seed_positions(alpha: jnp.ndarray, N: int = 100):
    """
    Place N seeds using the phyllotaxis rule.
    Returns (x, y) arrays of shape (N,).
    """
    n = jnp.arange(1, N + 1, dtype=jnp.float32)
    x = jnp.sqrt(n) * jnp.cos(n * alpha)
    y = jnp.sqrt(n) * jnp.sin(n * alpha)
    return x, y


def crowding_loss(model: PhyllotaxisModel, N: int = 100) -> jnp.ndarray:
    """
    Pairwise Coulomb-style repulsion: sum of 1/d² over all seed pairs
    (upper triangle only). High when seeds cluster; minimized when maximally spread.
    """
    x, y = seed_positions(model.alpha, N)
    dx = x[:, None] - x[None, :]   # (N, N)
    dy = y[:, None] - y[None, :]
    d2 = dx ** 2 + dy ** 2
    # push diagonal to infinity so it contributes zero to 1/d²
    d2_safe = d2 + jnp.eye(N, dtype=jnp.float32) * 1e6
    upper = jnp.triu(jnp.ones((N, N), dtype=jnp.float32), k=1)
    return jnp.sum(upper / d2_safe)


def train(
    initial_alpha_deg: float,
    n_steps: int = 4000,
    lr: float = 3e-4,
    log_every: int = 20,
    N: int = 100,
) -> tuple:
    """
    Train PhyllotaxisModel from initial_alpha_deg.
    Returns (final_model, history_dict).

    history_dict keys: steps, alpha_degrees, loss
    """
    alpha0 = jnp.array(initial_alpha_deg * jnp.pi / 180.0, dtype=jnp.float32)
    model = PhyllotaxisModel(alpha=alpha0)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step_fn(model, opt_state):
        loss_val, grads = eqx.filter_value_and_grad(crowding_loss)(model)
        updates, new_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        return eqx.apply_updates(model, updates), new_state, loss_val

    history = {"steps": [], "alpha_degrees": [], "loss": []}

    for i in range(n_steps):
        model, opt_state, loss_val = step_fn(model, opt_state)
        if i % log_every == 0:
            alpha_deg = float(model.alpha) * 180.0 / jnp.pi % 360.0
            history["steps"].append(i)
            history["alpha_degrees"].append(round(alpha_deg, 4))
            history["loss"].append(round(float(loss_val), 4))

    return model, history


if __name__ == "__main__":
    print(f"Golden angle reference: {GOLDEN_ANGLE_DEG:.4f}°\n")

    for start in [90.0, 120.0, 45.0]:
        print(f"Training from {start}°...")
        model, history = train(initial_alpha_deg=start, n_steps=4000, lr=3e-4)
        final = float(model.alpha) * 180.0 / jnp.pi % 360.0
        print(f"  Final: {final:.4f}°  (target ≈ {GOLDEN_ANGLE_DEG:.4f}°)")
        print(f"  Loss : {history['loss'][-1]:.4f}\n")
