import json
import numpy as onp
from pathlib import Path
from proejectile import Level1, Level2, Level3   # relative import
import jax
import jax.numpy as np

OUT = Path(__file__).parent.parent.parent / "docs/chapters/01-nature-and-math/data"
OUT.mkdir(parents=True, exist_ok=True)

V0_VALUES    = range(10, 45, 5)    # 10 15 20 … 40
THETA_VALUES = range(15, 80, 5)    # 15 20 … 75
SPIN_RPM     = 300                 # rpm → rad/s below

omega = SPIN_RPM * 2 * onp.pi / 60

v0s     = np.array([float(v) for v in V0_VALUES])     # shape (7,)
thetas  = np.array([float(t) for t in THETA_VALUES])  # shape (13,)


models = {
    "level1": Level1(),
    "level2": Level2(),
    "level3": Level3(omega=omega),
}

# precompute all trajectories in one batched+JIT'd call per model
trajectories = {}
for name, model in models.items():
    batch_traj = jax.jit(jax.vmap(jax.vmap(model.trajectory,
                                             in_axes=(None, 0)),
                                    in_axes=(0, None)))
    xs, ys = batch_traj(v0s, thetas)          # (n_v0, n_theta, n_points)
    trajectories[name] = (onp.array(xs), onp.array(ys))


theta_list = list(THETA_VALUES)
v0_list    = list(V0_VALUES)
for i_v0, v0 in enumerate(v0_list):
    for i_theta, theta in enumerate(theta_list):

        payload = {"params": {"v0": v0, "theta_deg": theta}, "models": {}}

        for name in models:
            x = trajectories[name][0][i_v0, i_theta]  # slice → shape (n_points,)
            y = trajectories[name][1][i_v0, i_theta]

            mask = onp.array(y) >= 0                   # clip below ground
            payload["models"][name] = {
                "x": onp.array(x[mask]).round(4).tolist(),
                "y": onp.array(y[mask]).round(4).tolist(),
            }

        fname = OUT / f"projectile_v0_{v0}_theta_{theta}.json"
        fname.write_text(json.dumps(payload, separators=(",", ":")))
        print(f"wrote {fname.name}")

