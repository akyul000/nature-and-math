import numpy as onp
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from pathlib import Path 

out = Path("../../docs/chapters/01-nature-and-math/data")
out.mkdir(parents=True, exist_ok=True)


@dataclass
class ProjectileModel(ABC): # base class for projectile models, with an abstract method vector_field that subclasses must implement
    g: float = field(init=False, default=9.81)
    @abstractmethod
    def vector_field(self, t, state, args):
        """Return d/dt of state = [x, y, vx, vy]."""
        ...

    def trajectory(self, v0: float, theta_degree: float, t0=0.0, t1=5.0, n_points=200, dt_solver=0.01):
        """
        Integrate from t0 to t1 and return (x, y) arrays.
        """
        theta_rad = np.radians(theta_degree)
        state0 = np.array([0., 0., v0 * np.cos(theta_rad), v0 * np.sin(theta_rad)]) # d/dt(state) 

        term = ODETerm(self.vector_field) # the right-hand side of the ODE, i.e. the vector field function (d/dt(state) = f(t, state))
        solver = Tsit5() 
        saveat = SaveAt(ts=np.linspace(t0, t1, n_points)) # specify the time points at which to save the solution
        # solve the ODE using Diffrax's diffeqsolve function, which takes the ODE term, solver, time span, initial state, and other options
        sol = diffeqsolve(term, solver, t0=t0, t1=t1,dt0=dt_solver, y0=state0, saveat=saveat, stepsize_controller=PIDController(rtol=1e-6, atol=1e-8))
        x = sol.ys[:, 0]
        y = sol.ys[:, 1]
        return x, y

# Level 1
@dataclass
class Level1(ProjectileModel): # subclass of ProjectileModel that implements the vector_field method for a point mass under gravity only
    """Point mass, gravity only. Closed form but written as ODE
    for consistency with levels 2 and 3."""

    def vector_field(self, t, state, args):
        x, y, vx, vy = state
        return np.array([vx, vy, 0.0, -self.g])

# Level 2
@dataclass
class Level2(ProjectileModel):
    """Gravity + quadratic aerodynamic drag."""
    rho: float = 1.225
    Cd: float = 0.47
    r: float  = 0.033
    A: float = field(init=False)
    m: float = 0.057
    k: float = field(init=False)
    def __post_init__(self):
        self.A = np.pi * self.r**2
        self.k = self.rho * self.Cd * self.A / (2 * self.m)
    def vector_field(self, t, state, args):
        x, y, vx, vy = state
        v_mag = np.sqrt(vx**2 + vy**2)
        return np.array([vx, vy, -self.k * v_mag * vx, -self.g - self.k * v_mag * vy])
    
# Level 3
@dataclass
class Level3(ProjectileModel):
    """Gravity + drag + Magnus force (spin)."""
    rho: float = 1.225
    Cd: float = 0.47
    Cl: float = 0.33
    omega: float = 0.0
    r: float  = 0.033
    A: float = field(init=False)
    m: float = 0.057
    k: float = field(init=False)
    S: float = field(init=False)
    def __post_init__(self):
        self.A = np.pi * self.r**2
        self.k = self.rho * self.Cd * self.A / (2 * self.m)
        self.S = self.Cl * self.rho * self.A * self.r
    def vector_field(self, t, state, args):
        x, y, vx, vy = state
        v_mag = np.sqrt(vx**2 + vy**2)
        ax = -self.S * self.omega / self.m * vy - self.k * v_mag * vx
        ay = self.S * self.omega / self.m * vx - self.k * v_mag * vy - self.g
        return np.array([vx, vy, ax, ay])



V0_VALUES    = range(10, 45, 5)    # 10 15 20 … 40
THETA_VALUES = range(15, 80, 5)    # 15 20 … 75
SPIN_RPM     = 300                 # rpm → rad/s below
omega = SPIN_RPM * 2 * onp.pi / 60


v0s     = np.array([float(v) for v in V0_VALUES])     # shape (7,)
thetas  = np.array([float(t) for t in THETA_VALUES])  # shape (13,)

models = {
    "level1": Level1(),
    "level2": Level2(),
    "level3": Level3(omega=omega)
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

        fname = out / f"projectile_v0_{v0}_theta_{theta}.json"
        fname.write_text(json.dumps(payload, separators=(",", ":")))
        print(f"wrote {fname.name}")


cmap       = cm.plasma

fig, axes = plt.subplots(
    1, len(v0_list),
    figsize=(4 * len(v0_list), 4),
    sharey=True,
)

linestyles = {"level1": "-", "level2": "--", "level3": ":"}  # solid / dashed / dotted

for i_v0, (ax, v0) in enumerate(zip(axes, v0_list)):
    for i_theta, theta in enumerate(theta_list):
        color = cmap(i_theta / (len(theta_list) - 1))
        for name in models:
            x = trajectories[name][0][i_v0, i_theta]
            y = trajectories[name][1][i_v0, i_theta]
            mask = y >= 0
            ax.plot(x[mask], y[mask], color=color,
                    linestyle=linestyles[name], linewidth=1.2)

    ax.set_title(f"$v_0 = {v0}$ m/s")
    ax.set_xlabel("x (m)")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_ylim(bottom=0)

axes[0].set_ylabel("y (m)")

# legend for model levels
from matplotlib.lines import Line2D
labels = {
    "level1": "L1 — gravity only",
    "level2": "L2 — + drag",
    "level3": f"L3 — + Magnus (ω={SPIN_RPM} rpm)",
}
legend_handles = [Line2D([0], [0], color="gray", linestyle=ls, linewidth=1.5,
                         label=labels[name])
                  for name, ls in linestyles.items()]
axes[-1].legend(handles=legend_handles, loc="upper right", fontsize=7)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(theta_list[0], theta_list[-1]))
sm.set_array([])
fig.colorbar(sm, ax=axes, label="launch angle (°)", shrink=0.8)

fig.suptitle("Projectile trajectories — gravity only (—) · +drag (--) · +Magnus (···)", y=1.01)
plt.tight_layout()
plt.savefig("trajectories.png", dpi=150, bbox_inches="tight")
plt.show()