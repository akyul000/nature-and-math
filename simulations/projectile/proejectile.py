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