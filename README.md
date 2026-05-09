# Nature & Math

An interactive textbook connecting the mathematics behind natural phenomena — from first principles to PDEs.

Live site: **https://akyul000.github.io/nature-and-math/**

---

## Chapters

| # | Title | Topics |
|---|---|---|
| 01 | Nature & Math | Modeling philosophy, digital twins, surrogate models |
| 02 | Building Intuition | What ∂u/∂t and ∇²u actually mean |
| 03 | Fourier's Story | How solving heat led to Fourier series |
| 04 | The Heat Equation | Derivation, energy balance, interactive diffusion sim |
| 05 | PDE Classification | Elliptic / parabolic / hyperbolic — the discriminant B²−4AC |
| 06 | Elliptic PDEs | Laplace, Poisson — steady-state phenomena |
| 07 | Parabolic PDEs | Diffusion, smoothing, one-way time |
| 08 | Hyperbolic PDEs | Wave equation, finite-speed propagation |

---

## Repository Layout

```
docs/           # GitHub Pages root — HTML, CSS, JS, pre-computed data
simulations/    # Python/JAX solvers and data generators (run offline)
.github/        # GitHub Actions: auto-deploy on push
```

## Running Simulations

```bash
cd simulations
pip install -r requirements.txt
python generate_all.py
```

This writes JSON data files into `docs/chapters/*/data/` which the browser loads for interactive plots.

## GitHub Pages Setup

1. Go to **Settings → Pages**
2. Source: **Deploy from branch `main`**, folder `/docs`
3. Save — your site will appear at `https://<username>.github.io/nature-and-math/`
