# Viscoelastic PorePy — Generalized Maxwell Model

A viscoelastic extension of [PorePy](https://github.com/pmgbergen/porepy)'s
`MomentumBalance` model, implementing a **Generalized Maxwell** constitutive
law with MPSA (Multi-Point Stress Approximation) discretization.

## Overview

The model introduces a second displacement variable **u₂** representing the
viscous (dashpot) branch of a Maxwell element. The total stress is the sum of
elastic and viscous contributions:

```
σ = σ₁(u) + σ₂(u₂)
```

The viscous branch is governed by the rate equation:

```
du₂/dt + β·u₂ − du/dt = 0,    β = μ₂/η
```

Based on the framework of Idesman et al. (2000), validated via the Method of
Manufactured Solutions (MMS).

## Project Structure

```
Program/
├── src/viscoelastic_porepy/     # Python package (the model)
│   ├── __init__.py
│   ├── material.py              # ViscoelasticSolidConstants (μ₂, λ₂, η)
│   ├── geometry.py              # 2D square domain geometry
│   ├── constitutive.py          # Elastic moduli, MPSA stress for u₂
│   ├── variables.py             # u₂ and interface_u₂ variables
│   ├── equations.py             # Rate equation (du₂/dt + β·u₂ − du/dt = 0)
│   ├── boundary_conditions.py   # MMS Dirichlet BCs
│   ├── initial_conditions.py    # Zero ICs for u₂
│   ├── solution_strategy.py     # MPSA setup for u₂ ("mechanics2")
│   ├── body_force.py            # Time-dependent MMS body force
│   └── model.py                 # ViscoelasticMomentumBalance (assembles all)
│
├── scripts/
│   ├── run_simulation.py        # Main entry point — run the simulation
│   ├── convergence_study.py     # Convergence analysis & plotting
│   └── compute_mms_force.py     # Symbolic MMS force derivation (SymPy)
│
├── tests/                       # Technical tests for PorePy internals
├── docs/change_logs/            # Development history
├── _output/                     # Simulation output (gitignored)
│   ├── plots/
│   └── visualization/
├── _archive/                    # Legacy scripts (gitignored)
│
├── pyproject.toml               # Project metadata & dependencies
├── README.md                    # This file
└── .gitignore
```

## Material Parameters

| Parameter         | Symbol | Value                  | Unit   |
|-------------------|--------|------------------------|--------|
| Elastic modulus   | E₁     | 22 575.7               | MPa    |
| Viscous modulus   | E₂     | 11 000.0               | MPa    |
| Poisson's ratio   | ν      | 0.0                    | —      |
| Relaxation time   | τ      | 45.45                  | days   |
| Viscosity         | η      | E₁ × τ                | Pa·s   |

## Quick Start

### Prerequisites

- Python ≥ 3.10
- PorePy (with dependencies: `numpy`, `scipy`, `gmsh`, etc.)

### Installation

```bash
# From the project root:
pip install -e .
```

### Running

```bash
# Run the full simulation (MMS verification)
python scripts/run_simulation.py

# Generate convergence plot only (uses pre-computed data)
python scripts/convergence_study.py

# Derive MMS body force symbolically
python scripts/compute_mms_force.py
```

## Convergence Results

The scheme achieves **2nd-order convergence** in the combined resolution
measure R = (Nx² · Nt)^{1/4}:

```
Level       Nx      dt         R        rel L2   order
------------------------------------------------------
0           10     8.0    1.8803    1.8676e-02      --
1           20     4.0    3.1623    4.8975e-03    1.93
2           40     2.0    5.3183    1.2434e-03    1.98
3           80     1.0    8.9443    3.1127e-04    2.00
```

## Architecture

The codebase follows PorePy's **mixin pattern** (cooperative multiple
inheritance). Each responsibility is encapsulated in a separate mixin class:

1. **Material** → `ViscoelasticSolidConstants` (extends `pp.SolidConstants`)
2. **Geometry** → `GeometryMixin` (domain size, mesh)
3. **Constitutive** → `ViscousElasticModuli`, `MechanicalStressU2`
4. **Variables** → `VariablesU2` (registers u₂ in the equation system)
5. **Equations** → `RateEquation` (the coupling ODE)
6. **BCs/ICs** → Manufactured Solution boundary and initial conditions
7. **Strategy** → MPSA discretization under keyword `"mechanics2"`

All mixins are assembled in `ViscoelasticMomentumBalance`, which inherits
from `pp.MomentumBalance` as the base class.

## References

- Idesman, A., Niekamp, R., & Stein, E. (2000). *Finite elements in
  analysis and design.*
- PorePy: [github.com/pmgbergen/porepy](https://github.com/pmgbergen/porepy)
