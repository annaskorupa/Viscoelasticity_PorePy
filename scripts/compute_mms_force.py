#!/usr/bin/env python
"""Compute MMS body force symbolically using SymPy.

Derives the forcing term f(x, t) = −∇·σ from the manufactured
displacement field and viscoelastic constitutive law.
"""

import sympy as sp

# Variables
x, y, t = sp.symbols("x y t")
Ax, Ay, b, beta = sp.symbols("Ax Ay b beta")
Lx, Ly = sp.symbols("Lx Ly")
shear_modulus, shear_modulus2, k = sp.symbols(
    "shear_modulus shear_modulus2 k"
)

# Temporal functions
T1 = 1 - sp.exp(-b * t)
T2 = (b / (beta - b)) * (sp.exp(-b * t) - sp.exp(-beta * t))

# Spatial displacement field
ux_s = Ax * sp.sin(sp.pi * x / Lx) * sp.cos(sp.pi * y / Ly)
uy_s = Ay * sp.sin(sp.pi * x / Lx) * sp.sin(sp.pi * y / Ly)

# Spatial strain components
eps_xx_s = sp.diff(ux_s, x)
eps_yy_s = sp.diff(uy_s, y)
eps_xy_s = sp.Rational(1, 2) * (sp.diff(ux_s, y) + sp.diff(uy_s, x))

# Trace and deviatoric parts (plane strain)
tr_eps_s = eps_xx_s + eps_yy_s
dev_eps_xx_s = eps_xx_s - tr_eps_s / 3
dev_eps_yy_s = eps_yy_s - tr_eps_s / 3
dev_eps_xy_s = eps_xy_s

# Total viscoelastic stress
sig_xx = (
    2 * shear_modulus * dev_eps_xx_s * T1
    + 2 * shear_modulus2 * dev_eps_xx_s * T2
    + k * tr_eps_s * T1
)
sig_yy = (
    2 * shear_modulus * dev_eps_yy_s * T1
    + 2 * shear_modulus2 * dev_eps_yy_s * T2
    + k * tr_eps_s * T1
)
sig_xy = (
    2 * shear_modulus * dev_eps_xy_s * T1
    + 2 * shear_modulus2 * dev_eps_xy_s * T2
)

# MMS forcing: f = −∇·σ
fx = -(sp.diff(sig_xx, x) + sp.diff(sig_xy, y))
fy = -(sp.diff(sig_xy, x) + sp.diff(sig_yy, y))

if __name__ == "__main__":
    print("--- Force fx ---")
    print(sp.simplify(fx))
    print("\n--- Force fy ---")
    print(sp.simplify(fy))
