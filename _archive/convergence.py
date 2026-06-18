"""
Convergence plot for the viscoelastic MPSA-Newmark scheme.

Produces a publication-quality log-log convergence plot styled after
Jacobsen et al. (2025), ARC Geophysical Research, doi:10.5149/ARC-GR.1598
(Figures 3 and 5).

The combined resolution measure is  R = (Nx^2 * Nt)^{1/4}  where Nx is the
number of cells per direction (2D Cartesian grid) and Nt = 1/dt is the
reciprocal time-step size (both refined simultaneously).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Use LaTeX-style rendering (Computer Modern) ─────────────────────────────
mpl.rcParams.update({
    # Text
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "CMU Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.unicode_minus": False,
    # Font sizes  (matching typical journal figures)
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    # Lines & markers
    "lines.linewidth": 1.5,
    "lines.markersize": 7,
    # Axes
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linewidth": 0.5,
    # Ticks
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.top": True,
    "ytick.right": True,
    # Figure
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# =============================================================================
# Data
# =============================================================================
# Mesh steps: 10×10 → h=0.08, 20×20 → 0.04, 40×40 → 0.02, 80×80 → 0.01
Nx = np.array([10, 20, 40, 80])

# Time steps (s): 8, 4, 2, 1  →  Nt = 1/dt
dt = np.array([8.0, 4.0, 2.0, 1.0])
Nt = 1.0 / dt

# Absolute L2 errors [m]
errors_abs = np.array([3.276869e-11, 8.593092e-12, 2.181605e-12, 5.461565e-13])

# Relative L2 errors [-]
errors_rel = np.array([1.867602e-02, 4.897503e-03, 1.243373e-03, 3.112736e-04])

# Combined resolution measure (same as in the article)
resolution = (Nx**2 * Nt) ** 0.25

# ── Print convergence table ─────────────────────────────────────────────────
header = f"{'Level':<8}{'Nx':>6}{'dt':>8}{'R':>10}{'rel L2':>14}{'order':>8}"
print(header)
print("-" * len(header))
for i in range(len(errors_rel)):
    order_str = "--" if i == 0 else f"{np.log2(errors_rel[i-1] / errors_rel[i]):.2f}"
    print(
        f"{i:<8}{Nx[i]:>6}{dt[i]:>8.1f}{resolution[i]:>10.4f}"
        f"{errors_rel[i]:>14.4e}{order_str:>8}"
    )

# =============================================================================
# Plot
# =============================================================================
fig, ax = plt.subplots(figsize=(5.5, 4.5))

# ── Main data curve ──────────────────────────────────────────────────────────
ax.loglog(
    resolution,
    errors_rel,
    "o-",
    color="#1f77b4",
    markerfacecolor="white",
    markeredgewidth=1.4,
    markeredgecolor="#1f77b4",
    zorder=5,
    label=r"$\|\mathbf{u} - \mathbf{u}_h\|_{L^2}$ / $\|\mathbf{u}\|_{L^2}$",
)

# ── Reference slope triangle (order 2) ──────────────────────────────────────
# Place between the last two data points, shifted below the data
x1, x2 = resolution[-2], resolution[-1]
y_base = errors_rel[-2] * 2.0          # shift below data line
y_top  = y_base * (x1 / x2) ** 2       # y ~ x^{-2}  → top-left is higher

# Three-sided triangle
tri_x = [x1, x2, x2, x1]
tri_y = [y_base, y_base, y_top, y_base]   # closed path

ax.fill(tri_x, tri_y, color="#d3d3d3", edgecolor="0.3", linewidth=0.8, zorder=3)
ax.plot([x1, x2], [y_base, y_base], color="0.3", linewidth=0.8, zorder=3)
ax.plot([x2, x2], [y_base, y_top],  color="0.3", linewidth=0.8, zorder=3)
ax.plot([x1, x2], [y_base, y_top],  color="0.3", linewidth=0.8, zorder=3)

# Label "2" inside the triangle (geometric center)
ax.text(
    np.sqrt(x1 * x2),
    (y_base * y_top) ** 0.5 * 0.85,
    "2",
    fontsize=12,
    fontweight="bold",
    ha="center",
    va="center",
    color="0.2",
    zorder=6,
)

# ── Axis labels & formatting ────────────────────────────────────────────────
ax.set_xlabel(r"$(N_x^{\,2}\cdot N_t)^{1/4}$")
ax.set_ylabel(r"Relative $L^2$ error")

ax.grid(True, which="major", ls="-",  lw=0.5, alpha=0.35)
ax.grid(True, which="minor", ls=":",  lw=0.3, alpha=0.25)

ax.legend(loc="upper right", frameon=True, fancybox=False,
          edgecolor="0.7", framealpha=0.95)

# Keep tick labels clean (no scientific notation on log axes)
ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

fig.tight_layout()

# ── Save & show ──────────────────────────────────────────────────────────────
fig.savefig("convergence.png", dpi=300)
print("\n[OK]  Saved convergence.png")
plt.show()
print("done")
