"""Utility functions for viscoelastic simulations.

Includes:
- Strain computation at a cell via least-squares gradient reconstruction
- Publication-quality matplotlib style setup
- Experimental and 1D reference data (digitized from article Figure 3)
- Result save/load helpers
"""

import json
import numpy as np
import matplotlib as mpl


# ============================================================================
# Strain computation
# ============================================================================
def compute_strain_at_cell(sd, u_vec_flat, cell_idx, adj_csr, nd=2):
    """Compute strain tensor at a single cell via least-squares gradient.

    Uses displacement values from neighboring cells (connected via shared
    faces) to reconstruct the displacement gradient, then extracts the
    symmetric part.

    Parameters
    ----------
    sd : pp.Grid
        The subdomain grid.
    u_vec_flat : np.ndarray
        Displacement vector, F-ordered: [x0,x1,...,y0,y1,...].
    cell_idx : int
        Index of the cell at which to compute strain.
    adj_csr : scipy.sparse.csr_matrix
        Precomputed cell adjacency matrix.
    nd : int
        Number of spatial dimensions (2).

    Returns
    -------
    exx, eyy, exy : float
        Strain tensor components.
    """
    u = u_vec_flat.reshape(nd, -1, order="F")
    ux, uy = u[0], u[1]
    cc = sd.cell_centers[:nd, :]

    # Neighbor indices from precomputed adjacency
    row_start = adj_csr.indptr[cell_idx]
    row_end = adj_csr.indptr[cell_idx + 1]
    neighbors = adj_csr.indices[row_start:row_end]
    neighbors = neighbors[neighbors != cell_idx]

    if len(neighbors) < 2:
        return 0.0, 0.0, 0.0

    dx = cc[0, neighbors] - cc[0, cell_idx]
    dy = cc[1, neighbors] - cc[1, cell_idx]
    A = np.column_stack([dx, dy])

    dux = ux[neighbors] - ux[cell_idx]
    duy = uy[neighbors] - uy[cell_idx]

    grad_ux, _, _, _ = np.linalg.lstsq(A, dux, rcond=None)
    grad_uy, _, _, _ = np.linalg.lstsq(A, duy, rcond=None)

    exx = grad_ux[0]
    eyy = grad_uy[1]
    exy = 0.5 * (grad_ux[1] + grad_uy[0])
    return exx, eyy, exy


# ============================================================================
# Publication-quality matplotlib style
# ============================================================================
def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "CMU Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "lines.linewidth": 1.5,
        "lines.markersize": 7,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.top": True,
        "ytick.right": True,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


# ============================================================================
# Experimental data (digitized from article Figure 3)
# ============================================================================
EXPERIMENTAL_DATA_T = np.array([
    0.0, 0.15, 0.30, 0.45, 0.70, 1.00, 1.25, 1.50, 1.80, 2.10,
    2.35, 2.60, 2.85, 3.10, 3.35, 3.60, 3.85, 4.10, 4.35, 4.60,
    4.85, 5.10, 5.35, 5.60, 5.85, 6.10, 6.35, 6.60, 6.85, 7.10,
    7.35, 7.60,
])

EXPERIMENTAL_DATA_EPS = np.array([
    0.1115, 0.1180, 0.1280, 0.1340, 0.1375, 0.1380, 0.1390, 0.1390,
    0.1400, 0.1400, 0.1395, 0.1405, 0.1405, 0.1400, 0.1390, 0.1395,
    0.1390, 0.1390, 0.1395, 0.1395, 0.1390, 0.1395, 0.1390, 0.1390,
    0.1390, 0.1395, 0.1400, 0.1410, 0.1400, 0.1395, 0.1400, 0.1400,
])

# 1D simulation data (digitized from article's red curve)
SIM_1D_T = np.array([
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2,
    1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0,
])

SIM_1D_EPS = np.array([
    0.1110, 0.1155, 0.1205, 0.1245, 0.1280, 0.1310, 0.1330, 0.1355,
    0.1370, 0.1380, 0.1390, 0.1395, 0.1398, 0.1400, 0.1400, 0.1400,
    0.1400,
])


# ============================================================================
# Result save/load
# ============================================================================
def save_convergence_results(filepath: str, results: list[dict]) -> None:
    """Save convergence results to a JSON file.

    Parameters
    ----------
    filepath : str
        Output JSON file path.
    results : list[dict]
        List of dicts with keys: cell_size, dt, error_abs, error_rel, etc.
    """
    # Convert numpy types to Python types for JSON serialization
    serializable = []
    for r in results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                sr[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                sr[k] = float(v)
            else:
                sr[k] = v
        serializable.append(sr)

    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)


def load_convergence_results(filepath: str) -> list[dict]:
    """Load convergence results from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def save_strain_history(filepath: str, strain_history: dict) -> None:
    """Save strain history to NPZ file."""
    np.savez(filepath, **{k: np.array(v) for k, v in strain_history.items()})


def load_strain_history(filepath: str) -> dict:
    """Load strain history from NPZ file."""
    data = np.load(filepath)
    return {k: data[k] for k in data.files}
