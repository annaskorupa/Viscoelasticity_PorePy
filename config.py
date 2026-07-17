"""Shared material and simulation constants.

This module centralizes the physical parameters used across all runner
scripts so that every simulation starts from the same set of constants.

Two parameter sets are provided:

* **Creep test** (8-hour simulation, Sections 4.1 / 4.2 of the article):
  ``NU``, ``E1``, ``E2``, ``ETA``, ``DT``, ``FINAL_TIME``.

* **Convergence test** (100-second benchmark, higher stiffness):
  ``CONV_NU``, ``CONV_E1``, ``CONV_E2``, ``CONV_ETA``, ``CONV_T_FINAL``,
  ``CONV_CELL_SIZES``, ``CONV_DT_VALUES``.
"""

import porepy as pp

# =====================================================================
# Creep test constants (8-hour simulation)
# =====================================================================
NU = 0.0
E1 = 2_143_000_000.0             # 2143 MPa  [Pa]
E2 = 584_000_000.0               # 584 MPa   [Pa]
ETA = 180_000_000.0 * (60.0 * 60.0)  # 180 MPa·h → Pa·s

DT = 1.0 * pp.SECOND             # time step  [s]
FINAL_TIME = 8.0 * pp.HOUR       # final time [s]

# =====================================================================
# Convergence test constants (100 s benchmark, higher stiffness)
# =====================================================================
CONV_NU = 0.0
CONV_E1 = 22_575_700_000.0       # 22575.7 MPa [Pa]
CONV_E2 = 11_000_000_000.0       # 11000.0 MPa [Pa]
CONV_ETA = 11_000_000_000.0 * (45.454545 * 24.0 * 60.0 * 60.0)  # Pa·s

CONV_T_FINAL = 100.0             # snapshot time [s]

# Mesh / time-step refinement levels (4 levels for proper convergence study).
# Nx = [10, 20, 40, 80] on the 0.1×0.1 m domain.
# The finest level (cell_size=0.00125, Nx=80) is used as the reference
# solution, so only the first 3 levels contribute error data points.
CONV_CELL_SIZES = [0.01, 0.005, 0.0025, 0.00125]
CONV_DT_VALUES = [8.0, 4.0, 2.0, 1.0]
