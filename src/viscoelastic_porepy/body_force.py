"""MMS body force for the viscoelastic verification problem.

Computes the forcing term f(x, t) derived from the Manufactured Method
of Solutions so that the analytical displacement field is an exact
solution of the governing equations.
"""

import numpy as np
import porepy as pp

from viscoelastic_porepy.material import ViscoelasticSolidConstants


class BodyForceMixin:
    """Time-dependent MMS body force.

    The force is updated every time step via
    ``before_nonlinear_loop`` and stored in the data dictionary
    for ``TimeDependentDenseArray``.
    """

    solid: ViscoelasticSolidConstants
    units: pp.Units

    def _compute_body_force_values(
        self, subdomains: list[pp.Grid]
    ) -> np.ndarray:
        """Compute MMS body force at the current time step.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Concatenated body force array for all subdomains.
        """
        vals = []

        # MMS parameters
        A_MMS = 1.0e-3
        Ax, Ay = A_MMS, A_MMS
        b_MMS = 0.5 * (self.solid.shear_modulus2 / self.solid.viscosity)
        beta = self.solid.shear_modulus2 / self.solid.viscosity
        t = self.time_manager.time

        # Material parameters
        shear_modulus = self.solid.shear_modulus
        shear_modulus2 = self.solid.shear_modulus2
        lame_lambda = self.solid.lame_lambda
        lame_lambda2 = self.solid.lame_lambda2

        # Domain and wave numbers
        L = 0.8
        kx = np.pi / L
        ky = np.pi / L

        for sd in subdomains:
            data = np.zeros((sd.num_cells, self.nd))

            if sd.dim == 2:
                cc = sd.cell_centers
                x, y = cc[0], cc[1]

                sx = np.sin(kx * x)
                cx = np.cos(kx * x)
                sy = np.sin(ky * y)
                cy = np.cos(ky * y)

                # Temporal functions
                T1 = 1.0 - np.exp(-b_MMS * t)
                T2 = (b_MMS / (beta - b_MMS)) * (
                    np.exp(-b_MMS * t) - np.exp(-beta * t)
                )

                # Elastic branch spatial terms
                term_x_E = (
                    (lame_lambda + 2 * shear_modulus) * kx**2 * Ax
                    + shear_modulus * ky**2 * Ax
                ) * sx * sy - (
                    (lame_lambda + shear_modulus) * kx * ky * Ay
                ) * cx * cy

                term_y_E = (
                    (lame_lambda + 2 * shear_modulus) * ky**2 * Ay
                    + shear_modulus * kx**2 * Ay
                ) * sx * sy - (
                    (lame_lambda + shear_modulus) * kx * ky * Ax
                ) * cx * cy

                # Viscous branch spatial terms
                term_x_E_vis = (
                    (lame_lambda2 + 2 * shear_modulus2) * kx**2 * Ax
                    + shear_modulus2 * ky**2 * Ax
                ) * sx * sy - (
                    (lame_lambda2 + shear_modulus2) * kx * ky * Ay
                ) * cx * cy

                term_y_E_vis = (
                    (lame_lambda2 + 2 * shear_modulus2) * ky**2 * Ay
                    + shear_modulus2 * kx**2 * Ay
                ) * sx * sy - (
                    (lame_lambda2 + shear_modulus2) * kx * ky * Ax
                ) * cx * cy

                # Combined force
                force_x = term_x_E * T1 + term_x_E_vis * T2
                force_y = term_y_E * T1 + term_y_E_vis * T2

                data[:, 0] = force_x * sd.cell_volumes
                data[:, 1] = force_y * sd.cell_volumes

            vals.append(data.ravel())

        return np.concatenate(vals)

    def body_force(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Return body force as a TimeDependentDenseArray.

        Values are updated in the data dictionary each time step
        by ``before_nonlinear_loop``.
        """
        self._bf_subdomains = subdomains
        return pp.ad.TimeDependentDenseArray("body_force", subdomains)
