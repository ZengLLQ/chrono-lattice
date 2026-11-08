"""Shan-Chen single-component multiphase pseudopotential interaction model."""

from __future__ import annotations
import numpy as np
from typing import Tuple
from .lattice import D2Q9Lattice


class ShanChenMultiphase:
    r"""Shan-Chen pseudopotential model for liquid-vapor phase separation.

    The interparticle interaction force is given by:
        F(x) = -G * \psi(x) * \sum_{i=1}^8 w_i * \psi(x + c_i) * c_i
    """

    def __init__(self, g_interaction: float = -5.0, rho_0: float = 1.0, psi_mode: str = "exponential"):
        self.g = float(g_interaction)
        self.rho_0 = float(rho_0)
        self.psi_mode = psi_mode

    def compute_pseudopotential(self, rho: np.ndarray) -> np.ndarray:
        r"""Evaluates effective mass \psi(\rho)."""
        rho_safe = np.maximum(rho, 1e-9)
        if self.psi_mode == "exponential":
            # \psi(\rho) = \rho_0 * (1 - exp(-\rho / \rho_0))
            return self.rho_0 * (1.0 - np.exp(-rho_safe / self.rho_0))
        elif self.psi_mode == "sqrt":
            return np.sqrt(rho_safe)
        elif self.psi_mode == "density":
            return rho_safe
        else:
            raise ValueError(f"Unknown psi_mode: {self.psi_mode}")

    def compute_interaction_force(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates non-local interaction force (Fx, Fy) using nearest-neighbor stencil."""
        psi = self.compute_pseudopotential(rho)
        fx = np.zeros_like(rho, dtype=np.float64)
        fy = np.zeros_like(rho, dtype=np.float64)

        for i in range(1, 9):
            cx = D2Q9Lattice.C[i, 0]
            cy = D2Q9Lattice.C[i, 1]
            w = D2Q9Lattice.WEIGHTS[i]

            # Roll psi to sample neighbor at (y + cy, x + cx)
            psi_neighbor = np.roll(psi, shift=(-cy, -cx), axis=(0, 1))

            term = w * psi_neighbor
            fx += term * cx
            fy += term * cy

        fx = -self.g * psi * fx
        fy = -self.g * psi * fy
        return fx, fy

    def compute_equation_of_state_pressure(self, rho: np.ndarray) -> np.ndarray:
        r"""Calculates non-ideal pressure: P = c_s^2 * \rho + 0.5 * c_s^2 * G * \psi(\rho)^2."""
        psi = self.compute_pseudopotential(rho)
        return D2Q9Lattice.CS2 * rho + 0.5 * D2Q9Lattice.CS2 * self.g * (psi * psi)
