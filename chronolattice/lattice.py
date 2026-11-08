"""D2Q9 Lattice Boltzmann discrete velocity model, weights, and equilibrium distribution."""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


class D2Q9Lattice:
    """Standard 2D 9-velocity (D2Q9) lattice structure."""

    # Discrete velocity vectors: c_i = [c_ix, c_iy]
    C = np.array([
        [0, 0],    # 0: Rest particle
        [1, 0],    # 1: East
        [0, 1],    # 2: North
        [-1, 0],   # 3: West
        [0, -1],   # 4: South
        [1, 1],    # 5: North-East
        [-1, 1],   # 6: North-West
        [-1, -1],  # 7: South-West
        [1, -1],   # 8: South-East
    ], dtype=np.int32)

    # Direction opposite mapping: c[OPPOSITE[i]] == -c[i]
    OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

    # Lattice weights
    WEIGHTS = np.array([
        4.0 / 9.0,   # i = 0
        1.0 / 9.0,   # i = 1, 2, 3, 4
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 36.0,  # i = 5, 6, 7, 8
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
    ], dtype=np.float64)

    CS2 = 1.0 / 3.0       # Speed of sound squared: c_s^2 = 1/3
    CS4 = 1.0 / 9.0       # c_s^4 = 1/9
    INV_CS2 = 3.0         # 1 / c_s^2
    INV_2CS4 = 4.5        # 1 / (2 c_s^4)
    INV_2CS2 = 1.5        # 1 / (2 c_s^2)

    @classmethod
    def compute_equilibrium(cls, rho: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        r"""Computes the 2nd-order Maxwell-Boltzmann discrete equilibrium distribution:

        f_i^{eq} = w_i \rho \left(1 + \frac{c_i \cdot u}{c_s^2} + \frac{(c_i \cdot u)^2}{2 c_s^4} - \frac{u^2}{2 c_s^2}\right)
        """
        ny, nx = rho.shape
        f_eq = np.zeros((9, ny, nx), dtype=np.float64)
        u2 = u * u + v * v

        for i in range(9):
            cx = cls.C[i, 0]
            cy = cls.C[i, 1]
            ci_dot_u = cx * u + cy * v
            f_eq[i] = cls.WEIGHTS[i] * rho * (
                1.0 + cls.INV_CS2 * ci_dot_u + cls.INV_2CS4 * (ci_dot_u * ci_dot_u) - cls.INV_2CS2 * u2
            )
        return f_eq

    @classmethod
    def compute_macroscopic(
        cls,
        f: np.ndarray,
        force_x: Optional[np.ndarray] = None,
        force_y: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Recovers macroscopic density \rho and velocity (u, v) from distributions:

        \rho = \sum_i f_i
        \rho u = \sum_i f_i c_ix + 0.5 * F_x
        \rho v = \sum_i f_i c_iy + 0.5 * F_y
        """
        rho = np.sum(f, axis=0)
        # Prevent division by zero
        rho_safe = np.where(rho < 1e-12, 1e-12, rho)

        # Momentum calculation
        rhou = (
            f[1] - f[3] + f[5] - f[6] - f[7] + f[8]
        )
        rhov = (
            f[2] - f[4] + f[5] + f[6] - f[7] - f[8]
        )

        if force_x is not None:
            rhou += 0.5 * force_x
        if force_y is not None:
            rhov += 0.5 * force_y

        u = rhou / rho_safe
        v = rhov / rho_safe
        return rho, u, v
