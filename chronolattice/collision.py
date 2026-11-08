"""Collision operators for LBM: BGK (SRT) with Guo forcing and MRT (Multiple Relaxation Time)."""

from __future__ import annotations
import abc
import numpy as np
from typing import Optional, Tuple
from .lattice import D2Q9Lattice


class CollisionOperator(abc.ABC):
    """Abstract base class for lattice Boltzmann collision operators."""

    @abc.abstractmethod
    def collide(
        self,
        f: np.ndarray,
        rho: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        force_x: Optional[np.ndarray] = None,
        force_y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Executes local collision step, returning post-collision distribution f^*."""
        pass

    @property
    @abc.abstractmethod
    def relaxation_time(self) -> float:
        pass

    @property
    def kinematic_viscosity(self) -> float:
        """nu = c_s^2 * (tau - 0.5) = (2 * tau - 1) / 6."""
        return (2.0 * self.relaxation_time - 1.0) / 6.0


class BGKCollision(CollisionOperator):
    """Single Relaxation Time (Bhatnagar-Gross-Krook) with Guo forcing scheme."""

    def __init__(self, tau: float):
        if tau <= 0.5:
            raise ValueError(f"Relaxation time tau must be strictly > 0.5 for numerical stability (got {tau}).")
        self._tau = float(tau)
        self.omega = 1.0 / self._tau

    @property
    def relaxation_time(self) -> float:
        return self._tau

    def collide(
        self,
        f: np.ndarray,
        rho: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        force_x: Optional[np.ndarray] = None,
        force_y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        f_eq = D2Q9Lattice.compute_equilibrium(rho, u, v)
        f_post = f - self.omega * (f - f_eq)

        if force_x is not None or force_y is not None:
            fx = np.zeros_like(rho) if force_x is None else force_x
            fy = np.zeros_like(rho) if force_y is None else force_y

            # Guo forcing term: F_i = w_i * (1 - 0.5 * omega) * [ (c_i - u)/cs2 + (c_i . u) c_i / cs4 ] . F
            factor = 1.0 - 0.5 * self.omega
            for i in range(9):
                cx = D2Q9Lattice.C[i, 0]
                cy = D2Q9Lattice.C[i, 1]
                ci_dot_u = cx * u + cy * v

                term_x = (cx - u) * D2Q9Lattice.INV_CS2 + (ci_dot_u * cx) * (1.0 / D2Q9Lattice.CS4)
                term_y = (cy - v) * D2Q9Lattice.INV_CS2 + (ci_dot_u * cy) * (1.0 / D2Q9Lattice.CS4)

                source_i = factor * D2Q9Lattice.WEIGHTS[i] * (term_x * fx + term_y * fy)
                f_post[i] += source_i

        return f_post


class MRTCollision(CollisionOperator):
    """Multiple Relaxation Time (MRT) collision in Gram-Schmidt orthogonal moment space."""

    M = np.array([
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1],  # 0: density
        [-4, -1, -1, -1, -1,  2,  2,  2,  2],  # 1: energy e
        [ 4, -2, -2, -2, -2,  1,  1,  1,  1],  # 2: energy square eps
        [ 0,  1,  0, -1,  0,  1, -1, -1,  1],  # 3: momentum jx
        [ 0, -2,  0,  2,  0,  1, -1, -1,  1],  # 4: heat flux qx
        [ 0,  0,  1,  0, -1,  1,  1, -1, -1],  # 5: momentum jy
        [ 0,  0, -2,  0,  2,  1,  1, -1, -1],  # 6: heat flux qy
        [ 0,  1, -1,  1, -1,  0,  0,  0,  0],  # 7: normal stress pxx
        [ 0,  0,  0,  0,  0,  1, -1,  1, -1],  # 8: shear stress pxy
    ], dtype=np.float64)

    M_INV = np.linalg.inv(M)

    def __init__(self, tau: float, s_e: float = 1.1, s_eps: float = 1.0, s_q: float = 1.2):
        if tau <= 0.5:
            raise ValueError(f"Relaxation time tau must be strictly > 0.5 (got {tau}).")
        self._tau = float(tau)
        s_nu = 1.0 / self._tau
        # Conserved moments (0, 3, 5) have relaxation rates that do not affect physics (set to 0.0 or 1.0)
        self.S = np.array([0.0, s_e, s_eps, 0.0, s_q, 0.0, s_q, s_nu, s_nu], dtype=np.float64)

    @property
    def relaxation_time(self) -> float:
        return self._tau

    def collide(
        self,
        f: np.ndarray,
        rho: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        force_x: Optional[np.ndarray] = None,
        force_y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        ny, nx = rho.shape
        f_flat = f.reshape(9, -1)  # (9, N)

        # Transform to moment space: m = M @ f
        m = self.M @ f_flat

        # Compute equilibrium moments
        u2 = u.reshape(-1) ** 2 + v.reshape(-1) ** 2
        rho_flat = rho.reshape(-1)
        u_flat = u.reshape(-1)
        v_flat = v.reshape(-1)

        m_eq = np.zeros_like(m)
        m_eq[0] = rho_flat
        m_eq[1] = rho_flat * (-2.0 + 3.0 * u2)
        m_eq[2] = rho_flat * (1.0 - 3.0 * u2)
        m_eq[3] = rho_flat * u_flat
        m_eq[4] = -rho_flat * u_flat
        m_eq[5] = rho_flat * v_flat
        m_eq[6] = -rho_flat * v_flat
        m_eq[7] = rho_flat * (u_flat ** 2 - v_flat ** 2)
        m_eq[8] = rho_flat * (u_flat * v_flat)

        # Relaxation in moment space: m^* = m - S * (m - m_eq)
        m_post = m - self.S[:, None] * (m - m_eq)

        # Transform back to velocity space: f^* = M^-1 @ m^*
        f_post = self.M_INV @ m_post

        # Add external forcing if present
        if force_x is not None or force_y is not None:
            fx = np.zeros_like(rho) if force_x is None else force_x
            fy = np.zeros_like(rho) if force_y is None else force_y
            omega = 1.0 / self._tau
            factor = 1.0 - 0.5 * omega
            for i in range(9):
                cx = D2Q9Lattice.C[i, 0]
                cy = D2Q9Lattice.C[i, 1]
                ci_dot_u = cx * u + cy * v
                term_x = (cx - u) * D2Q9Lattice.INV_CS2 + (ci_dot_u * cx) * (1.0 / D2Q9Lattice.CS4)
                term_y = (cy - v) * D2Q9Lattice.INV_CS2 + (ci_dot_u * cy) * (1.0 / D2Q9Lattice.CS4)
                source_i = factor * D2Q9Lattice.WEIGHTS[i] * (term_x * fx + term_y * fy)
                f_post[i] += source_i.reshape(-1)

        return f_post.reshape(9, ny, nx)
