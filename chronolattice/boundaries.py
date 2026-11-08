"""Boundary condition implementations for Lattice Boltzmann simulations."""

from __future__ import annotations
import abc
import numpy as np
from .lattice import D2Q9Lattice


class BoundaryCondition(abc.ABC):
    """Abstract base class for LBM boundary conditions."""

    @abc.abstractmethod
    def apply(self, f: np.ndarray, f_prev: np.ndarray) -> None:
        """Applies boundary conditions in-place to distribution array f."""
        pass


class BounceBackBoundary(BoundaryCondition):
    """Half-way no-slip bounce-back boundary on arbitrary solid geometry."""

    def __init__(self, solid_mask: np.ndarray):
        self.solid_mask = np.asarray(solid_mask, dtype=bool)

    def apply(self, f: np.ndarray, f_prev: np.ndarray) -> None:
        """Reflects streaming distributions back in reverse direction on solid nodes."""
        if not np.any(self.solid_mask):
            return

        for i in range(9):
            opp = D2Q9Lattice.OPPOSITE[i]
            # Replace incoming direction on solid node with previous outgoing direction
            f[i, self.solid_mask] = f_prev[opp, self.solid_mask]


class ZouHeVelocityInlet(BoundaryCondition):
    """Zou-He velocity boundary condition on western inlet boundary (x = 0)."""

    def __init__(self, u_in: float, v_in: float = 0.0):
        self.u_in = float(u_in)
        self.v_in = float(v_in)

    def apply(self, f: np.ndarray, f_prev: np.ndarray) -> None:
        u = self.u_in
        v = self.v_in

        # Local inlet line x = 0
        f0 = f[0, :, 0]
        f2 = f[2, :, 0]
        f3 = f[3, :, 0]
        f4 = f[4, :, 0]
        f6 = f[6, :, 0]
        f7 = f[7, :, 0]

        # Inflow density calculation
        rho_in = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - u)

        # Reconstruct unknown incoming populations (1, 5, 8)
        f[1, :, 0] = f3 + (2.0 / 3.0) * rho_in * u
        f[5, :, 0] = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_in * u + 0.5 * rho_in * v
        f[8, :, 0] = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_in * u - 0.5 * rho_in * v


class ConvectiveOutlet(BoundaryCondition):
    """Zero-gradient convective outlet boundary on eastern boundary (x = -1)."""

    def apply(self, f: np.ndarray, f_prev: np.ndarray) -> None:
        # Copy distribution from adjacent internal slice
        f[:, :, -1] = f[:, :, -2]


def create_cylinder_obstacle(nx: int, ny: int, cx: float, cy: float, radius: float) -> np.ndarray:
    """Generates a boolean 2D mask with a circular obstacle."""
    y, x = np.ogrid[:ny, :nx]
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
    return dist_sq <= (radius ** 2)


def create_channel_walls(nx: int, ny: int) -> np.ndarray:
    """Generates no-slip boundary mask for top and bottom channel walls."""
    mask = np.zeros((ny, nx), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    return mask
