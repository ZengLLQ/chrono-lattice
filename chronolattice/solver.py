"""Core Lattice Boltzmann Method (LBM) flow solver."""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple
from .lattice import D2Q9Lattice
from .collision import CollisionOperator, BGKCollision
from .boundaries import BoundaryCondition, BounceBackBoundary
from .shanchen import ShanChenMultiphase


class LBMSolver:
    """Manages lattice state, collision, streaming, forcing, and macroscopic diagnostics."""

    def __init__(
        self,
        nx: int,
        ny: int,
        collision_op: Optional[CollisionOperator] = None,
        boundaries: Optional[List[BoundaryCondition]] = None,
        multiphase: Optional[ShanChenMultiphase] = None,
        body_force_x: float = 0.0,
        body_force_y: float = 0.0,
        initial_rho: float = 1.0,
        initial_u: float = 0.0,
        initial_v: float = 0.0,
    ):
        self.nx = int(nx)
        self.ny = int(ny)
        self.collision_op = collision_op or BGKCollision(tau=1.0)
        self.boundaries = boundaries or []
        self.multiphase = multiphase

        self.body_force_x = float(body_force_x)
        self.body_force_y = float(body_force_y)

        # Initialize macroscopic fields
        self.rho = np.full((self.ny, self.nx), initial_rho, dtype=np.float64)
        self.u = np.full((self.ny, self.nx), initial_u, dtype=np.float64)
        self.v = np.full((self.ny, self.nx), initial_v, dtype=np.float64)

        # Initialize distribution to equilibrium
        self.f = D2Q9Lattice.compute_equilibrium(self.rho, self.u, self.v)
        self.step_count = 0

    def set_density_field(self, rho_field: np.ndarray) -> None:
        """Sets non-uniform initial density field and initializes distributions."""
        self.rho = np.asarray(rho_field, dtype=np.float64)
        self.f = D2Q9Lattice.compute_equilibrium(self.rho, self.u, self.v)

    def set_velocity_field(self, u_field: np.ndarray, v_field: np.ndarray) -> None:
        """Sets non-uniform initial velocity fields."""
        self.u = np.asarray(u_field, dtype=np.float64)
        self.v = np.asarray(v_field, dtype=np.float64)
        self.f = D2Q9Lattice.compute_equilibrium(self.rho, self.u, self.v)

    def compute_forces(self) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregates body force and Shan-Chen multiphase interaction force."""
        fx = np.full((self.ny, self.nx), self.body_force_x, dtype=np.float64)
        fy = np.full((self.ny, self.nx), self.body_force_y, dtype=np.float64)

        if self.multiphase is not None:
            f_sc_x, f_sc_y = self.multiphase.compute_interaction_force(self.rho)
            fx += f_sc_x
            fy += f_sc_y

        return fx, fy

    def step(self) -> None:
        """Executes a single LBM cycle: Forces -> Collision -> Streaming -> Boundaries -> Macro."""
        # 1. Total forces
        fx, fy = self.compute_forces()

        # 2. Macro recovery with force shift
        self.rho, self.u, self.v = D2Q9Lattice.compute_macroscopic(self.f, fx, fy)

        # 3. Collision: f -> f_post
        f_post = self.collision_op.collide(self.f, self.rho, self.u, self.v, fx, fy)

        # 4. Streaming: f_i(x + c_ix, y + c_iy) = f_i^*(x, y)
        f_streamed = np.zeros_like(self.f)
        for i in range(9):
            cx = D2Q9Lattice.C[i, 0]
            cy = D2Q9Lattice.C[i, 1]
            f_streamed[i] = np.roll(f_post[i], shift=(cy, cx), axis=(0, 1))

        # 5. Boundary conditions
        for bc in self.boundaries:
            bc.apply(f_streamed, f_post)

        self.f = f_streamed
        self.step_count += 1

        # Final macro update for diagnostics
        self.rho, self.u, self.v = D2Q9Lattice.compute_macroscopic(self.f, fx, fy)

    def run(self, steps: int) -> None:
        """Advances simulation by given number of time steps."""
        for _ in range(steps):
            self.step()

    @property
    def total_mass(self) -> float:
        """Total integrated mass across lattice domain."""
        return float(np.sum(self.rho))

    @property
    def total_kinetic_energy(self) -> float:
        r"""Total integrated kinetic energy: 0.5 * \sum \rho (u^2 + v^2)."""
        return float(0.5 * np.sum(self.rho * (self.u ** 2 + self.v ** 2)))

    def compute_vorticity(self) -> np.ndarray:
        r"""Recovers 2D curl / vorticity field: \omega = \partial v / \partial x - \partial u / \partial y."""
        dv_dx = np.gradient(self.v, axis=1)
        du_dy = np.gradient(self.u, axis=0)
        return dv_dx - du_dy
