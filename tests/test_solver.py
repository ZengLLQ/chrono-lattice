"""Tests for full LBMSolver execution, mass conservation, and fluid flow."""

import pytest
import numpy as np
from chronolattice.collision import BGKCollision
from chronolattice.boundaries import BounceBackBoundary, create_channel_walls
from chronolattice.solver import LBMSolver


def test_mass_conservation_periodic():
    """In a closed periodic domain without source/sink, mass must be conserved to floating-point precision."""
    nx, ny = 20, 20
    solver = LBMSolver(
        nx=nx,
        ny=ny,
        collision_op=BGKCollision(tau=0.8),
        initial_rho=1.5,
        initial_u=0.02,
        initial_v=-0.01,
    )
    initial_mass = solver.total_mass

    solver.run(steps=50)

    final_mass = solver.total_mass
    assert pytest.approx(final_mass, rel=1e-10) == initial_mass


def test_poiseuille_channel_flow():
    """Body-force driven channel flow between no-slip walls must develop a symmetric velocity profile."""
    nx, ny = 30, 21
    walls = create_channel_walls(nx, ny)
    bcs = [BounceBackBoundary(walls)]

    solver = LBMSolver(
        nx=nx,
        ny=ny,
        collision_op=BGKCollision(tau=0.9),
        boundaries=bcs,
        body_force_x=1e-4,
    )
    solver.run(steps=150)

    center_y = ny // 2
    u_profile = solver.u[:, nx // 2]

    # Centerline velocity must be positive and greater than near-wall velocity
    assert u_profile[center_y] > 0.0
    assert u_profile[center_y] > u_profile[1]
    assert u_profile[center_y] > u_profile[-2]

    # Symmetric profile u(y) == u(H - y)
    np.testing.assert_allclose(u_profile[1:center_y], u_profile[-2:center_y:-1], rtol=0.05)


def test_vorticity_calculation():
    nx, ny = 16, 16
    solver = LBMSolver(nx=nx, ny=ny)
    # Set shear velocity field: u(y) = y, v = 0 -> dv/dx - du/dy = -1
    y, x = np.mgrid[:ny, :nx]
    solver.u = y.astype(float) * 0.01
    solver.v = np.zeros((ny, nx))

    vort = solver.compute_vorticity()
    # In interior nodes, vorticity should be close to -0.01
    np.testing.assert_allclose(vort[2:-2, 2:-2], -0.01, rtol=1e-3)
