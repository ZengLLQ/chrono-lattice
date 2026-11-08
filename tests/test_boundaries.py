"""Tests for LBM boundary condition routines."""

import pytest
import numpy as np
from chronolattice.lattice import D2Q9Lattice
from chronolattice.boundaries import (
    BounceBackBoundary,
    ZouHeVelocityInlet,
    create_cylinder_obstacle,
    create_channel_walls,
)


def test_create_masks():
    cyl = create_cylinder_obstacle(nx=50, ny=50, cx=25, cy=25, radius=5)
    assert cyl[25, 25] is np.True_ or cyl[25, 25] == True
    assert cyl[0, 0] is np.False_ or cyl[0, 0] == False

    walls = create_channel_walls(nx=30, ny=20)
    assert np.all(walls[0, :])
    assert np.all(walls[-1, :])
    assert not np.any(walls[1:-1, :])


def test_bounce_back_reflection():
    ny, nx = 5, 5
    solid_mask = np.zeros((ny, nx), dtype=bool)
    solid_mask[2, 2] = True

    f_prev = np.zeros((9, ny, nx))
    f_prev[1, 2, 2] = 10.0  # Moving East
    f_curr = np.zeros((9, ny, nx))

    bb = BounceBackBoundary(solid_mask)
    bb.apply(f_curr, f_prev)

    # Direction 1 (East) opposite is direction 3 (West)
    assert f_curr[3, 2, 2] == 10.0
    assert f_curr[1, 2, 2] == 0.0


def test_zou_he_velocity_inlet():
    ny, nx = 8, 8
    f = np.full((9, ny, nx), 0.1)
    f_prev = f.copy()

    u_in = 0.05
    v_in = 0.0
    zh = ZouHeVelocityInlet(u_in=u_in, v_in=v_in)
    zh.apply(f, f_prev)

    # Macroscopic velocity on inlet must equal target
    rho, u, v = D2Q9Lattice.compute_macroscopic(f)
    np.testing.assert_allclose(u[:, 0], u_in, rtol=1e-3)
    np.testing.assert_allclose(v[:, 0], v_in, atol=1e-5)
