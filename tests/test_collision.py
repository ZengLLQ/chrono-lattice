"""Tests for BGK and MRT collision operators."""

import pytest
import numpy as np
from chronolattice.lattice import D2Q9Lattice
from chronolattice.collision import BGKCollision, MRTCollision


def test_bgk_collision_conservation():
    tau = 0.8
    bgk = BGKCollision(tau=tau)
    assert pytest.approx(bgk.kinematic_viscosity) == (2.0 * 0.8 - 1.0) / 6.0

    ny, nx = 6, 6
    rho = np.full((ny, nx), 1.0)
    u = np.full((ny, nx), 0.05)
    v = np.full((ny, nx), 0.02)

    # Start with arbitrary non-equilibrium perturbation
    f = D2Q9Lattice.compute_equilibrium(rho, u, v)
    f[1] += 0.01
    f[3] -= 0.01

    f_post = bgk.collide(f, rho, u, v)

    # Mass must be conserved
    rho_post = np.sum(f_post, axis=0)
    np.testing.assert_allclose(rho_post, rho, rtol=1e-12)


def test_mrt_collision_conservation():
    tau = 0.65
    mrt = MRTCollision(tau=tau)

    ny, nx = 8, 8
    rho = np.full((ny, nx), 1.1)
    u = np.full((ny, nx), -0.04)
    v = np.full((ny, nx), 0.03)

    f = D2Q9Lattice.compute_equilibrium(rho, u, v)
    f[2] += 0.02
    f[4] -= 0.02

    f_post = mrt.collide(f, rho, u, v)

    rho_post = np.sum(f_post, axis=0)
    np.testing.assert_allclose(rho_post, rho, rtol=1e-12)


def test_invalid_tau_rejection():
    with pytest.raises(ValueError, match="strictly > 0.5"):
        BGKCollision(tau=0.5)

    with pytest.raises(ValueError, match="strictly > 0.5"):
        MRTCollision(tau=0.45)
