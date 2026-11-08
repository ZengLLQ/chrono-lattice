"""Tests for D2Q9 lattice velocity sets, weights, and equilibrium moments."""

import pytest
import numpy as np
from chronolattice.lattice import D2Q9Lattice


def test_weights_sum_to_one():
    assert pytest.approx(np.sum(D2Q9Lattice.WEIGHTS)) == 1.0


def test_opposite_vectors():
    for i in range(9):
        opp = D2Q9Lattice.OPPOSITE[i]
        c_i = D2Q9Lattice.C[i]
        c_opp = D2Q9Lattice.C[opp]
        np.testing.assert_array_equal(c_i, -c_opp)


def test_equilibrium_moments():
    ny, nx = 4, 4
    rho = np.full((ny, nx), 1.25)
    u = np.full((ny, nx), 0.05)
    v = np.full((ny, nx), -0.02)

    f_eq = D2Q9Lattice.compute_equilibrium(rho, u, v)

    # 0th moment: Sum(f_eq) == rho
    rho_recovered = np.sum(f_eq, axis=0)
    np.testing.assert_allclose(rho_recovered, rho, rtol=1e-12)

    # 1st moment: Sum(f_eq * c) == rho * u
    rho_u_x = (
        f_eq[1] - f_eq[3] + f_eq[5] - f_eq[6] - f_eq[7] + f_eq[8]
    )
    rho_u_y = (
        f_eq[2] - f_eq[4] + f_eq[5] + f_eq[6] - f_eq[7] - f_eq[8]
    )
    np.testing.assert_allclose(rho_u_x, rho * u, rtol=1e-12)
    np.testing.assert_allclose(rho_u_y, rho * v, rtol=1e-12)


def test_compute_macroscopic():
    ny, nx = 5, 5
    rho_target = np.full((ny, nx), 2.0)
    u_target = np.full((ny, nx), 0.10)
    v_target = np.full((ny, nx), 0.05)

    f_eq = D2Q9Lattice.compute_equilibrium(rho_target, u_target, v_target)
    rho_rec, u_rec, v_rec = D2Q9Lattice.compute_macroscopic(f_eq)

    np.testing.assert_allclose(rho_rec, rho_target, rtol=1e-12)
    np.testing.assert_allclose(u_rec, u_target, rtol=1e-12)
    np.testing.assert_allclose(v_rec, v_target, rtol=1e-12)
