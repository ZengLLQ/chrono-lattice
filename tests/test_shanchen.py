"""Tests for Shan-Chen multiphase pseudopotential model."""

import pytest
import numpy as np
from chronolattice.shanchen import ShanChenMultiphase


def test_uniform_density_net_force_zero():
    """In a uniform density field, isotropic neighbor interactions must sum to zero net force."""
    sc = ShanChenMultiphase(g_interaction=-5.0, rho_0=1.0)
    ny, nx = 16, 16
    rho = np.full((ny, nx), 1.0)

    fx, fy = sc.compute_interaction_force(rho)

    np.testing.assert_allclose(fx, 0.0, atol=1e-12)
    np.testing.assert_allclose(fy, 0.0, atol=1e-12)


def test_pseudopotential_monotonicity():
    sc = ShanChenMultiphase(rho_0=1.0, psi_mode="exponential")
    rho_arr = np.linspace(0.1, 5.0, 50)
    psi_arr = sc.compute_pseudopotential(rho_arr)

    # Effective mass must increase monotonically with density
    diffs = np.diff(psi_arr)
    assert np.all(diffs > 0)


def test_equation_of_state_pressure():
    sc = ShanChenMultiphase(g_interaction=-6.0, rho_0=1.0)
    rho = np.array([[1.0, 2.0]])
    p = sc.compute_equation_of_state_pressure(rho)

    assert p.shape == (1, 2)
    assert np.all(np.isfinite(p))
