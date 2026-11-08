"""Example: Multiphase spinodal decomposition via Shan-Chen pseudopotential."""

import argparse
import numpy as np
from chronolattice.collision import BGKCollision
from chronolattice.shanchen import ShanChenMultiphase
from chronolattice.solver import LBMSolver


def run_spinodal(steps: int = 100):
    nx, ny = 64, 64
    g_interaction = -5.5
    print(f"Initializing Shan-Chen multiphase phase separation ({nx}x{ny}, G={g_interaction})...")

    rng = np.random.default_rng(1337)
    rho_init = 1.0 + 0.08 * (rng.random((ny, nx)) - 0.5)

    multiphase_model = ShanChenMultiphase(g_interaction=g_interaction, rho_0=1.0)
    solver = LBMSolver(
        nx=nx,
        ny=ny,
        collision_op=BGKCollision(tau=1.0),
        multiphase=multiphase_model,
    )
    solver.set_density_field(rho_init)

    for s in range(1, steps + 1):
        solver.step()
        if s % 25 == 0 or s == steps:
            r_min = float(np.min(solver.rho))
            r_max = float(np.max(solver.rho))
            print(f"Step {s:4d} / {steps}: Vapor Density = {r_min:.4f}, Liquid Density = {r_max:.4f}, Ratio = {(r_max/r_min):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    run_spinodal(steps=args.steps)
