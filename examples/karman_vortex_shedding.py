"""Example: 2D Karman vortex shedding around a cylinder obstacle."""

import argparse
import numpy as np
from chronolattice.collision import MRTCollision
from chronolattice.boundaries import (
    BounceBackBoundary,
    ZouHeVelocityInlet,
    ConvectiveOutlet,
    create_cylinder_obstacle,
    create_channel_walls,
)
from chronolattice.solver import LBMSolver


def run_karman(steps: int = 100):
    nx, ny = 120, 40
    u_in = 0.05
    cx, cy, r = nx / 4.0, ny / 2.0, ny / 10.0

    print(f"Initializing Karman vortex simulation ({nx}x{ny}, Re ~ 75)...")
    cylinder = create_cylinder_obstacle(nx, ny, cx, cy, r)
    walls = create_channel_walls(nx, ny)
    solid_mask = cylinder | walls

    # Enforce inflow/outflow first, then bounce-back on solid obstacles
    bcs = [
        ZouHeVelocityInlet(u_in=u_in),
        ConvectiveOutlet(),
        BounceBackBoundary(solid_mask),
    ]

    solver = LBMSolver(
        nx=nx,
        ny=ny,
        collision_op=MRTCollision(tau=0.60),
        boundaries=bcs,
        initial_u=u_in,
    )

    for s in range(1, steps + 1):
        solver.step()
        if s % 25 == 0 or s == steps:
            vort = solver.compute_vorticity()
            max_vort = float(np.max(np.abs(vort)))
            print(f"Step {s:4d} / {steps}: Peak Vorticity = {max_vort:.5f}, Total Energy = {solver.total_kinetic_energy:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    run_karman(steps=args.steps)
