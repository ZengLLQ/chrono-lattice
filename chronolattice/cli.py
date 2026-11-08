"""Command line interface for ChronoLattice simulations."""

import click
import numpy as np
from .collision import BGKCollision, MRTCollision
from .boundaries import BounceBackBoundary, ZouHeVelocityInlet, ConvectiveOutlet, create_cylinder_obstacle, create_channel_walls
from .shanchen import ShanChenMultiphase
from .solver import LBMSolver


@click.group()
def cli():
    """ChronoLattice: High-Performance LBM Fluid & Multiphase Simulator CLI."""
    pass


@cli.command()
@click.option("--nx", default=100, help="Grid length in X.")
@click.option("--ny", default=40, help="Grid height in Y.")
@click.option("--tau", default=0.8, help="Relaxation time tau.")
@click.option("--body-force", default=1e-4, help="Driving body force in X direction.")
@click.option("--steps", default=500, help="Simulation steps.")
def poiseuille(nx: int, ny: int, tau: float, body_force: float, steps: int):
    """Simulates 2D channel Poiseuille flow between no-slip walls."""
    click.echo(f"Running Poiseuille channel flow: {nx}x{ny}, tau={tau}, force={body_force}, steps={steps}...")
    walls = create_channel_walls(nx, ny)
    bcs = [BounceBackBoundary(walls)]

    solver = LBMSolver(
        nx=nx,
        ny=ny,
        collision_op=BGKCollision(tau=tau),
        boundaries=bcs,
        body_force_x=body_force,
    )
    solver.run(steps)

    center_y = ny // 2
    u_max = float(np.max(solver.u))
    u_center = float(solver.u[center_y, nx // 2])
    kin_energy = solver.total_kinetic_energy

    click.echo(f"Simulation complete:")
    click.echo(f"  Max Velocity u_max:       {u_max:.6f}")
    click.echo(f"  Centerline Velocity:      {u_center:.6f}")
    click.echo(f"  Total Kinetic Energy:     {kin_energy:.6e}")
    click.echo(f"  Kinematic Viscosity nu:   {solver.collision_op.kinematic_viscosity:.4f}")


@cli.command()
@click.option("--nx", default=150, help="Domain width.")
@click.option("--ny", default=50, help="Domain height.")
@click.option("--u-in", default=0.05, help="Inlet flow velocity.")
@click.option("--steps", default=300, help="Simulation steps.")
def karman(nx: int, ny: int, u_in: float, steps: int):
    """Simulates 2D Karman vortex shedding around a cylinder obstacle."""
    click.echo(f"Simulating Karman vortex street around cylinder: {nx}x{ny}, u_in={u_in}, steps={steps}...")
    cx = nx / 4.0
    cy = ny / 2.0
    radius = ny / 10.0
    cylinder = create_cylinder_obstacle(nx, ny, cx, cy, radius)
    walls = create_channel_walls(nx, ny)
    solid_mask = cylinder | walls

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
    solver.run(steps)

    vort = solver.compute_vorticity()
    max_vort = float(np.max(np.abs(vort)))
    reynolds = (u_in * (2.0 * radius)) / solver.collision_op.kinematic_viscosity

    click.echo(f"Karman Simulation Finished:")
    click.echo(f"  Reynolds Number Re:       {reynolds:.2f}")
    click.echo(f"  Peak Vorticity |omega|:   {max_vort:.4f}")
    click.echo(f"  Total Domain Mass:        {solver.total_mass:.2f}")


@cli.command()
@click.option("--nx", default=64, help="Domain size X.")
@click.option("--ny", default=64, help="Domain size Y.")
@click.option("--g-interaction", default=-5.5, help="Shan-Chen interaction strength.")
@click.option("--steps", default=200, help="Simulation steps.")
def multiphase(nx: int, ny: int, g_interaction: float, steps: int):
    """Simulates Shan-Chen spontaneous liquid-gas phase separation."""
    click.echo(f"Running Shan-Chen phase separation: {nx}x{ny}, G={g_interaction}, steps={steps}...")
    rng = np.random.default_rng(42)
    # Mean density 1.0 with small random perturbations
    rho_init = 1.0 + 0.05 * (rng.random((ny, nx)) - 0.5)

    multiphase_model = ShanChenMultiphase(g_interaction=g_interaction, rho_0=1.0)
    solver = LBMSolver(
        nx=nx,
        ny=ny,
        collision_op=BGKCollision(tau=1.0),
        multiphase=multiphase_model,
    )
    solver.set_density_field(rho_init)
    solver.run(steps)

    rho_min = float(np.min(solver.rho))
    rho_max = float(np.max(solver.rho))
    click.echo(f"Phase separation complete:")
    click.echo(f"  Vapor Phase Density:      {rho_min:.4f}")
    click.echo(f"  Liquid Phase Density:     {rho_max:.4f}")
    click.echo(f"  Density Ratio:            {(rho_max / rho_min):.2f}")


if __name__ == "__main__":
    cli()
