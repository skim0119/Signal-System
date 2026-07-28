from pathlib import Path

import click
import jax
import jax.numpy as jnp

from ss.control import MPPIController
from ss.system import CartPoleSystem, simulate

from .cost import CostWeights
from .post_processing import (
    plot_simulation,
    render_animation,
)


@click.command()
@click.option(
    "--duration", type=click.FloatRange(min=0, min_open=True), default=5.0
)
@click.option(
    "--time-step", type=click.FloatRange(min=0, min_open=True), default=0.02
)
@click.option("--horizon", type=click.IntRange(min=1), default=60)
@click.option("--num-samples", type=click.IntRange(min=1), default=1024)
@click.option("--batch-size", type=click.IntRange(min=1), default=1)
@click.option(
    "--temperature",
    type=click.FloatRange(min=0, min_open=True),
    default=2.0,
)
@click.option(
    "--noise-sigma",
    type=click.FloatRange(min=0, min_open=True),
    default=10.0,
)
@click.option(
    "--control-limit",
    type=click.FloatRange(min=0, min_open=True),
    default=40.0,
)
@click.option("--initial-angle", type=float, default=0.8)
@click.option("--save-dir", type=click.Path(file_okay=False, path_type=Path))
def main(
    duration: float,
    time_step: float,
    horizon: int,
    num_samples: int,
    batch_size: int,
    temperature: float,
    noise_sigma: float,
    control_limit: float,
    initial_angle: float,
    save_dir: Path | None,
) -> None:
    num_steps = round(duration / time_step)
    system = CartPoleSystem(time_step=time_step, batch_size=batch_size)
    weights = CostWeights()

    random_key = jax.random.PRNGKey(0)
    state = system.initial_state()
    state = state.at[:, 2].add(initial_angle)
    controller = MPPIController(
        control_dim=system.control_dim,
        batch_size=system.batch_size,
        rollout_system=system,
        running_cost=weights.running_cost,
        terminal_cost=weights.terminal_cost,
        horizon=horizon,
        num_samples=num_samples,
        temperature=temperature,
        noise_sigma=noise_sigma,
        control_limit=control_limit,
    )
    times, states, _, controls = simulate(
        system,
        0.0,
        num_steps,
        state,
        random_key,
        controller=controller,
    )
    costs = weights.running_cost(states, controls)
    click.echo(f"final_state={states[-1]}")
    click.echo(f"total_running_cost={jnp.sum(costs, axis=0) * time_step}")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_simulation(
            times,
            states,
            controls,
            costs,
            save_dir / "mppi_cart_pole_plot.png",
        )
        render_animation(
            times, states, system.pole_length, save_dir / "mppi_cart_pole.mp4"
        )
        click.echo(f"Saved MPPI results to {save_dir}")


if __name__ == "__main__":
    main()
