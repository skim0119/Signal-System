from pathlib import Path

import click
import jax

from ss.system import CartPoleSystem, simulate

from .post_processing import plot_simulation, render_animation


@click.command()
@click.option(
    "--duration",
    type=click.FloatRange(min=0, min_open=True),
    default=10.0,
    help="Set the duration to run system (positive value).",
)
@click.option(
    "--cart-mass",
    type=click.FloatRange(min=0, min_open=True),
    default=1.0,
    help="Set the mass of the cart (positive value).",
)
@click.option(
    "--pole-mass",
    type=click.FloatRange(min=0, min_open=True),
    default=0.01,
    help="Set the mass of the pole (positive value).",
)
@click.option(
    "--pole-length",
    type=click.FloatRange(min=0, min_open=True),
    default=2.0,
    help="Set the length of the pole (positive value).",
)
@click.option(
    "--gravity",
    type=click.FloatRange(min=0),
    default=9.81,
    help="Set the value of gravity (positive value).",
)
@click.option(
    "--time-step",
    type=click.FloatRange(min=0, min_open=True),
    default=0.01,
    help="Set the time step (positive value).",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=1,
    help="Set the batch size (positive integers).",
)
@click.option(
    "--save-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Save the trajectory plot and MP4 animation in this directory.",
)
def main(
    duration: float,
    cart_mass: float,
    pole_mass: float,
    pole_length: float,
    gravity: float,
    time_step: float,
    batch_size: int,
    save_dir: Path | None,
) -> None:
    num_steps = round(duration / time_step)
    system = CartPoleSystem(
        cart_mass=cart_mass,
        pole_mass=pole_mass,
        pole_length=pole_length,
        gravity=gravity,
        time_step=time_step,
        batch_size=batch_size,
    )
    random_key = jax.random.PRNGKey(0)

    initial_state = system.initial_state()

    times, states, _, _ = simulate(
        system,
        0.0,
        num_steps,
        initial_state,
        random_key,
    )

    print(f"final_state={states[-1]}")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

        plot_path = save_dir / "cart_pole_plot.png"
        animation_path = save_dir / "cart_pole_animation.mp4"

        plot_simulation(times, states, save_path=plot_path)
        render_animation(times, states, pole_length, animation_path)

        click.echo(f"Saved plot to {plot_path}")
        click.echo(f"Saved animation to {animation_path}")


if __name__ == "__main__":
    main()
