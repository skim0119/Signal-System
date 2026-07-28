"""
This script demonstrates the simulation of a discrete state dynamic system,
specifically a Hidden Markov Model (HMM), and the application of filtering
techniques to estimate the hidden states based on observations. The script
initializes an HMM with specified transition and emission matrices, simulates
a single rollout of the system, and then applies a filtering algorithm to
estimate the hidden states from the observed data.
"""

from __future__ import annotations

from pathlib import Path

import click
import jax
import jax.numpy as jnp

from ss.estimation.filtering import HmmFilter, filtering
from ss.system import HiddenMarkovModel, simulate
from ss.utility.parameter.probability import ProbabilityParameter

from .post_processing import plot_filtering


@click.command()
@click.option(
    "--simulation-steps",
    type=click.IntRange(min=1),
    default=30,
    help="The simulation time steps.",
)
# @click.option(
#     "--step-skip",
#     type=click.IntRange(min=1),
#     default=1,
#     help="Subsample stride when reporting trajectories.",
# )
# @click.option(
#     "--state-dim",
#     type=click.IntRange(min=1),
#     default=3,
#     help="Number of discrete hidden states.",
# )
# @click.option(
#     "--discrete-observation-dim",
#     type=click.IntRange(min=1),
#     default=7,
#     help="Number of discrete observation symbols.",
# )
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=1,
    help="Batch size (positive integers).",
)
@click.option(
    "--random-seed",
    type=click.IntRange(min=0),
    default=2024,
    help="The random seed (non-negative integers).",
)
@click.option(
    "--save-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Save the filtering plot in this directory.",
)
def main(
    simulation_steps: int,
    # step_skip: int,
    # state_dim: int,
    # discrete_observation_dim: int,
    batch_size: int,
    random_seed: int,
    save_dir: Path | None,
) -> None:
    key = jax.random.PRNGKey(random_seed)
    filter_key, simulate_key = jax.random.split(key)
    # Set HMM system and simulate to get observationss

    # fmt: off
    # NOTE: hardcoded for demonstration purposes. Later, replace with some sort of
    # table generation with arbitrary state and observation dimensions.
    state_dim = 2
    discrete_observation_dim = 2
    transition_parameter = ProbabilityParameter(
        [[0.7, 0.3],
         [0.4, 0.6]]
    )
    emission_parameter = ProbabilityParameter(
        [[0.9, 0.1],
         [0.2, 0.8]]
    )
    # fmt: on

    system = HiddenMarkovModel(
        transition=transition_parameter,
        emission=emission_parameter,
        discrete_state_dim=state_dim,
        discrete_observation_dim=discrete_observation_dim,
        batch_size=batch_size,
    )

    initial_state = system.initial_state()
    times, states, observations, _ = simulate(
        system, 0.0, simulation_steps, initial_state, simulate_key
    )

    # Set HMM filter and apply filtering to get beliefs.
    filter = HmmFilter(
        transition=transition_parameter,
        emission=emission_parameter,
        discrete_state_dim=state_dim,
        discrete_observation_dim=discrete_observation_dim,
        state_dim=state_dim,
        batch_size=batch_size,
    )
    initial_belief = jnp.full((batch_size, filter.state_dim), 1.0 / state_dim)
    beliefs = filtering(filter, initial_belief, observations)

    print(f"system: {system}")
    print(f"filter: {filter}")
    print(f"states shape: {states.shape}")
    print(f"observations shape: {observations.shape}")
    print(f"beliefs shape: {beliefs.shape}")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "hmm_filter_plot.png"
        plot_filtering(
            times, states, observations, beliefs, save_path=save_path
        )


if __name__ == "__main__":
    main()
