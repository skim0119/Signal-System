"""
A discrete-state dynamical system, such as a Hidden Markov Model (HMM).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ss.system import HiddenMarkovModel, simulate
from ss.utility.parameter.probability import ProbabilityParameter

if __name__ == "__main__":
    print(
        "=== Discrete State Dynamic System Simulation: Hidden Markov Model ==="
    )

    transition = jnp.array([[0.7, 0.3], [0.4, 0.6]])
    emission = jnp.array([[0.8, 0.2], [0.2, 0.8]])
    system = HiddenMarkovModel(
        transition=ProbabilityParameter(transition),
        emission=ProbabilityParameter(emission),
        discrete_state_dim=2,
        discrete_observation_dim=2,
    )

    print("transition_matrix:\n", system.transition_matrix)
    print("emission_matrix:\n", system.emission_matrix)

    system = system.with_transition_matrix(jnp.array([[0.6, 0.4], [0.5, 0.5]]))
    print("updated transition_matrix:\n", system.transition_matrix)

    print("=== single rollout ===")
    time_horizon = 100

    random_key = jax.random.PRNGKey(0)

    initial_state = system.initial_state(random_key)

    times, states, observations, _ = simulate(
        system, 0, time_horizon, initial_state, random_key
    )

    print("times:", times.shape)
    print("states:", states.shape)
    print("observations:", observations.shape)

    print("=== batched rollout ===")
    batch_size = 15
    systems = system.duplicate(batch_size=batch_size)

    random_key = jax.random.PRNGKey(0)
    initial_state_key, random_key = jax.random.split(random_key)
    init_states = systems.initial_state(initial_state_key)

    batch_times, batch_states, batch_observations, _ = simulate(
        systems, 0, time_horizon, init_states, random_key
    )

    print("batch_times shape:", batch_times.shape)
    print("batch_states shape:", batch_states.shape)
    print("batch_observations shape:", batch_observations.shape)
