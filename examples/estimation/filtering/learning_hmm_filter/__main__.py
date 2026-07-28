"""
This example demonstrates how to learn the parameters of a Hidden Markov Model
(HMM) using the cross-entropy loss function. The code initializes a random HMM
system, simulates observations, and trains a filter to learn the transition
and emission matrices from the simulated data.
"""

from __future__ import annotations

import optax
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from ss.estimation.filtering import HmmFilter, filtering
from ss.utility.learning import LearningProcess
from ss.system import HiddenMarkovModel, simulate
from ss.utility.parameter.probability import ProbabilityParameter


# random initialize transition and emission matrices (rows are distributions)
def random_stochastic_matrix(
    n_rows: int, n_cols: int, random_key: PRNGKeyArray
) -> Array:
    # sample each row from a symmetric Dirichlet
    keys = jax.random.split(random_key, n_rows)
    rows = [jax.random.dirichlet(k, jnp.ones(n_cols)) for k in keys]
    return jnp.stack(rows)


def cross_entropy_loss(
    model: HmmFilter, batch: Array, random_key: PRNGKeyArray | None
) -> Array:
    """Compute the cross-entropy loss between the predicted and true
    observation distributions for a batch of sequences.

    Args:
        model: The HMM model.
        batch: An array of shape
          ``(sequence_length+1, batch_size, observation_dim=1)`` containing
          the true observations.
        random_key: A random key for generating random numbers.
    Returns:
        The average cross-entropy loss over the batch.
    """
    assert random_key is not None, (
        "random_key must be provided for loss computation"
    )

    sequence_length_plus_one, batch_size, observation_dim = batch.shape
    assert batch_size == model.batch_size, (
        f"batch size {batch_size} must match filter.batch_size {model.batch_size}"
    )
    input_observations = batch[:-1, :, :]
    target_observations = batch[1:, :, :]

    initial_belief = jax.random.dirichlet(
        random_key, alpha=jnp.ones(model.state_dim), shape=(batch_size,)
    )

    beliefs = filtering(model, initial_belief, input_observations)

    # Compute the predicted observation distributions
    predicted_observation_distributions = jnp.einsum(
        "tbs,so->tbo", beliefs, model.emission_matrix
    )
    # TODO: Refactor to compute cross-entropy (-sum(targets * log(probs))).
    # Passing log(probs) into softmax_cross_entropy works mathematically, but
    # it causes redundant exp/log operations under the hood.
    # safe_probs = jnp.clip(predicted_observation_distributions, 1e-7, 1.0)
    pseudo_logits = jnp.log(predicted_observation_distributions)

    # Compute the cross-entropy loss
    losses = optax.losses.softmax_cross_entropy(
        pseudo_logits, target_observations
    )

    return jnp.mean(losses)


if __name__ == "__main__":
    print(
        "=== Discrete State Dynamic System Simulation: Hidden Markov Model ==="
    )

    random_key = jax.random.PRNGKey(0)
    key_sys, key_sim = jax.random.split(random_key)

    key_transition, key_emission = jax.random.split(key_sys)
    transition_matrix = random_stochastic_matrix(2, 2, key_transition)
    emission_matrix = random_stochastic_matrix(2, 2, key_emission)

    system = HiddenMarkovModel(
        transition=ProbabilityParameter(transition_matrix),
        emission=ProbabilityParameter(emission_matrix),
        discrete_state_dim=2,
        discrete_observation_dim=2,
    )

    print(f"System: {system}")

    random_key = jax.random.PRNGKey(0)

    print("=== single rollout ===")
    time_horizon = 10

    initial_state = system.initial_state(random_key)

    times, states, observations, _ = simulate(
        system, 0, time_horizon, initial_state, random_key
    )

    print(states)
    print(observations)

    # random initialize filter parameters (independent of system)
    key_filter = jax.random.PRNGKey(0)
    key_transition, key_emission = jax.random.split(key_filter)
    transition_matrix = random_stochastic_matrix(2, 2, key_transition)
    emission_matrix = random_stochastic_matrix(2, 2, key_emission)

    filter = HmmFilter(
        transition=ProbabilityParameter(transition_matrix),
        emission=ProbabilityParameter(emission_matrix),
        discrete_state_dim=2,
        discrete_observation_dim=2,
        state_dim=2,
    )
    print("=== filtering ===")
    print(f"filter: {filter}")
    print(filter.transition_matrix)

    learning_process = LearningProcess[HmmFilter](
        filter, cross_entropy_loss, optax.adam(1e-2)
    )

    learning_process.train_one_epoch(
        # (time, batch, observation_dim)
        training_data_loader=[observations for _ in range(10)],
        validation_data_loader=None,
        random_key=jax.random.PRNGKey(0),
    )

    learned_filter = learning_process.model

    print("=== learned filter ===")
    print(learned_filter.transition_matrix)
