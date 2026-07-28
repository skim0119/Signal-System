"""
A discrete-state dynamical system, such as a Hidden Markov Model (HMM).
The state and observation are both discrete, and the system is defined
by a transition matrix and an emission matrix. The transition matrix defines
the probabilities of moving from one state to another, while the emission
matrix defines the probabilities of observing a particular observation given
the current state.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, Int, PRNGKeyArray

from ss.utility.parameter.probability import ProbabilityParameter

from ._system import DiscreteTimeSystem


type TransitionMatrix = Float[Array, "discrete_state_dim discrete_state_dim"]
type EmissionMatrix = Float[
    Array, "discrete_state_dim discrete_observation_dim"
]


class HiddenMarkovModel(DiscreteTimeSystem):
    """Discrete-state HMM with transition and emission probability matrices."""

    transition: ProbabilityParameter
    emission: ProbabilityParameter

    # NOTE: kw_only to enforce the user to explicitly write the dimension.
    discrete_state_dim: int = eqx.field(static=True, default=2, kw_only=True)
    discrete_observation_dim: int = eqx.field(
        static=True, default=2, kw_only=True
    )

    time_step: float = eqx.field(static=True, default=1.0, kw_only=True)
    state_dim: int = eqx.field(static=True, default=1, kw_only=True)
    observation_dim: int = eqx.field(static=True, default=1, kw_only=True)
    control_dim: int = eqx.field(static=True, default=0, kw_only=True)
    batch_size: int = eqx.field(static=True, default=1, kw_only=True)

    def __check_init__(self) -> None:
        super().__check_init__()
        transition_shape = self.transition_matrix.shape
        assert (
            transition_shape[0]
            == transition_shape[1]
            == self.discrete_state_dim
        ), (
            f"transition_matrix must be square "
            f"({self.discrete_state_dim}, {self.discrete_state_dim}), "
            f"got {transition_shape}"
        )
        emission_shape = self.emission_matrix.shape
        assert emission_shape[0] == self.discrete_state_dim, (
            f"emission_matrix must have {self.discrete_state_dim} rows, "
            f"got shape {emission_shape}"
        )
        assert emission_shape[1] == self.discrete_observation_dim, (
            f"emission_matrix must have {self.discrete_observation_dim} "
            f"columns, got shape {emission_shape}"
        )

    @property
    def transition_matrix(self) -> TransitionMatrix:
        return self.transition.value()

    @property
    def emission_matrix(self) -> EmissionMatrix:
        return self.emission.value()

    def with_transition_matrix(
        self,
        transition_matrix: TransitionMatrix,
    ) -> HiddenMarkovModel:
        expected_shape = self.transition_matrix.shape
        assert transition_matrix.shape == expected_shape, (
            f"transition_matrix must have shape {expected_shape}, "
            f"got {transition_matrix.shape}"
        )
        return eqx.tree_at(
            lambda model: model.transition,
            self,
            ProbabilityParameter(jnp.asarray(transition_matrix)),
        )

    def with_emission_matrix(
        self,
        emission_matrix: EmissionMatrix,
    ) -> HiddenMarkovModel:
        assert emission_matrix.shape == self.emission_matrix.shape, (
            f"emission_matrix must have shape {self.emission_matrix.shape}, "
            f"got {emission_matrix.shape}"
        )
        return eqx.tree_at(
            lambda model: model.emission,
            self,
            ProbabilityParameter(jnp.asarray(emission_matrix)),
        )

    def initial_state(
        self,
        random_key: PRNGKeyArray | None = None,
        initial_distribution: Array
        | None = None,  # FIXME: parameter overscoped.
    ) -> Int[Array, "batch_size state_dim"]:
        if random_key is None:
            random_key = jax.random.PRNGKey(42)
        if initial_distribution is None:
            initial_distribution = (
                jnp.ones(self.discrete_state_dim) / self.discrete_state_dim
            )

        logits = jnp.log(initial_distribution)
        return jax.random.categorical(
            random_key, logits, shape=(self.batch_size, 1)
        )

    def process(
        self,
        time: float,
        state: Int[Array, "batch_size"],
        control: Array | None,
        random_key: PRNGKeyArray,
    ) -> tuple[float, Int[Array, "batch_size state_dim"]]:
        return time + self.time_step, jax.random.categorical(
            random_key,
            jnp.log(self.transition_matrix[state]),
        )

    def observe(
        self,
        time: float,
        state: Int[Array, "batch_size"],
        random_key: PRNGKeyArray,
    ) -> Int[Array, "batch_size observation_dim"]:
        return jax.random.categorical(
            random_key,
            jnp.log(self.emission_matrix[state]),
        )
