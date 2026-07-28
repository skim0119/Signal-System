"""Hidden Markov Model filter."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ss.utility.parameter.probability import ProbabilityParameter

from ._filtering import Filter


class HmmFilter(Filter):
    """HMM belief filter over discrete states and observations."""

    transition: ProbabilityParameter
    emission: ProbabilityParameter
    state_dim: int = eqx.field(static=True, kw_only=True)
    batch_size: int = eqx.field(static=True, default=1, kw_only=True)

    # NOTE: kw_only to enforce the user to explicitly write the dimension
    time_step: float = eqx.field(static=True, default=1.0, kw_only=True)
    observation_dim: int = eqx.field(static=True, default=1, kw_only=True)
    control_dim: int = eqx.field(static=True, default=0, kw_only=True)

    def __check_init__(self) -> None:
        super().__check_init__()
        transition_shape = self.transition_matrix.shape
        assert self.state_dim == transition_shape[0], (
            f"state_dim {self.state_dim} must match transition state dimension "
            f"{transition_shape[0]}"
        )
        emission_shape = self.emission_matrix.shape
        assert emission_shape[0] == transition_shape[0], (
            f"emission_matrix must have {transition_shape[0]} rows, "
            f"got shape {emission_shape}"
        )

    @property
    def discrete_state_dim(self) -> int:
        return self.transition_matrix.shape[0]

    @property
    def discrete_observation_dim(self) -> int:
        return self.emission_matrix.shape[1]

    @property
    def transition_matrix(self) -> Float[Array, "state_dim state_dim"]:
        return self.transition.value()

    @property
    def emission_matrix(self) -> Float[Array, "state_dim observation_dim"]:
        return self.emission.value()

    def with_transition_matrix(
        self,
        transition_matrix: Array,
    ) -> HmmFilter:
        assert transition_matrix.shape == self.transition_matrix.shape, (
            "transition_matrix must have shape "
            f"{self.transition_matrix.shape}, got {transition_matrix.shape}"
        )
        return eqx.tree_at(
            lambda model: model.transition,
            self,
            ProbabilityParameter(jnp.asarray(transition_matrix)),
        )

    def with_emission_matrix(
        self,
        emission_matrix: Array,
    ) -> HmmFilter:
        assert emission_matrix.shape == self.emission_matrix.shape, (
            f"emission_matrix must have shape {self.emission_matrix.shape}, "
            f"got {emission_matrix.shape}"
        )
        return eqx.tree_at(
            lambda model: model.emission,
            self,
            ProbabilityParameter(jnp.asarray(emission_matrix)),
        )

    def update(
        self,
        prior: Float[Array, "batch_size state_dim"],
        observation: Int[Array, "batch_size observation_dim"],
    ) -> Float[Array, "batch_size state_dim"]:
        """Return the filtered posterior given prior and observation."""
        # emission columns for each batch observation: (batch, state)
        likelihood = self.emission_matrix[:, observation[:, 0]].T
        updated = prior * likelihood
        return updated / jnp.sum(updated, axis=-1, keepdims=True)

    def predict(
        self,
        posterior: Float[Array, "batch_size state_dim"],
    ) -> Float[Array, "batch_size state_dim"]:
        """Chapman–Kolmogorov prediction step."""
        return posterior @ self.transition_matrix
