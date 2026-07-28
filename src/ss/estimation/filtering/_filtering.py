from __future__ import annotations

from abc import abstractmethod
from copy import copy
from typing import Self, TypeVar

import equinox as eqx
import jax
from jaxtyping import Array, Float, Shaped


class Filter(eqx.Module):
    time_step: float = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    observation_dim: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        assert self.time_step >= 0, f"time_step {self.time_step} must be >= 0"
        assert self.state_dim > 0, f"state_dim {self.state_dim} must be > 0"
        assert self.observation_dim > 0, (
            f"observation_dim {self.observation_dim} must be > 0"
        )
        assert self.control_dim >= 0, (
            f"control_dim {self.control_dim} must be >= 0"
        )
        assert self.batch_size > 0, f"batch_size {self.batch_size} must be > 0"

    def duplicate(self, *, batch_size: int | None = None) -> Self:
        """Return an immutable copy configured for a new batch size."""
        if batch_size is None:
            batch_size = self.batch_size
        assert batch_size > 0, f"batch_size {batch_size} must be > 0"
        duplicate = copy(self)
        object.__setattr__(duplicate, "batch_size", batch_size)
        return duplicate

    @abstractmethod
    def update(
        self,
        prior: Float[Array, "batch_size state_dim"],
        observation: Float[Array, "batch_size observation_dim"],
    ) -> Float[Array, "batch_size state_dim"]:
        """Return filtered posterior given prior and observation.

        This is the only required filtering kernel. Some filters naturally
        combine update and prediction into one operation; those filters
        can keep the default identity implementation of :meth:`predict`.
        """

    def predict(
        self,
        posterior: Float[Array, "batch_size state_dim"],
    ) -> Float[Array, "batch_size state_dim"]:
        """Predict next-step prior from posterior.

        Why this is not abstract:
            Not every filter has a separate prediction step. For update-only
            filters (or filters that fold dynamics into :meth:`update`),
            forcing a `predict` override only adds boilerplate identity code.

        When to override:
            Override this method when your filter has explicit transition
            dynamics, e.g. a Chapman-Kolmogorov step for HMMs or model-based
            temporal propagation.

        Default behavior:
            Identity map, so ``prior_{t+1} = posterior_t``.
        """
        return posterior


FilterT = TypeVar("FilterT", bound=Filter)


def filtering(
    filter: FilterT,
    initial_belief: Float[Array, "batch_size state_dim"],
    observations: Shaped[Array, "time batch_size observation_dim"],
) -> Float[Array, "time batch_size state_dim"]:
    """Run filtering over a time-leading observation sequence.

    Layout matches ``simulate`` / ``lax.scan``: time axis first, then batch.
    Returns filtered beliefs ``p(x_t | y_{1:t})``.

    Args:
        filter: The filter to use for the filtering process.
        initial_belief: Prior before the first observation,
            shape ``(batch_size, state_dim)``.
        observations: Observations with shape
            ``(time, batch_size, observation_dim)``.

    Returns:
        Filtered beliefs, shape ``(time, batch_size, state_dim)``.
    """

    def step(
        prior: Float[Array, "batch_size state_dim"],
        observation: Float[Array, "batch_size observation_dim"],
    ) -> tuple[
        Float[Array, "batch_size state_dim"],  # next prior (scan carry)
        Float[
            Array, "batch_size state_dim"
        ],  # filtered posterior (scan output)
    ]:
        posterior = filter.update(prior, observation)
        return filter.predict(posterior), posterior

    _, beliefs = jax.lax.scan(step, initial_belief, observations)
    return beliefs
