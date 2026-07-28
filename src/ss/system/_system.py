"""Framework for simulating continuous and discrete-time systems."""

from __future__ import annotations

from abc import abstractmethod
from copy import copy
from typing import TYPE_CHECKING, TypeVar, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, Shaped

if TYPE_CHECKING:
    from ss.control._control import Controller, ControllerState


class System(eqx.Module):
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
        # TODO: This is temporary duplication for different batch size.
        # Mostly used for testing and demonstration, but may need to change the
        # syntax in the future.
        if batch_size is None:
            batch_size = self.batch_size
        assert batch_size > 0, f"batch_size {batch_size} must be > 0"
        duplicate = copy(self)
        object.__setattr__(duplicate, "batch_size", batch_size)
        return duplicate

    @abstractmethod
    def initial_state(
        self, random_key: PRNGKeyArray | None = None
    ) -> Float[Array, "batch_size state_dim"]:
        pass

    @abstractmethod
    def process(
        self,
        time: float,
        state: Float[Array, "batch_size state_dim"],
        control: Float[Array, "batch_size control_dim"] | None,
        random_key: PRNGKeyArray,
    ) -> tuple[float, Float[Array, "batch_size state_dim"]]:
        pass

    @abstractmethod
    def observe(
        self,
        time: float,
        state: Float[Array, "batch_size state_dim"],
        random_key: PRNGKeyArray,
    ) -> Float[Array, "batch_size observation_dim"]:
        pass


class ContinuousTimeSystem(System):
    process_noise_covariance: Float[Array, "state_dim state_dim"]
    observation_noise_covariance: Float[
        Array, "observation_dim observation_dim"
    ]

    def __check_init__(self) -> None:
        super().__check_init__()
        s = (self.state_dim, self.state_dim)
        o = (self.observation_dim, self.observation_dim)
        assert self.process_noise_covariance.shape == s, (
            f"process_noise_covariance must have shape {s}, got "
            f"{self.process_noise_covariance.shape}"
        )
        assert self.observation_noise_covariance.shape == o, (
            f"observation_noise_covariance must have shape {o}, got "
            f"{self.observation_noise_covariance.shape}"
        )

    def _process_noise(
        self, time: float, state: Array, random_key: PRNGKeyArray
    ) -> Array:
        cov = self.process_noise_covariance * jnp.sqrt(self.time_step)
        # multivariate_normal requires positive-definite covariance; when
        # covariance is identically zero (no noise requested) skip sampling
        # entirely rather than producing NaN.
        return jax.lax.cond(
            jnp.all(cov == 0),
            lambda: jnp.zeros(self.state_dim),
            lambda: jax.random.multivariate_normal(
                random_key, jnp.zeros(self.state_dim), cov
            ),
        )

    def _observation_noise(
        self, time: float, state: Array, random_key: PRNGKeyArray
    ) -> Array:
        cov = self.observation_noise_covariance * jnp.sqrt(self.time_step)
        return jax.lax.cond(
            jnp.all(cov == 0),
            lambda: jnp.zeros(self.observation_dim),
            lambda: jax.random.multivariate_normal(
                random_key, jnp.zeros(self.observation_dim), cov
            ),
        )


class DiscreteTimeSystem(System):
    discrete_state_dim: int = eqx.field(static=True)
    discrete_observation_dim: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        super().__check_init__()
        assert self.time_step == 1, (
            "DiscreteTimeSystem requires time_step == 1"
        )

    def state_one_hot(
        self, state: Int[Array, "batch_size discrete_state_dim"]
    ) -> Array:
        return jax.nn.one_hot(state, self.discrete_state_dim)

    def observation_one_hot(
        self, observation: Int[Array, "batch_size discrete_observation_dim"]
    ) -> Array:
        return jax.nn.one_hot(observation, self.discrete_observation_dim)


SystemT = TypeVar("SystemT", bound=System)


def simulate(
    system: SystemT,
    initial_time: float,
    number_of_steps: int,
    initial_state: Shaped[Array, "batch_size ..."],
    random_key: PRNGKeyArray | None = None,
    controller: Controller | None = None,
) -> tuple[
    Float[Array, "time"],  # times
    Shaped[Array, "time batch_size ..."],  # states
    Shaped[Array, "time batch_size observation_dim"],  # observations
    Float[Array, "time batch_size control_dim"] | None,  # controls
]:
    """Simulate batches with a time scan containing system steps.

    Layout matches ``filtering`` / ``lax.scan``: time axis first, then batch.
    Dim names come from ``SystemT`` (``batch_size``, ``observation_dim``,
    ``control_dim``); state trailing dims vary by concrete system.

    Args:
        system: System to simulate.
        initial_time: Time before the first step.
        number_of_steps: Number of observe/process steps to run.
        initial_state: Initial system state, shape ``(batch_size, ...)``.
        random_key: PRNG key. If ``None``, a fixed default key is used.
        controller: Optional controller applied each step.

    Returns:
        ``(times, states, observations, controls)`` with leading time axis of
        length ``number_of_steps``. ``controls`` is ``None`` if no controller.
    """
    assert number_of_steps > 0, (
        f"number_of_steps {number_of_steps} must be > 0"
    )
    if random_key is None:
        random_key = jax.random.PRNGKey(43)

    if controller is not None:
        assert system.batch_size == controller.batch_size, (
            f"system.batch_size {system.batch_size} must match "
            f"controller.batch_size {controller.batch_size}"
        )
        random_key, controller_key = jax.random.split(random_key)
        controller_state = controller.initial_state(controller_key)
    else:
        controller_state = None

    keys = jax.random.split(random_key, number_of_steps)

    @jax.jit
    def body(
        carry: tuple[
            Float,  # time
            Array,  # system state
            ControllerState | None,  # controller state
        ],
        random_key: PRNGKeyArray,
    ) -> tuple[
        tuple[Float, Array, ControllerState | None],
        tuple[Float, Array, Array, Array | None],
    ]:
        previous_time, previous_state, controller_state = carry

        observe_key, process_key, controller_key = jax.random.split(
            random_key, 3
        )

        observation = system.observe(
            previous_time, previous_state, observe_key
        )

        if controller is None:
            control = None
            next_controller_state = None
            next_time, state = system.process(
                previous_time, previous_state, control, process_key
            )
        else:
            control, next_controller_state, _ = controller(
                controller_state,
                previous_time,
                observation,
                controller_key,
            )
            next_time, state = system.process(
                previous_time, previous_state, control, process_key
            )

        return (
            next_time,
            state,
            next_controller_state,
        ), (next_time, state, observation, control)

    _, (times, states, observations, controls) = jax.lax.scan(
        body,
        (initial_time, initial_state, controller_state),
        keys,
    )

    return times, states, observations, controls
