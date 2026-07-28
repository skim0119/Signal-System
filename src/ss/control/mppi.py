from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ss.system import System

from ._control import Controller

type RolloutCarry = tuple[
    Float[Array, "batch_size num_samples"],  # times
    Float[Array, "batch_size num_samples state_dim"],  # states
    Float[Array, "batch_size num_samples"],  # accumulated costs
]
type RolloutInputs = tuple[
    Float[Array, "batch_size num_samples control_dim"],  # controls
    PRNGKeyArray,  # random key
]


class MPPIControllerState(eqx.Module):
    nominal_controls: Float[Array, "batch_size horizon control_dim"]


class MPPIDiagnostics(eqx.Module):
    minimum_rollout_cost: Float[Array, "batch_size"]
    mean_rollout_cost: Float[Array, "batch_size"]


class MPPIController(Controller):
    """Model Predictive Path Integral controller."""

    rollout_system: System

    running_cost: Callable[
        [
            Float[Array, "num_samples state_dim"],  # state
            Float[Array, "num_samples control_dim"],  # control
        ],
        Float[Array, "num_samples"],  # cost
    ] = eqx.field(static=True)
    terminal_cost: Callable[
        [Float[Array, "num_samples state_dim"]],  # state
        Float[Array, "num_samples"],  # cost
    ] = eqx.field(static=True)

    horizon: int = eqx.field(static=True, default=60)
    num_samples: int = eqx.field(static=True, default=1024)
    temperature: float = eqx.field(static=True, default=2.0)
    noise_sigma: float = eqx.field(static=True, default=10.0)
    control_limit: float = eqx.field(static=True, default=40.0)

    def __check_init__(self) -> None:
        super().__check_init__()
        object.__setattr__(
            self,
            "rollout_system",
            self.rollout_system.duplicate(batch_size=self.num_samples),
        )
        assert self.rollout_system.control_dim == self.control_dim
        assert self.horizon > 0
        assert self.num_samples > 0
        assert self.temperature > 0
        assert self.noise_sigma > 0
        assert self.control_limit > 0

    def initial_state(
        self, random_key: PRNGKeyArray | None = None
    ) -> MPPIControllerState:
        return MPPIControllerState(
            jnp.zeros((self.batch_size, self.horizon, self.control_dim))
        )

    def __call__(
        self,
        controller_state: MPPIControllerState,
        time: Float[Array, ""],
        observation: Float[Array, "batch_size observation_dim"],
        random_key: PRNGKeyArray,
    ) -> tuple[
        Float[Array, "batch_size control_dim"],  # control
        MPPIControllerState,  # next controller state
        MPPIDiagnostics,  # diagnostics
    ]:
        noise_key, rollout_key = jax.random.split(random_key)
        noise = self.noise_sigma * jax.random.normal(
            noise_key,
            (
                self.horizon,
                self.batch_size,
                self.num_samples,
                self.control_dim,
            ),
        )

        nominal_controls = jnp.swapaxes(  # scan need time-axis in the front
            controller_state.nominal_controls, 0, 1
        )
        sampled_controls = jnp.clip(
            nominal_controls[:, :, None, :] + noise,
            -self.control_limit,
            self.control_limit,
        )
        rollout_states = jnp.broadcast_to(
            observation[:, None, :],
            (
                self.batch_size,
                self.num_samples,
                observation.shape[-1],
            ),
        )
        rollout_times = jnp.full((self.batch_size, self.num_samples), time)
        rollout_keys = jax.random.split(
            rollout_key,
            (self.horizon, self.batch_size, self.num_samples),
        )

        def rollout_step(
            carry: RolloutCarry,
            inputs: RolloutInputs,
        ) -> tuple[RolloutCarry, None]:
            times, states, total_cost = carry
            controls, process_keys = inputs
            next_times, next_states = jax.vmap(self.rollout_system.process)(
                times,
                states,
                controls,
                process_keys,
            )
            total_cost += self.rollout_system.time_step * jax.vmap(
                self.running_cost
            )(next_states, controls)
            return (next_times, next_states, total_cost), None

        (_, final_states, costs), _ = jax.lax.scan(
            rollout_step,
            (
                rollout_times,
                rollout_states,
                jnp.zeros((self.batch_size, self.num_samples)),
            ),
            (sampled_controls, rollout_keys),
        )
        costs += jax.vmap(self.terminal_cost)(final_states)
        minimum_cost = jnp.min(costs, axis=-1)
        mean_cost = jnp.mean(costs, axis=-1)
        weights = jax.nn.softmax(
            -(costs - minimum_cost[:, None]) / self.temperature,
            axis=-1,
        )
        update = jnp.einsum("bs,hbsi->hbi", weights, noise)
        controls = jnp.clip(
            nominal_controls + update,
            -self.control_limit,
            self.control_limit,
        )
        shifted = jnp.concatenate(
            (controls[1:], jnp.zeros_like(controls[:1])), axis=0
        )
        return (
            controls[0],
            MPPIControllerState(jnp.swapaxes(shifted, 0, 1)),
            MPPIDiagnostics(minimum_cost, mean_cost),
        )
