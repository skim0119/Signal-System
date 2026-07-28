import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ._system import System


class CartPoleSystem(System):
    """Batched cart-pole dynamics with an RK4 integration step."""

    time_step: float = eqx.field(static=True, default=0.001)
    state_dim: int = eqx.field(static=True, default=4)
    observation_dim: int = eqx.field(static=True, default=4)
    control_dim: int = eqx.field(static=True, default=1)
    batch_size: int = eqx.field(static=True, default=1)

    cart_mass: float = 1.0
    pole_mass: float = 0.01
    pole_length: float = 2.0
    gravity: float = 9.81

    def __check_init__(self) -> None:
        super().__check_init__()
        assert self.time_step > 0, "time_step must be > 0"
        assert self.cart_mass > 0, "cart_mass must be > 0"
        assert self.pole_mass > 0, "pole_mass must be > 0"
        assert self.pole_length > 0, "pole_length must be > 0"
        assert self.gravity >= 0, "gravity must be >= 0"
        assert self.batch_size > 0, "batch_size must be > 0"

    def initial_state(
        self, random_key: PRNGKeyArray | None = None
    ) -> Float[Array, "batch_size state_dim"]:
        """Initialize near the unstable upright equilibrium."""
        state = jnp.broadcast_to(
            jnp.zeros(self.state_dim),
            (self.batch_size, self.state_dim),
        )
        if random_key is None:
            return state

        # NOTE: deviation is arbitrary for demonstration purpose.
        standard_deviation = jnp.array([0.05, 0.02, 0.05, 0.02])
        return state + standard_deviation * jax.random.normal(
            random_key, shape=(self.batch_size, self.state_dim)
        )

    def observe(
        self,
        time: Float,
        state: Float[Array, "state_dim"],
        random_key: PRNGKeyArray,
    ) -> Float[Array, "observation_dim"]:
        return state

    def process(
        self,
        time: Float,
        state: Array,
        control: Array | None,
        random_key: PRNGKeyArray,
    ) -> tuple[Array, Array]:
        half_step = 0.5 * self.time_step
        k1 = self.dynamics(time, state, control)
        k2 = self.dynamics(time + half_step, state + half_step * k1, control)
        k3 = self.dynamics(time + half_step, state + half_step * k2, control)
        k4 = self.dynamics(
            time + self.time_step,
            state + self.time_step * k3,
            control,
        )
        next_state = state + self.time_step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return time + self.time_step, next_state

    def dynamics(
        self,
        time: float,
        state: Float[Array, "batch_size state_dim"],
        control: Array | None = None,
    ) -> Float[Array, "batch_size state_dim"]:
        """Evaluate the cart-pole ordinary differential equation."""

        # ellipsis ... enables both single-instance and batch instance
        cart_velocity = state[..., 1]
        pole_angle = state[..., 2]
        pole_angular_velocity = state[..., 3]

        total_mass = self.cart_mass + self.pole_mass
        adjusted_mass = self.cart_mass + (
            self.pole_mass * jnp.sin(pole_angle) ** 2
        )
        pole_mass_length = self.pole_mass * self.pole_length

        common_numerator = (
            pole_mass_length * jnp.sin(pole_angle) * pole_angular_velocity**2
        )
        angular_acceleration_numerator = total_mass * self.gravity * jnp.sin(
            pole_angle
        ) - pole_mass_length * pole_angular_velocity**2 * jnp.sin(
            pole_angle
        ) * jnp.cos(pole_angle)
        # NOTE: branching for control case
        if control is not None:
            force = control[..., 0]
            common_numerator += force
            angular_acceleration_numerator -= force * jnp.cos(pole_angle)

        pole_angular_acceleration = angular_acceleration_numerator / (
            adjusted_mass * self.pole_length
        )
        cart_acceleration = (
            common_numerator
            - self.pole_mass
            * self.gravity
            * jnp.sin(pole_angle)
            * jnp.cos(pole_angle)
        ) / adjusted_mass

        return jnp.stack(
            (
                cart_velocity,
                cart_acceleration,
                pole_angular_velocity,
                pole_angular_acceleration,
            ),
            axis=-1,
        )
