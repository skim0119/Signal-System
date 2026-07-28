from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

type ControllerState = PyTree[Array]
type Diagnostics = PyTree[Array]


class Controller(eqx.Module):
    """Base class for batched stateful controllers."""

    control_dim: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        assert self.control_dim > 0, "control_dim must be > 0"
        assert self.batch_size > 0, "batch_size must be > 0"

    def initial_state(
        self, random_key: PRNGKeyArray | None = None
    ) -> ControllerState:
        return ()

    @abstractmethod
    def __call__(
        self,
        controller_state: ControllerState,
        time: Float[Array, ""],
        observation: Float[Array, "batch_size observation_dim"],
        random_key: PRNGKeyArray,
    ) -> tuple[
        Float[Array, "batch_size control_dim"],
        ControllerState,
        Diagnostics,
    ]:
        raise NotImplementedError
