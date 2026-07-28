import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ._parameter import Parameter


class SoftmaxTransformer(eqx.Module):
    """
    A transformer that maps raw values to a probability simplex via softmax.
    """

    log_zero_offset: float = eqx.field(static=True, default=20.0)

    def forward(self, raw_value: Array) -> Array:
        return jax.nn.softmax(raw_value, axis=-1)

    def inverse(self, value: Array) -> Array:
        # TODO: There should be a better way to handle the zero entries in the
        # softmax output. The current implementation computes the log of the
        # minimum non-zero value and subtracts a fixed offset to get a "safe"
        # log value for the zero entries. This is a workaround and may not be
        # the best approach for all use cases.
        zero_mask = value == 0
        log_nonzero_min = jnp.log(
            jnp.min(jnp.where(zero_mask, jnp.inf, value))
        )
        log_zero_value = log_nonzero_min - self.log_zero_offset
        safe_value = jnp.where(zero_mask, jnp.exp(log_zero_value), value)
        return jnp.log(safe_value)


class ProbabilityParameter(Parameter[SoftmaxTransformer]):
    """
    A Parameter that is constrained to the probability simplex via a
    SoftmaxTransformer. This is useful for parameters that represent
    probabilities, such as the rows of a transition matrix in an HMM.
    """

    def __init__(self, value: Array) -> None:
        super().__init__(
            value=jnp.asarray(value), transformer=SoftmaxTransformer()
        )
