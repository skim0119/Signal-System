import jax
import jax.numpy as jnp

from ss.estimation.filtering import HmmFilter, filtering
from ss.system import HiddenMarkovModel, simulate
from ss.utility.parameter.probability import ProbabilityParameter


def _hmm_filter(*, batch_size: int = 1) -> HmmFilter:
    transition = jnp.array(
        [[0.75, 0.25, 0.0], [0.0, 0.75, 0.25], [0.25, 0.0, 0.75]]
    )
    emission = jnp.array([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]])
    return HmmFilter(
        transition=ProbabilityParameter(transition),
        emission=ProbabilityParameter(emission),
        state_dim=3,
        batch_size=batch_size,
    )


def test_hmm_filter_dimensions() -> None:
    filt = _hmm_filter()
    assert filt.state_dim == 3
    assert filt.observation_dim == 1
    assert filt.control_dim == 0
    assert filt.batch_size == 1
    assert filt.time_step == 1.0
    assert filt.transition_matrix.shape == (3, 3)
    assert filt.emission_matrix.shape == (3, 2)


def test_hmm_filter_update_and_predict() -> None:
    filt = _hmm_filter()
    prior = jnp.array([[1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0]])

    posterior = filt.update(prior, jnp.array([[0]]))
    assert jnp.allclose(posterior, jnp.array([[0.4, 0.1, 0.5]]), atol=1e-7)

    predicted = filt.predict(posterior)
    assert jnp.allclose(predicted, jnp.array([[0.425, 0.175, 0.4]]), atol=1e-7)

    posterior = filt.update(predicted, jnp.array([[1]]))
    assert jnp.allclose(
        posterior,
        jnp.array([[0.2, 0.3294117647, 0.4705882353]]),
        atol=1e-7,
    )


def test_hmm_filter_with_matrices_is_immutable() -> None:
    filt = _hmm_filter()
    new_transition = jnp.eye(3)
    new_emission = jnp.ones((3, 2)) / 2.0

    updated = filt.with_transition_matrix(new_transition).with_emission_matrix(
        new_emission
    )

    assert updated is not filt
    assert jnp.allclose(updated.transition_matrix, new_transition)
    assert jnp.allclose(updated.emission_matrix, new_emission)
    assert not jnp.allclose(filt.transition_matrix, new_transition)


def test_filtering_single_and_batch() -> None:
    filt = _hmm_filter()
    initial_belief = jnp.array([[1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0]])
    observations = jnp.array([[[0]], [[1]], [[0]]])  # (time, batch, obs)

    beliefs = filtering(filt, initial_belief, observations)
    assert beliefs.shape == (3, 1, 3)
    assert jnp.allclose(beliefs[0, 0], jnp.array([0.4, 0.1, 0.5]), atol=1e-7)

    filt = filt.duplicate(batch_size=2)
    batch_observations = jnp.concatenate([observations, observations], axis=1)
    initial_beliefs = jnp.concatenate([initial_belief, initial_belief], axis=0)
    batch_beliefs = filtering(filt, initial_beliefs, batch_observations)
    assert batch_beliefs.shape == (3, 2, 3)
    assert jnp.allclose(batch_beliefs[:, 0], beliefs[:, 0])
    assert jnp.allclose(batch_beliefs[:, 1], beliefs[:, 0])


def test_filtering_is_jittable() -> None:
    filt = _hmm_filter()
    initial_belief = jnp.array([[1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0]])
    observations = jnp.array([[[0]], [[1]]])  # (time, batch, obs)

    jitted = jax.jit(filtering)
    beliefs = jitted(filt, initial_belief, observations)
    assert beliefs.shape == (2, 1, 3)

    filt = filt.duplicate(batch_size=2)
    batch_beliefs = jitted(
        filt,
        jnp.concatenate([initial_belief, initial_belief], axis=0),
        jnp.concatenate([observations, observations], axis=1),
    )
    assert batch_beliefs.shape == (2, 2, 3)


def test_filtering_from_simulate_batch() -> None:
    """simulate/filtering share time-leading layout; state aligns with obs."""
    transition = jnp.array([[0.7, 0.3], [0.4, 0.6]])
    emission = jnp.array([[0.9, 0.1], [0.2, 0.8]])
    system = HiddenMarkovModel(
        transition=ProbabilityParameter(transition),
        emission=ProbabilityParameter(emission),
    ).duplicate(batch_size=4)
    filt = HmmFilter(
        transition=ProbabilityParameter(transition),
        emission=ProbabilityParameter(emission),
        state_dim=2,
        batch_size=4,
    )

    key = jax.random.PRNGKey(0)
    init_key, scan_key = jax.random.split(key)
    init_states = system.initial_state(init_key)
    _, states, observations, _ = simulate(
        system, 0.0, 6, init_states, scan_key
    )
    assert states.shape[:2] == observations.shape[:2] == (6, 4)
    assert observations.shape == (6, 4, 1)

    initial_belief = jnp.full((4, 2), 0.5)
    beliefs = filtering(filt, initial_belief, observations)
    assert beliefs.shape == (6, 4, 2)
    assert jnp.allclose(jnp.sum(beliefs, axis=-1), 1.0)
