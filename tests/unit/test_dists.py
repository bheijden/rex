import pytest
import jax
import jax.numpy as jnp
import numpy as onp
from distrax import Deterministic, Normal, MixtureSameFamily, Categorical
from rex.base import StaticDist, TrainableDist, DelayDistribution, InputState


@pytest.mark.parametrize("static_dist", ["Deterministic", "Normal", "MixtureSameFamily"])
def test_static_dist_quantile(static_dist):
    if static_dist == "Deterministic":
        static_dist = Deterministic(1.0)
    elif static_dist == "Normal":
        static_dist = Normal(loc=1.0, scale=0.1)
    elif static_dist == "MixtureSameFamily":
        static_dist = MixtureSameFamily(
            mixture_distribution=Categorical(probs=jnp.array([0.5, 0.5])),
            components_distribution=Normal(loc=jnp.array([1.0, 1.0]), scale=jnp.array([0.1, 0.1])),
        )
    dist = StaticDist.create(static_dist)
    q = dist.quantile(0.5)
    if isinstance(static_dist, Deterministic):
        assert q == 1.0
    elif isinstance(static_dist, Normal):
        assert q == 1.0
    elif isinstance(static_dist, MixtureSameFamily):
        assert jnp.isclose(q, 1.0, atol=1e-1)


def test_static_dist_api():
    dist = StaticDist.create(Deterministic(1.0))

    # Test reset
    new_rng = jax.random.PRNGKey(42)
    reset_dist = dist.reset(new_rng)
    assert reset_dist.rng is new_rng

    # Test sample and sample_pure
    new_dist, samples = dist.sample()
    assert samples == 1.0
    new_dist, samples = dist.sample((3,))
    assert samples.shape == (3,)
    assert jnp.all(samples == 1.0)
    assert new_dist.rng is not dist.rng

    static_sample_pure, pure_samples = DelayDistribution.sample_pure(dist, (3,))
    assert jnp.all(pure_samples == 1.0)

    # Test quantile and quantile_pure
    q = dist.quantile(0.5)
    assert q == 1.0
    pure_q = DelayDistribution.quantile_pure(dist, 0.5)
    assert pure_q == 1.0

    # Test mean and mean_pure
    mean = dist.mean()
    assert mean == 1.0
    pure_mean = DelayDistribution.mean_pure(dist)
    assert pure_mean == 1.0

    # Test pdf and pdf_pure
    pdf = dist.pdf(1.0)
    assert pdf == 1.0
    pure_pdf = DelayDistribution.pdf_pure(dist, 1.0)
    assert pure_pdf == 1.0

    # Test window and equivalent
    assert dist.window(10) == 0
    assert dist.equivalent(StaticDist.create(Deterministic(1.0)))

    # Test inherited apply_delay (no-op for StaticDist)
    input_state = InputState(
        seq=jnp.array([1]),
        ts_sent=jnp.array([0.0]),
        ts_recv=jnp.array([0.0]),
        data=jnp.array([1.0]),
        delay_dist=dist,
    )
    delayed_input = dist.apply_delay(10.0, input_state, ts_start=0.0)
    assert delayed_input is input_state


def test_trainable_dist_api():
    dist = TrainableDist.create(delay=1.0, min=0.0, max=2.0)

    # Test get alpha
    alpha = dist.get_alpha(1.5)
    assert alpha == 0.75

    # Test reset
    new_rng = jax.random.PRNGKey(42)
    reset_dist = dist.reset(new_rng)
    assert reset_dist is dist  # No state change

    # Test sample and sample_pure
    _, samples = dist.sample((3,))
    assert samples.shape == (3,)
    assert jnp.all(samples == 1.0)

    pure_sample_pure, pure_samples = DelayDistribution.sample_pure(dist, (3,))
    assert jnp.all(pure_samples == 1.0)

    # Test quantile and quantile_pure
    q = dist.quantile(0.5)
    assert q == 1.0
    pure_q = DelayDistribution.quantile_pure(dist, 0.5)
    assert pure_q == 1.0

    # Test mean and mean_pure
    mean = dist.mean()
    assert mean == 1.0
    pure_mean = DelayDistribution.mean_pure(dist)
    assert pure_mean == 1.0

    # Test pdf and pdf_pure
    pdf = dist.pdf(1.0)
    assert pdf == 1.0
    pure_pdf = DelayDistribution.pdf_pure(dist, 1.0)
    assert pure_pdf == 1.0

    # Test equivalent
    other_dist = TrainableDist.create(delay=1.0, min=0.0, max=2.0)
    assert dist.equivalent(other_dist)
    assert not dist.equivalent(StaticDist.create(Deterministic(1.0)))
    assert not dist.equivalent(dist.replace(max=3.0))
    assert not dist.equivalent(dist.replace(min=0.1))
    assert not dist.equivalent(dist.replace(interp="linear"))

    # Test window
    assert dist.window(rate_out=10.0) == 20
    noop_dist = TrainableDist(alpha=0.0, min=1.0, max=1.0)
    assert noop_dist.window(rate_out=10) == 0

    # Prepare mock input state
    seq = onp.array([-1, -1, 0, 1, 2, 3])
    ts_sent = onp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3])
    ts_recv = ts_sent + dist.mean()
    data = onp.array([-100, -100, 0.0, 1.0, 2.0, 3.0])[:, None]
    input_state = InputState.from_outputs(
        seq=seq, ts_sent=ts_sent, ts_recv=ts_recv, outputs=data, delay_dist=noop_dist, is_data=True
    )
    interp_input_state = input_state.delay_dist.apply_delay(10.0, input_state, ts_start=0.0)
    assert interp_input_state.seq.shape == input_state.seq.shape


@pytest.mark.parametrize("interp_method", ["zoh", "linear", "linear_real_only"])
def test_trainable_dist_apply_delay(interp_method):
    dist = TrainableDist.create(delay=0.2, min=0.0, max=0.4, interp=interp_method)

    # Prepare mock input state
    seq = onp.array([-1, -1, 0, 1, 2, 3])
    ts_sent = onp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3])
    ts_recv = ts_sent + dist.mean()  # I believe, this has no influence, as it's overridden in apply_delay.
    data = onp.array([-100, -100, 0.0, 1.0, 2.0, 3.0])[:, None]
    input_state = InputState.from_outputs(
        seq=seq, ts_sent=ts_sent, ts_recv=ts_recv, outputs=data, delay_dist=dist, is_data=True
    )

    # Test apply_delay
    rate_out = 10
    ts_start = 0.0
    interp_input_state = input_state.delay_dist.apply_delay(rate_out, input_state, ts_start + dist.mean() + 0.15)
    print(f"{interp_method}: ", interp_input_state)
    assert interp_input_state.seq.shape == (
        2,
    )  # Only 2 samples should be left after delay application (i.e. original window was 2)
