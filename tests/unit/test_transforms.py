import jax.numpy as jnp

from rex.base import Chain, Denormalize, Exponential, Extend, Identity, Shared


# Sample data for testing
min_params = {"a": None, "b": jnp.array(1.0)}
max_params = {"a": None, "b": jnp.array(3.0)}
base_params = {"a": jnp.array(1.0), "b": jnp.array(1.5)}
opt_params = {"a": None, "b": jnp.array(0.0)}


# Test for Identity
def test_identity():
    transform = Identity.init()
    applied = transform.apply(base_params)
    inverted = transform.inv(base_params)
    assert applied == base_params
    assert inverted == base_params


# Test for Extend
def test_extend():
    transform = Extend.init(base_params, opt_params)
    extended = transform.apply(opt_params)
    filtered = transform.inv(extended)
    assert extended["a"] == base_params["a"]
    assert extended["b"] == opt_params["b"]
    assert filtered == opt_params


# Test for Denormalize
def test_denormalize():
    transform = Denormalize.init(min_params, max_params)
    denormalized = transform.apply(opt_params)
    normalized = transform.inv(denormalized)
    assert normalized == opt_params
    assert denormalized["b"] == jnp.array(2.0)


# Test for Chain
def test_chain():
    transform1 = Denormalize.init(min_params, max_params)
    transform2 = Extend.init(base_params, opt_params)
    chain = Chain.init(transform1, transform2)
    applied = chain.apply(opt_params)
    inverted = chain.inv(applied)
    assert applied["a"] == base_params["a"]
    assert applied["b"] == jnp.array(2.0)
    assert inverted == opt_params


# Test for ExpTransform
def test_exp_transform():
    transform = Exponential.init()
    applied = transform.apply(base_params)
    inverted = transform.inv(applied)
    assert jnp.allclose(applied["a"], jnp.exp(base_params["a"]))
    assert jnp.allclose(inverted["a"], base_params["a"])


# Test for Shared
def test_shared():
    where_fn = lambda p: p["a"]
    replace_fn = lambda p: p["b"]
    inverse_fn = lambda p: None
    transform = Shared.init(where=where_fn, replace_fn=replace_fn, inverse_fn=inverse_fn)
    applied = transform.apply(opt_params)
    inverted = transform.inv(applied)
    assert applied["a"] == jnp.array(0.0)
    assert inverted["a"] == opt_params["a"]
