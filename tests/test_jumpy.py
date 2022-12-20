import rex.jumpy as rjp


def test_jumpy():

    with rjp.use(backend="jax"):
        pass

    with rjp.use(backend="numpy"):
        pass

