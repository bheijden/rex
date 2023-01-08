from rex.spaces import Box, Discrete
import jumpy as jp


def test_box_space():
	low = jp.array([0, 1, 2])
	high = jp.array([1, 2, 3])
	b = Box(low=low, high=high)
	assert b.contains(b.sample(jp.random_prngkey(0)))
	assert b.contains(b.sample(jp.random_prngkey(1)))


def test_discrete_space():
	d = Discrete(3)
	assert d.contains(d.sample(jp.random_prngkey(0)))
	assert d.contains(d.sample(jp.random_prngkey(1)))