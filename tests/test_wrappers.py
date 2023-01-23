import jumpy
import jumpy.numpy as jp
from rex.wrappers import GymWrapper, VecGymWrapper, AutoResetWrapper
from scripts.dummy import build_dummy_env, DummyEnv, build_dummy_compiled_env


def test_auto_reset_wrapper():
	# Create dummy graph
	env_traced, env, nodes = build_dummy_compiled_env()

	# Test wrapper on non-compiled environments
	try:
		has_failed = False
		env: DummyEnv = AutoResetWrapper(env)  # Wrap into auto reset wrapper
	except TypeError as e:
		has_failed = True
		print(e)
	assert has_failed, "AutoResetWrapper should fail on non-compiled environments"

	# Apply wrapper on traced environment
	env_traced: DummyEnv = AutoResetWrapper(env_traced)  # Wrap into auto reset wrapper

	# Get spaces
	action_space = env_traced.action_space()
	observation_space = env_traced.observation_space()

	# Run environment
	done, (graph_state, obs) = False, env_traced.reset(jumpy.random.PRNGKey(0))
	for _ in range(4):
		while not done:
			action = action_space.sample(jumpy.random.PRNGKey(0))
			graph_state, obs, rewards, dones, info = env_traced.step(graph_state, action)
			done = dones.any()
	env_traced.close()


def test_gym_wrapper():
	# Create dummy graph
	env, _, nodes = build_dummy_compiled_env()

	# Apply wrapper
	env = GymWrapper(env)  # Wrap into gym wrapper

	# Get spaces
	observation_space = env.observation_space
	env._graph_state = None  # Just to trigger autoreset to obtain action space
	action_space = env.action_space

	# Jit
	env.jit()

	# Run environment
	for _ in range(1):
		done, obs = False, env.reset()
		while not done:
			action = action_space.sample()
			obs, reward, done, info = env.step(action)
	env.close()


def test_vec_gym_wrapper():
	# Create dummy graph
	env, _, nodes = build_dummy_compiled_env()

	# Apply wrapper
	env = AutoResetWrapper(env)  # Wrap into auto reset wrapper
	env = VecGymWrapper(env, num_envs=2)  # Wrap into vectorized environment

	# Call api
	env.env_is_wrapped(VecGymWrapper)
	env.env_is_wrapped(GymWrapper)

	# Get spaces
	action_space = env.action_space
	_action_space = env._action_space

	# Jit
	env.jit()

	# Run environment
	for _ in range(1):
		done, obs = False, env.reset()
		while not done:
			action = jp.array([action_space.sample()] * env.num_envs)
			obs, rewards, dones, info = env.step(action)
			done = dones.any()
	env.close()