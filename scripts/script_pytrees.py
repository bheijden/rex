import jumpy
import jax
from rex.distributions import Gaussian
from dummy import DummyNode, DummyEnv, DummyAgent, DummyOutput
from rex.constants import LATEST, BUFFER, WARN
from rex.base import StepState
from flax.core import FrozenDict
import rex.jumpy as rjp
import jumpy.numpy as jp
import numpy as onp

# Initialize nodes
world = DummyNode("world", rate=20, delay_sim=Gaussian(0 / 1e3), log_level=WARN, color="magenta")
sensor = DummyNode("sensor", rate=20, delay_sim=Gaussian(7 / 1e3), log_level=WARN, color="yellow")
observer = DummyNode("observer", rate=30, delay_sim=Gaussian(16 / 1e3), log_level=WARN, color="cyan")
agent = DummyAgent("agent", rate=45, delay_sim=Gaussian(5 / 1e3, 1 / 1e3), log_level=WARN, color="blue", advance=True)
actuator = DummyNode("actuator", rate=45, delay_sim=Gaussian(1 / 45), log_level=WARN, color="green", advance=False, stateful=False)
nodes = [world, sensor, observer, agent, actuator]
nodes = {n.name: n for n in nodes}

# Connect
sensor.connect(world, blocking=False, delay_sim=Gaussian(4 / 1e3), skip=False, jitter=LATEST)
observer.connect(sensor, blocking=False, delay_sim=Gaussian(3 / 1e3), skip=False, jitter=BUFFER)
observer.connect(agent, blocking=False, delay_sim=Gaussian(3 / 1e3), skip=True, jitter=LATEST)
agent.connect(observer, blocking=True, delay_sim=Gaussian(3 / 1e3), skip=False, jitter=BUFFER)
actuator.connect(agent, blocking=True, delay_sim=Gaussian(3 / 1e3, 1 / 1e3), skip=False, jitter=LATEST, delay=3 / 1e3)
world.connect(actuator, blocking=False, delay_sim=Gaussian(4 / 1e3), skip=True, jitter=LATEST)

# Define environment
env = DummyEnv(nodes, agent=agent, max_steps=200)

# Get initial graph_state
base_gs = env._get_graph_state(jumpy.random.PRNGKey(0))

# Replace dicts with FrozenDicts
nodes = FrozenDict(base_gs.nodes)
for name in list(nodes.keys()):
    ss = nodes[name]
    new_inputs = FrozenDict(ss.inputs)
    new_ss = ss.replace(inputs=new_inputs)
    nodes = nodes.copy({name: new_ss})

# Initial graph state & mask
base_gs = base_gs.replace(nodes=nodes)
base_mask = jax.tree_map(lambda x: False, base_gs)

must_update = lambda tree, val: jax.tree_map(lambda x: val, tree)

node_names = list(base_gs.nodes.keys())
gs_lst = []
mask_lst = []
step_lst = {i+1: name for i, name in enumerate(node_names)}
for i, name in step_lst.items():
    new_nodes = dict()
    new_nodes_mask = dict()

    # Emulate node step
    new_state = base_gs.nodes[name].state.replace(step=jp.int32(i))
    new_params = base_gs.nodes[name].params.replace(param_1=jp.int32(i))
    new_nodes[name] = base_gs.nodes[name].replace(state=new_state, params=new_params)  # NOTE! Shallow copies input dict
    # Emulate node step mask
    new_state_mask = must_update(base_mask.nodes[name].state, True)  # todo: if stateful
    new_params_mask = must_update(base_mask.nodes[name].params, True)  # todo: if not static
    new_rng_mask = must_update(base_mask.nodes[name].rng, True)
    new_nodes_mask[name] = base_mask.nodes[name].replace(state=new_state_mask, params=new_params_mask, rng=new_rng_mask)

    # Emulate pushing outputs to other nodes
    output = DummyOutput(seqs_sum=jp.int32(i), dummy_1=jp.array([0.0, 1.0], dtype=jp.float32))
    for node_name in node_names:
        new_inputs_mask = dict()
        for input_name in base_gs.nodes[node_name].inputs.keys():
            if input_name != name:
                continue
            # Push output
            new_input = base_gs.nodes[node_name].inputs[input_name].replace(seq=jp.array([i]))
            new_inputs = base_gs.nodes[node_name].inputs.copy({input_name: new_input})
            new_input_mask = must_update(base_mask.nodes[node_name].inputs[input_name], True)
            new_inputs_mask = base_mask.nodes[node_name].inputs.copy({input_name: new_input_mask})
            # NOTE! Currently, nodes cannot self-connect or have multiple inputs from the same node.
            assert node_name not in new_nodes, "Overwriting node. Should implement merging instead."
            new_nodes[node_name] = base_gs.nodes[node_name].replace(inputs=new_inputs)
            new_nodes_mask[node_name] = base_mask.nodes[node_name].replace(inputs=new_inputs_mask)

    new_gs = base_gs.replace(nodes=base_gs.nodes.copy(new_nodes))
    new_mask = base_mask.replace(nodes=base_mask.nodes.copy(new_nodes_mask))
    gs_lst.append(new_gs)
    mask_lst.append(new_mask)


class TreeLeaf:
    def __init__(self, container):
        self.c = container


@jax.jit
def update(_gs_lst, _mask_lst, old):
    gs_choice = jax.tree_map(lambda *args: TreeLeaf(args), *_gs_lst)
    mask_choice = jax.tree_map(lambda *args: TreeLeaf(args), *_mask_lst)
    jax.tree_map(lambda mask, next_gs, prev_gs: print(f"{mask.c=} | {next_gs.c=} | {prev_gs=}"), mask_choice, gs_choice, old)
    new_gs = jax.tree_map(lambda mask, next_gs, prev_gs: rjp.select(mask.c, next_gs.c, prev_gs), mask_choice, gs_choice,
                         old)
    return new_gs


new_gs = update(gs_lst, mask_lst, base_gs)

# def xor(*args):
#     return print(f"xor={sum(args) == 1} | {args}",)
# check = jax.tree_map(xor, *mask_lst)

import jax.numpy as jnp
import jax

timings_chunk = {0: jnp.array([10, 11, 22, 33, 44, 55, 66, 77, 88, 99, 100]),
                 1: jnp.array([1, 1]),
                 2: jnp.array([2, 2]),
                 3: jnp.array([3, 3]),
                 4: jnp.array([4, 4])}

indices = jnp.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])


@jax.jit
def f1(i: jnp.int32):
    idx = indices[i]
    print("trace f1")
    return timings_chunk[0][idx]

print(f1(jnp.array(0)), f1(jnp.array(1)))

m = (False, False, True, False, False)
arr = (jp.array(9), jp.array(1), jp.array(2), jp.array(3), jp.array(4))
f = (9, 1, 2, 3, 4)

onp.select(m, f)

t1 = [[1, 2], [3, 4], [5, 6]]
t2 = ((7, 8), (9, 10), (11, 12))
t3 = [[True, True], [False, False], [True, True]]

t4 = onp.select(True, t1, t2)
t5 = onp.where(t3, t1, t2)
t6 = jp.where(t3, t1, t2)