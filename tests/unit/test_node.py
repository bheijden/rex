import jax
import jax.numpy as jnp
from flax import struct as struct
from distrax import Normal
from rex.base import GraphState, TrainableDist, Base
from rex.node import BaseNode, BaseWorld
from rex.constants import LogLevel
from rex.utils import set_log_level



def test_node_basic_api():
    """Test basic API methods of Node."""
    node1 = BaseNode(name="node1", rate=50, color="pink", order=1)
    node2 = BaseNode(name="node2", rate=50, color="teal", order=3)
    node2.connect(node1, window=3, name="node1", blocking=True)

    # Test setting delay
    node2.set_delay(Normal(0.01, 0.01), 0.01)
    node2.inputs["node1"].set_delay(Normal(0.01, 0.01), 0.01)

    # Test phase
    _ = node2.phase
    _ = node2.phase_output

    # Test properties
    _ = node1.log_level
    _ = node1.log_color
    _ = node1.fcolor
    _ = node1.ecolor

    # Test start stop API
    gs = GraphState()
    node1.startup(gs)
    node1.stop()


def test_step_state_api():
    node1 = BaseNode(name="node1", rate=50, color="pink", order=1)
    node2 = BaseNode(name="node2", rate=50, color="teal", order=3)
    node2.connect(node1, window=3, name="node1", blocking=True)

    # Get step states
    ss_node1 = node1.init_step_state()
    ss_node2 = node2.init_step_state()
    ss = {node1.name: ss_node1, node2.name: ss_node2}

    # Initialize graph_state
    graph_state = GraphState().replace_step_states(ss)

    # Access the step_state API
    step_state_dict = graph_state.step_state

    # Test __getitem__
    step_state_node1 = step_state_dict["node1"]
    assert step_state_node1.seq == ss_node1.seq
    assert step_state_node1.ts == ss_node1.ts
    assert step_state_node1.params == ss_node1.params
    assert step_state_node1.state == ss_node1.state
    assert step_state_node1.inputs == ss_node1.inputs

    # Test __len__
    assert len(step_state_dict) == 2

    # Test keys
    assert set(step_state_dict.keys()) == {"node1", "node2"}

    # Test items
    items = step_state_dict.items()
    assert len(items) == 2

    # Test values
    values = step_state_dict.values()
    assert len(values) == 2

    # Test get
    assert step_state_dict.get("node1") == step_state_node1
    assert step_state_dict.get("non_existent", default=None) is None

    # Test iteration assert keys are the same
    keys = list(iter(step_state_dict))
    assert set(keys) == {"node1", "node2"}


def test_cyclic_connection():
    """Test cyclic connection."""
    node1 = BaseNode(name="node1", rate=50, color="pink", order=1)
    node2 = BaseNode(name="node2", rate=50, color="teal", order=3)
    node2.connect(node1, window=3, name="node1", blocking=True)
    node1.connect(node2, window=3, name="node2", blocking=True)

    try:
        node1.phase
    except RecursionError as e:
        str_e = str(e)
        assert "Algebraic loop detected" in str_e


def test_node_step_state():
    node1 = BaseNode(name="node1", rate=50, color="pink", order=1, delay_dist=Normal(0.01, 0.01))
    node2 = BaseNode(name="node2", rate=50, color="teal", order=3, delay_dist=Normal(0.01, 0.01))
    node2.connect(node1, window=3, name="node1", blocking=True, delay_dist=TrainableDist.create(0.01, 0.0, 0.02))

    # Test step state
    _ss = node2.init_step_state()


def test_node_reloading():
    """Test reloading a Node."""
    # Use from_info, connect_from_info.
    node1 = BaseNode(name="node1", rate=50, color="pink", order=1, delay_dist=Normal(0.01, 0.01))
    node2 = BaseNode(name="node2", rate=50, color="teal", order=3, delay_dist=Normal(0.01, 0.01))
    node2.connect(node1, window=3, name="node1", blocking=True, delay_dist=Normal(0.01, 0.01))
    nodes = {node1.name: node1, node2.name: node2}

    # Get infos
    infos = {node2.name: node2.info, node1.name: node1.info}

    # Reload node1 and node2 nodes
    node1_reloaded = node1.from_info(infos["node1"])
    node2_reloaded = node2.from_info(infos["node2"])

    # Reconnect from info
    nodes_reloaded = {node1.name: node1_reloaded, node2.name: node2_reloaded}
    node1_reloaded.connect_from_info(infos["node1"].inputs, nodes_reloaded)
    node2_reloaded.connect_from_info(infos["node2"].inputs, nodes_reloaded)

    assert node1_reloaded.info == node1.info
    assert node2_reloaded.info == node2.info
    assert node2_reloaded.inputs.keys() == node2.inputs.keys()
    assert node1_reloaded.inputs.keys() == node1.inputs.keys()


def test_logging_api():
    node1 = BaseNode(name="node1", rate=50, color="pink", order=1)

    # Set log level
    set_log_level(LogLevel.DEBUG, node1, color="red")

    # Log something
    node1.log("", "Logging something...", LogLevel.DEBUG)

    # Set global log level
    set_log_level(LogLevel.DEBUG)

    # Log something
    node1.log("Test", "Logging something with global log_level set to DEBUG... ", LogLevel.DEBUG)

    # Set global log level
    set_log_level(LogLevel.WARN)

    # node2 = BaseNode(name="node2", rate=50, color="teal", order=3)
    # node2.connect(node1, window=3, name="node1", blocking=True)


def test_base_world():
    world = BaseWorld(name="world", rate=50, color="grape", order=0)


def test_base_api():
    @struct.dataclass
    class Pytree(Base):
        """Arbitrary dataclass."""
        a: jax.Array

    # Create Pytree objects
    tree1 = Pytree(a=jnp.array([1, 2, 3]))
    tree2 = Pytree(a=jnp.array([4, 5, 6]))
    scalar = 2

    # Test addition
    result = tree1 + tree2
    assert jnp.array_equal(result.a, jnp.array([5, 7, 9]))

    result = tree1 + scalar
    assert jnp.array_equal(result.a, jnp.array([3, 4, 5]))

    # Test __repr__
    tree1.__repr__()
    tree1.__str__()

    # Test replace
    result = tree1.replace(a=jnp.array([7, 8, 9]))
    assert jnp.array_equal(result.a, jnp.array([7, 8, 9]))

    # Test subtraction
    result = tree1 - tree2
    assert jnp.array_equal(result.a, jnp.array([-3, -3, -3]))

    result = tree1 - scalar
    assert jnp.array_equal(result.a, jnp.array([-1, 0, 1]))

    # Test multiplication
    result = tree1 * tree2
    assert jnp.array_equal(result.a, jnp.array([4, 10, 18]))

    result = tree1 * scalar
    assert jnp.array_equal(result.a, jnp.array([2, 4, 6]))

    # Test negation
    result = -tree1
    assert jnp.array_equal(result.a, jnp.array([-1, -2, -3]))

    # Test division
    result = tree1 / tree2
    assert jnp.allclose(result.a, jnp.array([0.25, 0.4, 0.5]))

    result = tree1 / scalar
    assert jnp.allclose(result.a, jnp.array([0.5, 1.0, 1.5]))

    # Test indexing
    result = tree1.a[1]
    assert result == 2

    # Test reshape
    result = tree1.reshape((3, 1))
    assert result.a.shape == (3, 1)
    assert jnp.array_equal(result.a, jnp.array([[1], [2], [3]]))

    # Test slicing
    result = tree1.slice(0, 2)
    assert jnp.array_equal(result.a, jnp.array([1, 2]))

    # Test take
    result = tree1.take(jnp.array([0, 2]))
    assert jnp.array_equal(result.a, jnp.array([1, 3]))

    # Test concatenation
    tree3 = Pytree(a=jnp.array([7, 8, 9]))
    result = tree1.concatenate(tree2, tree3, axis=0)
    assert jnp.array_equal(result.a, jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    # Test select
    cond = jnp.array([1, 0, 1])
    result = tree1.select(tree2, cond)
    assert jnp.array_equal(result.a, jnp.array([1, 5, 3]))

    # Test index_set
    result = tree1.index_set(0, tree2[2])
    assert jnp.array_equal(result.a, jnp.array([6, 2, 3]))

    # Test index_sum
    result = tree1.index_sum(0, tree2[2])
    assert jnp.array_equal(result.a, jnp.array([7, 2, 3]))