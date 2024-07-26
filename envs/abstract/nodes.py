from math import ceil
from typing import Union, Tuple, Dict, Any, List, Sequence

import jax
from flax import struct
from flax.core import FrozenDict
import flax.linen as nn
import distrax
import equinox as eqx
from jax import numpy as jnp
import numpy as onp

from rexv2 import base
from rexv2.base import GraphState, StepState, Base, InputState
from rexv2.node import BaseNode, Connection
from rexv2.rl import NormalizeVec, SquashState
from rexv2.jax_utils import tree_dynamic_slice

MIN_DELAY = 0.0
MAX_DELAY = 1.0


@struct.dataclass
class State(base.Base):
    """Output of the node."""
    x: Union[float, jax.typing.ArrayLike]
    loss: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class Output(base.Base):
    """Output of the node."""
    y: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class LinearDynamicsParams(base.Base):
    """Parameters of the node."""
    init_x: Union[float, jax.typing.ArrayLike]
    A: Union[float, jax.typing.ArrayLike]
    B: Union[float, jax.typing.ArrayLike]
    C: Union[float, jax.typing.ArrayLike]
    D: Union[float, jax.typing.ArrayLike]
    delays: Union[Dict[str, base.TrainableDist], FrozenDict[str, base.TrainableDist]]

    def __repr__(self):
        return eqx.tree_pformat(self, short_arrays=False)

    @classmethod
    def init_params(cls, rng: jax.Array, delays: Dict[str, float]):
        true_alphas = {k: d / (MAX_DELAY - MIN_DELAY) for k, d in delays.items()}
        delays = {k: base.TrainableDist(alpha=a, min=MIN_DELAY, max=MAX_DELAY, interp="zoh") for k, a in true_alphas.items()}
        base_params = cls(init_x=0.0, A=0.0, B=0.0, C=0.0, D=0.0, delays=delays)
        min_params = base_params.min()
        max_params = base_params.max()
        min_leaves, treedef = jax.tree_util.tree_flatten(min_params)
        max_leaves, _ = jax.tree_util.tree_flatten(max_params)
        rnd_leaves = jax.random.uniform(rng, shape=(len(min_leaves),), minval=jnp.array(min_leaves), maxval=jnp.array(max_leaves))
        rnd_params = jax.tree_util.tree_unflatten(treedef, list(rnd_leaves))
        rnd_params = rnd_params.replace(delays={k: v.replace(alpha=true_alphas[k]) for k, v in rnd_params.delays.items()})
        return rnd_params

    def min(self):
        min_delays = {k: v.replace(alpha=0.0) for k, v in self.delays.items()}
        return LinearDynamicsParams(init_x=-10, A=-1., B=-1.0, C=0.5, D=-1.0, delays=min_delays)

    def max(self):
        max_delays = {k: v.replace(alpha=1.0) for k, v in self.delays.items()}
        return LinearDynamicsParams(init_x=10, A=-0.0, B=1.0, C=1.0, D=1.0, delays=max_delays)

    def init_state(self) -> State:
        return State(x=self.init_x, loss=0.0)

    def step(self, ts: float, dt: float, state: State, u: Dict[str, jax.typing.ArrayLike]) -> Tuple[State, Output]:
        x = state.x
        u_vec = jnp.array(list(u.values()))
        u_sum = u_vec.sum()
        x = jnp.exp(self.A * dt) * x + dt*self.B * u_sum
        y = self.C * x + self.D * u_sum
        return state.replace(x=x), Output(y=y)


@struct.dataclass
class HarmonicParams(base.Base):
    """Parameters of the node with harmonic properties."""
    frequency: Union[float, jax.typing.ArrayLike]
    phase_shift: Union[float, jax.typing.ArrayLike]
    delays: Union[Dict[str, base.TrainableDist], FrozenDict[str, base.TrainableDist]]

    def __repr__(self):
        return eqx.tree_pformat(self, short_arrays=False)

    @classmethod
    def init_params(cls, rng: jax.Array, delays: Dict[str, float]):
        true_alphas = {k: d / (MAX_DELAY - MIN_DELAY) for k, d in delays.items()}
        delays = {k: base.TrainableDist(alpha=a, min=MIN_DELAY, max=MAX_DELAY, interp="zoh") for k, a in true_alphas.items()}
        base_params = cls(frequency=0.2, phase_shift=0.0, delays=delays)
        min_params = base_params.min()
        max_params = base_params.max()
        min_leaves, treedef = jax.tree_util.tree_flatten(min_params)
        max_leaves, _ = jax.tree_util.tree_flatten(max_params)
        rnd_leaves = jax.random.uniform(rng, shape=(len(min_leaves),), minval=jnp.array(min_leaves), maxval=jnp.array(max_leaves))
        rnd_params = jax.tree_util.tree_unflatten(treedef, list(rnd_leaves))
        rnd_params = rnd_params.replace(delays={k: v.replace(alpha=true_alphas[k]) for k, v in rnd_params.delays.items()})
        return rnd_params

    def min(self):
        min_delays = {k: v.replace(alpha=0.0) for k, v in self.delays.items()}
        return HarmonicParams(frequency=0.2, phase_shift=-jnp.pi, delays=min_delays)

    def max(self):
        max_delays = {k: v.replace(alpha=1.0) for k, v in self.delays.items()}
        return HarmonicParams(frequency=2.0, phase_shift=jnp.pi, delays=max_delays)

    def init_state(self) -> State:
        return State(x=0, loss=0.0)

    def step(self, ts: float, dt: float, state: State, u: Dict[str, jax.typing.ArrayLike]) -> Tuple[State, Output]:
        u_vec = jnp.array(list(u.values()))
        u_sum = u_vec.sum()
        y = jnp.sin(2 * jnp.pi * ts * self.frequency + self.phase_shift*0) + u_sum # + (1+u_sum)
        return state, Output(y=y)


class Abstract(BaseNode):
    PARAM_CLS = {"linear": LinearDynamicsParams, "harmonic": HarmonicParams}

    def __init__(self, *args, param_cls: str = "linear", outputs: Output = None, seq: jax.typing.ArrayLike = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_cls = param_cls
        assert (seq is None) == (outputs is None), "Either both outputs and seq should be None or both should be provided."
        self._outputs = outputs
        self._seq = seq

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> LinearDynamicsParams:
        """Default params of the node."""
        rng = jax.random.PRNGKey(0) if rng is None else rng
        delays = {k: c.delay_dist.mean() for k, c in self.inputs.items()}
        params = self.PARAM_CLS[self.param_cls].init_params(rng, delays)
        return params

    def init_state(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> State:
        """Default state of the node."""
        graph_state = graph_state or base.GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        return params.init_state()

    def init_output(self, rng: jax.Array = None, graph_state: base.GraphState = None) -> Output:
        """Default output of the node."""
        output = Output(y=0.0)
        return output

    def init_inputs(self, rng: jax.Array = None, graph_state: GraphState = None) -> FrozenDict[str, Any]:
        """Default inputs of the node."""
        graph_state = graph_state or GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        inputs = super().init_inputs(rng, graph_state).unfreeze()
        new_inputs = {}
        for k, i in inputs.items():
            if isinstance(i.delay_dist, base.TrainableDist):
                new_inputs[k] = i.replace(delay_dist=params.delays[k])
        inputs.update(new_inputs)
        return FrozenDict(inputs)

    def step(self, step_state: base.StepState) -> Tuple[base.StepState, Output]:
        """Step the node."""
        # Unpack StepState
        rng, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs

        # Step the node
        dt = 1 / self.rate
        u = {k: i.data.y[-1] for k, i in inputs.items()}
        new_state, output = params.step(step_state.ts, dt, state, u)

        # Calculate loss
        if self._outputs is not None:
            num_samples = jax.tree_flatten(self._outputs)[0][0].shape[-1]
            max_seq = jnp.take(self._seq, step_state.eps)
            in_bounds = step_state.seq < max_seq  # num_samples
            seq_mod = step_state.seq % num_samples
            output_target = tree_dynamic_slice(self._outputs, jnp.array([step_state.eps, seq_mod]))
            new_loss = new_state.loss + in_bounds*(output.y - output_target.y)**2
            # if self.name in ["1", "2"]:
            #     keys = {}
            #     msg = self.name
            #     msg += " | eps={eps}"
            #     keys["eps"] = step_state.eps
            #     msg += " | seq={seq}"
            #     keys["seq"] = step_state.seq
            #     msg += " | ts={ts}"
            #     keys["ts"] = step_state.ts
            #     msg += " | seq_in={seq_in}"
            #     keys["seq_in"] = inputs[str(int(self.name)-1)].seq[-1]
            #     msg += " | ts_sent={ts_sent}"
            #     keys["ts_sent"] = inputs[str(int(self.name) - 1)].ts_sent[-1]
            #     msg += " | ts_recv={ts_recv}"
            #     keys["ts_recv"] = inputs[str(int(self.name) - 1)].ts_recv[-1]
            #     msg += " | u={u}"
            #     keys["u"] = u[str(int(self.name) - 1)]
            #     # msg += " | seq_mod={seq_mod}"
            #     # keys["seq_mod"] = seq_mod
            #     msg += " | y_target={y_target}"
            #     keys["y_target"] = output_target.y
            #     msg += " y={y}"
            #     keys["y"] = output.y
            #     msg += " | loss={loss}"
            #     keys["loss"] = new_loss
            #     jax.debug.print(msg, **keys)
            new_state = new_state.replace(loss=new_loss)

        # Update step_state (observation and action history)
        new_step_state = step_state.replace(rng=rng, state=new_state)
        return new_step_state, output

