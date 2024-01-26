from typing import TypeVar, Tuple, Callable, Any, Union, Dict, Sequence
import jax
import jax.numpy as jnp
from rex.jax_utils import tree_extend

Action = TypeVar("Action")
Output = TypeVar("Output")
State = TypeVar("State")
Params = TypeVar("Params")

InitBackend = Callable[[float, float, Any], Params]
InitSystem = Callable[[Params], Params]
InitPipeline = Callable[[Params, State], State]
Residual = Callable[[Params, Tuple[Params, State, Action, Output]], Union[Output, Any]]
ResidualArgs = Tuple[Params, State, Action, Output]
Step = Callable[[Params, State, Action], Tuple[State, Output]]
GetOutput = Callable[[Params, State], Output]


# @struct.dataclass
class Backend:
    name: str
    init_backend: InitBackend
    init_sys: InitSystem
    init_pipeline: InitPipeline
    step: Step
    get_output: GetOutput

    def rollout(self, params: Params, init_s: State, actions: Action) -> Tuple[State, Output]:
        init_y = self.get_output(params, init_s)

        def loop_body(s: State, a: Action) -> Tuple[State, Output]:
            new_s, y = self.step(params, s, a)
            return new_s, y

        final_state, ys = jax.lax.scan(loop_body, init_s, actions)
        init_y_ys = jax.tree_util.tree_map(lambda _init_y, _ys: jnp.concatenate((_init_y[None], _ys), axis=0), init_y, ys)
        return final_state, init_y_ys


def convert_wxyz_to_xyzw(quat: jax.typing.ArrayLike):
    """Convert quaternion (w,x,y,z) -> (x,y,z,w)"""
    return jnp.array([quat[1], quat[2], quat[3], quat[0]], dtype="float32")


def save(path, json_rollout):
    """Saves trajectory as an HTML text file."""
    from etils import epath
    path = epath.Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    path.write_text(json_rollout)


# def log_transformed(fn: Callable, where: Union[bool, None, Sequence[Any]] = True, pre: bool = False, post: bool = False) -> Callable:
#     """Decorator for log-transforming parameters before passing them to a function."""
#     def wrapped(*args) -> Params:
#         where_tree = tree_extend(args, where)
#
#
#         log_params = jax.tree_util.tree_map(jnp.log, params)
#         res = fn(log_params)
#         return res
#     return wrapped


def exp_transformed(fn: Callable, where: Union[bool, None, Sequence[Any]] = True, pre: bool = False, post: bool = False) -> Callable:
    """Decorator for exp-transforming parameters before passing them to a function."""
    def wrapped(*args) -> Params:
        where_extended = tree_extend(args, where)
        exp_args = jax.tree_util.tree_map(lambda x, y: jnp.where(x, jnp.exp(y), y), where_extended, args)
        return fn(*exp_args)
    return wrapped