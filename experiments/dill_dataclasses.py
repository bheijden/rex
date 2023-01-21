import flax.struct as struct
import jumpy.numpy as jp


@struct.dataclass
class Params:
    p: jp.float32
    t: jp.float32
    a: str = struct.field(pytree_node=False, default_factory=lambda: "a")


@struct.dataclass
class State:
    th: jp.float32
    thdot: jp.float32


@struct.dataclass
class StepState:
    state: State
    params: Params