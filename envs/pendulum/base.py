from typing import Union, Any
import jax
from flax import struct

from rexv2 import base


@struct.dataclass
class SensorParams(base.Base):
    sensor_delay: base.TrainableDist


@struct.dataclass
class ActuatorParams(base.Base):
    actuator_delay: base.TrainableDist


@struct.dataclass
class WorldParams(base.Base):
    actuator_delay: base.TrainableDist


@struct.dataclass
class WorldState(base.Base):
    """Pendulum state definition"""
    th: Union[float, jax.typing.ArrayLike]
    thdot: Union[float, jax.typing.ArrayLike]


@struct.dataclass
class ActuatorOutput(base.Base):
    """Pendulum actuator output"""
    action: jax.typing.ArrayLike  # Torque to apply to the pendulum
    state_estimate: Any = struct.field(default=None)  # Estimated state at the time the action is executed in the world


@struct.dataclass
class SensorOutput(base.Base):
    """Pendulum sensor output definition with timestamp of expected time of world state corresponding to the sensor output"""
    th: Union[float, jax.typing.ArrayLike]
    thdot: Union[float, jax.typing.ArrayLike]
    ts: Union[float, jax.typing.ArrayLike]
