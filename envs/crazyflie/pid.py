from typing import Any, Dict, Tuple, Union, List
import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct
from flax.core import FrozenDict
from rexv2.base import StepState, GraphState, Empty, TrainableDist, Base
from rexv2.node import BaseNode
from rexv2.jax_utils import tree_dynamic_slice


@struct.dataclass
class dFilter:
    samplingRate: float
    cutoffFreq: float


@struct.dataclass
class PidObject:
    """https://github.com/bitcraze/crazyflie-firmware/blob/22fb171c87b6fb78e6e524770d5dcc3544a97abd/src/modules/src/pid.c"""
    desired: float  # set point
    output: float  # previous output
    prevMeasured: float  # previous measurement
    prevError: float  # previous error
    integ: float  # integral
    deriv: float  # derivative
    kp: float  # proportional gain
    ki: float  # integral gain
    kd: float  # derivative gain
    kff: float  # feedforward gain
    outP: float  # proportional output (debugging)
    outI: float  # integral output (debugging)
    outD: float  # derivative output (debugging)
    outFF: float  # feedforward output (debugging)
    iLimit: float  # integral limit, absolute value. '0' means no limit.
    outputLimit: float  # total PID output limit, absolute value. '0' means no limit.
    dt: float  # delta-time dt
    dFilter: dFilter  # filter for D term
    enableDFilter: bool = struct.field(pytree_node=False) # filter for D term enable flag

    @classmethod
    def pidInit(cls, kp: float, ki: float, kd: float, outputLimit: float, iLimit: float,
                dt: float, samplingRate: float, cutoffFreq: float, enableDFilter: bool) -> "PidObject":
        dfilter = dFilter(samplingRate, cutoffFreq)
        return cls(
            desired=0.0,
            output=0.0,
            prevMeasured=0.0,
            prevError=0.0,
            integ=0.0,
            deriv=0.0,
            kp=kp,
            ki=ki,
            kd=kd,
            kff=0.0,
            outP=0.0,
            outI=0.0,
            outD=0.0,
            outFF=0.0,
            iLimit=iLimit,
            outputLimit=outputLimit,
            dt=dt,
            dFilter=dfilter,
            enableDFilter=enableDFilter
        )

    def pidReset(self) -> "PidObject":
        return self.replace(
            desired=0.0,
            output=0.0,
            prevMeasured=0.0,
            prevError=0.0,
            integ=0.0,
            deriv=0.0,
            kff=0.0,
            outP=0.0,
            outI=0.0,
            outD=0.0,
        )

    def pidUpdate(self, desired: float, measured: float) -> "PidObject":
        output = 0.0

        # Calculate error
        error = desired - measured

        # Proportional term
        outP = self.kp * error
        output = output + outP

        # Derivative term
        deriv = (error - self.prevError) / self.dt
        if self.enableDFilter:
            raise NotImplementedError("DFilter not implemented")
        deriv = jnp.nan_to_num(deriv, nan=0.0)
        outD = self.kd * deriv
        output = output + outD

        # Integral term
        integ = self.integ + error * self.dt
        integ_constrained = jnp.clip(integ, -self.iLimit, self.iLimit)
        integ = jnp.where(self.iLimit > 0, integ_constrained, integ)
        outI = self.ki * integ
        output = output + outI

        # constrain output
        output = jnp.nan_to_num(output, nan=0.0)
        output_constrained = jnp.clip(output, -self.outputLimit, self.outputLimit)
        output = jnp.where(self.outputLimit > 0, output_constrained, output)
        return self.replace(
            desired=desired,
            output=output,
            prevMeasured=measured,
            prevError=error,
            integ=integ,
            deriv=deriv,
            outP=outP,
            outI=outI,
            outD=outD,
            outFF=0.0
        )


@struct.dataclass
class dFilter:
    samplingRate: float
    cutoffFreq: float


@struct.dataclass
class LPFObject:
    """A 2-Pole Low-Pass Filter implementation.
    https://github.com/bitcraze/crazyflie-firmware/blob/master/src/utils/src/filter.c
    """
    input: float  # current input to the filter
    output: float  # current output of the filter
    a1: float  # first feedback coefficient
    a2: float  # second feedback coefficient
    b0: float  # feedforward coefficient (current input)
    b1: float  # first feedforward coefficient (first delay)
    b2: float  # second feedforward coefficient (second delay)
    delay_element_1: float  # first delay element
    delay_element_2: float  # second delay element
    dFilter: dFilter  # associated dFilter data for the cutoff frequency and sampling rate

    @classmethod
    def lpfInit(cls, samplingRate: float, cutoffFreq: float) -> "LPFObject":
        dfilter = dFilter(samplingRate, cutoffFreq)
        # Initialize filter coefficients
        fr = samplingRate / cutoffFreq
        ohm = jnp.tan(jnp.pi / fr)
        c = 1.0 + 2.0 * jnp.cos(jnp.pi / 4.0) * ohm + ohm**2
        b0 = ohm**2 / c
        b1 = 2.0 * b0
        b2 = b0
        a1 = 2.0 * (ohm**2 - 1.0) / c
        a2 = (1.0 - 2.0 * jnp.cos(jnp.pi / 4.0) * ohm + ohm**2) / c
        return cls(
            input=0.0,
            output=0.0,
            a1=a1,
            a2=a2,
            b0=b0,
            b1=b1,
            b2=b2,
            delay_element_1=0.0,
            delay_element_2=0.0,
            dFilter=dfilter
        )

    def lpfReset(self) -> "LPFObject":
        return self.replace(
            input=0.0,
            output=0.0,
            delay_element_1=0.0,
            delay_element_2=0.0,
        )

    def lpfUpdate(self, new_input: float) -> "LPFObject":
        delay_element_0 = (new_input - self.delay_element_1 * self.a1 -
                           self.delay_element_2 * self.a2)
        output = (delay_element_0 * self.b0 +
                  self.delay_element_1 * self.b1 +
                  self.delay_element_2 * self.b2)

        return self.replace(
            input=new_input,
            output=output,
            delay_element_2=self.delay_element_1,
            delay_element_1=delay_element_0
        )


@struct.dataclass
class PIDState(Base):
    pidZ: PidObject
    pidVz: PidObject
    lpfPhiRef: LPFObject
    lpfThetaRef: LPFObject
    lpfZRef: LPFObject


@struct.dataclass
class PIDOutput(Base):
    pwm_ref: Union[float, jax.typing.ArrayLike]  # Pwm thrust reference command 10000 to 60000
    phi_ref: Union[float, jax.typing.ArrayLike]  # Phi reference (roll), max: pi/6 rad
    theta_ref: Union[float, jax.typing.ArrayLike]  # Theta reference (pitch), max: pi/6 rad
    psi_ref: Union[float, jax.typing.ArrayLike]  # Psi reference (yaw)
    z_ref: Union[float, jax.typing.ArrayLike]
    state_estimate: Any = struct.field(default=None)  # Estimated state at the time the action is executed in the world
    has_landed: Any = struct.field(default=False)  # True if the agent has landed


@struct.dataclass
class PIDParams(Base):
    actuator_delay: TrainableDist
    sensor_delay: TrainableDist
    # Other
    UINT16_MAX: Union[int, jax.typing.ArrayLike]
    pwm_scale: Union[float, jax.typing.ArrayLike]
    pwm_base: Union[float, jax.typing.ArrayLike]
    pwm_range: Union[float, jax.typing.ArrayLike]
    vel_max_overhead: Union[float, jax.typing.ArrayLike]
    zvel_max: Union[float, jax.typing.ArrayLike]
    # PID
    pidZ: PidObject
    pidVz: PidObject
    # LPF
    lpfPhiRef: LPFObject
    lpfThetaRef: LPFObject
    lpfZRef: LPFObject

    def reset(self) -> PIDState:
        # Replace output limits
        z_outputLimit = jnp.maximum(0.5, self.zvel_max) * self.vel_max_overhead
        vz_outputLimit = self.UINT16_MAX / 2 / self.pwm_scale
        pidZ = self.pidZ.replace(outputLimit=z_outputLimit)
        pidVz = self.pidVz.replace(outputLimit=vz_outputLimit)
        # Reset PID
        pidZ = pidZ.pidReset()
        pidVz = pidVz.pidReset()
        # Reset LPF
        lpfPhiRef = self.lpfPhiRef.lpfReset()
        lpfThetaRef = self.lpfThetaRef.lpfReset()
        lpfZRef = self.lpfZRef.lpfReset()
        return PIDState(pidZ=pidZ, pidVz=pidVz,
                         lpfPhiRef=lpfPhiRef, lpfThetaRef=lpfThetaRef, lpfZRef=lpfZRef)

    def to_command(self, state: PIDState, output: PIDOutput, z: Union[float, jax.Array], vz: Union[float, jax.Array], att: Union[float, jax.Array] = None) -> Tuple[PIDState, PIDOutput]:
        # Get LPF objects
        lpfPhiRef = state.lpfPhiRef
        lpfThetaRef = state.lpfThetaRef
        lpfZRef = state.lpfZRef

        # Apply low-pass filter
        lpfPhiRef = lpfPhiRef.lpfUpdate(output.phi_ref)
        lpfThetaRef = lpfThetaRef.lpfUpdate(output.theta_ref)
        lpfZRef = lpfZRef.lpfUpdate(output.z_ref)
        # output = output.replace(phi_ref=lpfPhiRef.output,  # todo: UNCOMMENT?
        #                         theta_ref=lpfThetaRef.output,
        #                         z_ref=lpfZRef.output)

        # Get PID objects
        pidZ = state.pidZ
        pidVz = state.pidVz

        # subtract platform z, so that the controller is relative to the platform
        z_ref = output.z_ref

        # Run position controller
        pidZ = pidZ.pidUpdate(desired=z_ref, measured=z)
        vz_ref = pidZ.output

        # Run velocity controller
        pidVz = pidVz.pidUpdate(desired=vz_ref, measured=vz)
        pwmRaw = pidVz.output

        # Scale the thrust and add feed forward term
        pwm_unclipped = pwmRaw * self.pwm_scale + self.pwm_base
        pwm = jnp.clip(pwm_unclipped, self.pwm_range[0], self.pwm_range[1])

        # Update output
        new_output = output.replace(pwm_ref=pwm)

        # Update state
        new_state = state.replace(pidZ=pidZ, pidVz=pidVz, lpfPhiRef=lpfPhiRef, lpfThetaRef=lpfThetaRef, lpfZRef=lpfZRef)
        return new_state, new_output


class PID(BaseNode):
    def __init__(self, *args, outputs: Union[PIDOutput, PIDOutput] = None, **kwargs):
        """Initialize.

        Args:
        images: Recorded outputs to be used for system identification.
        """
        super().__init__(*args, **kwargs)
        self._outputs = outputs

    def init_params(self, rng: jax.Array = None, graph_state: GraphState = None) -> PIDParams:
        graph_state = graph_state or GraphState()
        pid_delay = TrainableDist.create(alpha=0., min=0.0, max=0.05)
        sensor_delay = TrainableDist.create(alpha=0., min=0.0, max=0.05)
        # Initialize PID controllers
        UINT16_MAX = 65_535
        zvel_max = 1.0
        vel_max_overhead = 1.1
        pwm_scale = 1_000
        z_outputLimit = jnp.maximum(0.5, zvel_max) * vel_max_overhead
        vz_outputLimit = UINT16_MAX / 2 / pwm_scale
        pidZ = PidObject.pidInit(kp=2.0, ki=0.5, kd=0.0, outputLimit=z_outputLimit, iLimit=1., dt=1/self.rate, samplingRate=self.rate, cutoffFreq=20., enableDFilter=False)
        pidVz = PidObject.pidInit(kp=25., ki=15., kd=0.0, outputLimit=vz_outputLimit, iLimit=5000., dt=1/self.rate, samplingRate=self.rate, cutoffFreq=20., enableDFilter=False)
        # Initialize low-pass filters
        lpfPhiRef = LPFObject.lpfInit(samplingRate=self.rate, cutoffFreq=10.)  # Initialize the Low Pass Filter
        lpfThetaRef = LPFObject.lpfInit(samplingRate=self.rate, cutoffFreq=10.0)  # Initialize the Low Pass Filter
        lpfZRef = LPFObject.lpfInit(samplingRate=self.rate, cutoffFreq=25.0)  # Initialize the Low Pass Filter
        params = PIDParams(
            actuator_delay=pid_delay,
            sensor_delay=sensor_delay,
            UINT16_MAX=UINT16_MAX,
            pwm_scale=pwm_scale,
            pwm_base=40_000.,  # 42_000
            pwm_range=onp.array([20_000, 60_000]),
            vel_max_overhead=vel_max_overhead,
            zvel_max=zvel_max,
            # PID
            pidZ=pidZ,
            pidVz=pidVz,
            # LPF
            lpfPhiRef=lpfPhiRef,
            lpfThetaRef=lpfThetaRef,
            lpfZRef=lpfZRef,
        )
        return params

    def init_state(self, rng: jax.Array = None, graph_state: GraphState = None) -> PIDState:
        graph_state = graph_state or GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        state = params.reset()
        return state

    def init_output(self, rng: jax.Array = None, graph_state: GraphState = None) -> PIDOutput:
        assert "agent" in self.inputs, "No agent connected"
        graph_state = graph_state or GraphState()
        # Get base output
        output = self.inputs["agent"].output_node.init_output(rng, graph_state)
        # Fill initial pwm_ref with pwm_base
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        output = output.replace(pwm_ref=params.pwm_base)
        return output

    def init_inputs(self, rng: jax.Array = None, graph_state: GraphState = None) -> FrozenDict[str, Any]:
        """Default inputs of the node."""
        graph_state = graph_state or GraphState()
        params = graph_state.params.get(self.name, self.init_params(rng, graph_state))
        inputs = super().init_inputs(rng, graph_state).unfreeze()
        if isinstance(inputs["world"].delay_dist, TrainableDist):
            inputs["world"] = inputs["world"].replace(delay_dist=params.sensor_delay)
        return FrozenDict(inputs)

    def step(self, step_state: StepState) -> Tuple[StepState, PIDOutput]:
        """Step the node."""
        # Unpack StepState
        _, state, params, inputs = step_state.rng, step_state.state, step_state.params, step_state.inputs
        state: PIDState
        params: PIDParams

        # Grab ctrl output
        assert len(self.inputs) > 0, "No controller connected to actuator"
        agent_output = inputs["agent"][-1].data
        state_estimate = agent_output.state_estimate

        # Get action from dataset or use passed through.
        if self._outputs is not None:
            output = tree_dynamic_slice(self._outputs, jnp.array([step_state.eps, step_state.seq]))
            output = output.replace(state_estimate=state_estimate)
            new_step_state = step_state  # No state update
        else:
            # Prepare output
            z = inputs["world"][-1].data.pos[-1]
            vz = inputs["world"][-1].data.vel[-1]
            att = inputs["world"][-1].data.att
            new_state, output = params.to_command(state, agent_output, z=z, vz=vz, att=att)

            # Update state
            new_step_state = step_state.replace(state=new_state)

        return new_step_state, output


if __name__ == "__main__":
    UINT16_MAX = 65535  # max value of uint16_t
    thrustScale = 1000
    thrustBase = 36000  #  Approximate throttle needed when in perfect hover. More weight/older battery can use a higher value
    thrustMin = 20000  # Minimum thrust value to output
    velMaxOverhead = 1.1
    zVelMax = 1.0  # m/s

    # z PID controller
    kp = 2.0
    ki = 0.5
    kd = 0.0
    rate = 100.
    dt = 1. / rate
    outputLimit = max(0.5, zVelMax) * velMaxOverhead
    pidZ = PidObject.pidInit(kp, ki, kd, outputLimit, 5000., dt, 100.0, 20.0, False)

    # vz PID controller
    kp = 25.0
    ki = 15.0
    kd = 0.0
    rate = 100.
    dt = 1. / rate
    outputLimit = UINT16_MAX / 2 / thrustScale
    pidVz = PidObject.pidInit(kp, ki, kd, 5000., outputLimit, dt, 100.0, 20.0, False)

    # Reset PID controllers
    pidZ = pidZ.pidReset()
    pidVz = pidVz.pidReset()

    # Run position controller
    z = 1.0   # Current
    vz = 1.0  # Current
    z_desired = 1.05
    pidZ = pidZ.pidUpdate(z_desired, z, True)
    vz_desired = pidZ.output

    # Run velocity controller
    pidVz = pidVz.pidUpdate(vz_desired, vz, True)
    thrustRaw = pidVz.output

    #Scale the thrust and add feed forward term
    thrust = thrustRaw * thrustScale + thrustBase
    thrust = jnp.clip(thrust, thrustMin, UINT16_MAX)
