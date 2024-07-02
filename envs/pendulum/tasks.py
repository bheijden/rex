import functools
from typing import Callable, Union, Dict, Any, Tuple
import equinox as eqx
import jax
import numpy as onp
from flax import struct
from jax import numpy as jnp
from matplotlib import pyplot as plt

import rexv2.base as base
from rexv2.graph import Graph
import rexv2.evo as evo


@struct.dataclass
class Task:
    init_params: Dict[str, base.Params]
    trans: base.Transform  # = struct.field(pytree_node=False)
    max_steps: int = struct.field(pytree_node=False)
    loss_filter: Dict[str, base.State] = struct.field(pytree_node=False)
    rollout: Callable[[Graph, Dict[str, base.Params], jax.Array], base.GraphState] = struct.field(pytree_node=False)
    plot: Callable[[base.GraphState, str], Any] = struct.field(pytree_node=False)
    solver: evo.EvoSolver = struct.field(pytree_node=False)
    graph: Graph = struct.field(pytree_node=False)
    description: str = struct.field(pytree_node=False)

    @classmethod
    def init(cls, graph: Graph, solver: evo.EvoSolver, max_steps: int, description: str,
             init_params: Dict[str, base.Params], trans: base.Transform, loss_filter: Dict[str, base.State],
             rollout: Callable[[Graph, Dict[str, base.Params], jax.Array], base.GraphState] = None,
             plot: Callable[[base.GraphState, str], Any] = None):
        return cls(init_params=init_params, trans=trans, max_steps=max_steps, loss_filter=loss_filter,
                   rollout=rollout, plot=plot, solver=solver, graph=graph, description=description)

    def solve(self, graph_state: base.GraphState, rng: jax.Array = None, init_sol_state: evo.EvoState = None, max_steps: int = None,
              verbose: bool = True, logger: evo.LogState = None) -> Tuple[evo.EvoState, base.Params, evo.LogState]:
        max_steps = max_steps or self.max_steps
        rng = jax.random.PRNGKey(0) if rng is None else rng
        logger = logger or self.solver.init_logger(num_generations=max_steps)

        # Get base structure for params
        base_params = {name: p for name, p in graph_state.params.items()}  # Get base structure for params

        # Create loss
        loss = make_loss(self.graph, self.rollout, self.loss_filter)

        # Initialize transforms
        extend = base.Extend.init(base_params, self.init_params)  # todo: Transform that matches dtype of base_params? x0, y0?
        trans_extend = base.Chain.init(self.trans, extend)

        # Initialize solver state
        init_params = self.trans.inv(self.init_params)
        init_sol_state = self.solver.init_state(init_params) if init_sol_state is None else init_sol_state

        # Run solver
        sol_state, log_state, losses = evo.evo(loss, self.solver, init_sol_state, (trans_extend,),
                                               max_steps=max_steps, rng=rng, verbose=verbose, logger=logger)
        opt_params = self.solver.unflatten(sol_state.best_member)
        if verbose:
            class _PrettyPrint:
                def __init__(self, xt, x):
                    self.xt = onp.array(xt)
                    self.x = x

                def __repr__(self):
                    """Relative to the transformed value."""
                    return f"{self.xt} Rel({self.x:.2f})"

            opt_params_trans = self.trans.apply(opt_params)
            extend = base.Extend.init(opt_params_trans, opt_params)
            opt_params_trans_inv = extend.inv(opt_params_trans)  # Filters shared parameters
            pp = jax.tree_util.tree_map(lambda _xt, _x: _PrettyPrint(_xt, _x), opt_params_trans_inv, opt_params)
            # Pretty print the parameters
            _ = eqx.tree_pprint(pp)
            # msg = ""
            # pdict = {}
            # for key, value in pp.items():
            #     leaves = jax.tree_util.tree_leaves(value)
            #     if len(leaves) > 0:
            #         pdict[key] = value
            #         msg += f"{key}: " + "{"f"{key}""}\n"
            # print((f"{self.description}: \n" + msg).format(**pdict))
            # jax.debug.print(f"{self.description}: \n" + msg, **pdict)
        opt_params_trans_ex = trans_extend.apply(opt_params)
        return sol_state, opt_params_trans_ex, log_state

    def to_extended_params(self, graph_state: base.GraphState, opt_params: Dict[str, base.Params]) -> Dict[str, base.Params]:
        # Get base structure for params
        base_params = {name: p for name, p in graph_state.params.items()}  # Get base structure for params

        # Convert params to step_states
        extend = base.Extend.init(base_params, opt_params)

        # Extend with base params
        extend_params = extend.apply(opt_params)
        return extend_params

    def evaluate(self, params: Dict[str, base.Params], rng: jax.Array = None, eps: int = 0, max_steps: int = None) -> base.GraphState:
        if rng is None:
            rng = jax.random.PRNGKey(0)

        # Initialize
        graph_state = self.graph.init(rng, params, starting_eps=eps, order=("supervisor",))  # todo: specific to pendulum

        # Rollout
        graph_states = self.graph.rollout(graph_state, eps=eps, max_steps=max_steps, carry_only=False)
        return graph_states


def make_loss(graph: Graph, rollout: Callable, loss_filter: base.Filter) -> base.Loss:
    def _loss(opt_params: Dict[str, base.Params], args: base.LossArgs, rng: jax.Array = None) -> Union[float, jax.Array]:
        if rng is None:
            rng = jax.random.PRNGKey(0)

        # Unpack args
        trans, = args

        # Extend with base params
        params_extend = trans.apply(opt_params)

        # Get rollout
        final_gs = rollout(graph, params_extend, rng)

        # Get states
        states = {name: s for name, s in final_gs.state.items()}

        # Filter states
        tree_loss = eqx.filter(states, loss_filter)

        # Sum losses
        leaves, _ = jax.tree_util.tree_flatten(tree_loss)
        cum_loss = jnp.array(0.) if len(leaves) == 0 else functools.reduce(jnp.add, leaves).sum()
        return 0.5 * cum_loss.real

    return _loss


def create_control_task(graph: Graph, graph_state: base.GraphState, strategy: str = "OpenES"):
    num_train_rollouts = 40

    # Create an empty filter
    base_states = {name: s for name, s in graph_state.state.items()}  # Converts from FrozenDict
    loss_filter = jax.tree_util.tree_map(lambda x: False, base_states)
    loss_filter["world"] = loss_filter["world"].replace(loss_task=True)  # Task loss

    # Create an empty skeleton of the params
    base_params = {name: params for name, params in graph_state.params.items()}  # Get base structure for params
    base_params["supervisor"] = base_params["supervisor"].replace()
    init_params = jax.tree_util.tree_map(lambda x: None, base_params)  # Create an empty skeleton of the params
    u_min = jax.tree_util.tree_map(lambda x: None, base_params)
    u_max = jax.tree_util.tree_map(lambda x: None, base_params)

    # Controller
    init_params["controller"] = base_params["controller"].replace(max_torque=None, min_torque=None)
    u_min["controller"] = jax.tree_util.tree_map(lambda x: jnp.ones_like(x)*-1e10, init_params["controller"])
    u_max["controller"] = jax.tree_util.tree_map(lambda x: jnp.ones_like(x)*1e10, init_params["controller"])

    def rollout_fn(graph: Graph, params: Dict[str, base.Params], rng: jax.Array = None) -> base.GraphState:
        rngs = jax.random.split(rng, num=num_train_rollouts)

        # Initialize graph state
        graph_init = functools.partial(graph.init, order=("supervisor", "actuator"))
        vmap_graph_init = jax.vmap(graph_init, in_axes=(0, None))
        gs = vmap_graph_init(rngs, params)

        # Rollout graph (5 seconds)
        graph_rollout = functools.partial(graph.rollout, max_steps=int(5 * graph.supervisor.rate), carry_only=True)
        final_gs = jax.vmap(graph_rollout, in_axes=(0,))(gs)
        final_gs = jax.tree_util.tree_map(lambda x: x.mean(axis=0), final_gs)
        return final_gs

    def plot_fn(graph_states: base.GraphState, identifier: str = "") -> Any:
        # Plot best result
        figs = []
        fig, axes = plt.subplots(nrows=2)
        figs.append(fig)
        ts_world = graph_states.ts["world"] + 1/graph.nodes["world"].rate
        th_world = graph_states.state["world"].th
        axes[0].plot(ts_world, th_world, label="world")
        if "estimator" in graph_states.inputs:
            axes[0].plot(graph_states.inputs["estimator"]["sensor"].ts_recv[:, 0],
                         graph_states.inputs["estimator"]["sensor"].data.th[:, 0], label="sensor")
        if "controller" in graph_states.inputs:
            if "estimator" in graph_states.inputs["controller"]:
                axes[0].plot(graph_states.inputs["controller"]["estimator"].ts_recv[:, 0],
                             graph_states.inputs["controller"]["estimator"].data.mean.th[:, 0], label="estimator")
            elif "world" in graph_states.inputs["controller"]:
                axes[0].plot(graph_states.inputs["controller"]["world"].ts_recv[:, 0],
                             graph_states.inputs["controller"]["world"].data.th[:, 0], label="world")
        if "actuator" in graph_states.inputs:
            axes[0].plot(graph_states.inputs["actuator"]["controller"].ts_recv[:, 0],
                         graph_states.inputs["actuator"]["controller"].data.state_estimate.mean.th[:, 0], label="controller")
        if "actuator" in graph_states.inputs["world"]:
            axes[0].plot(graph_states.inputs["world"]["actuator"].ts_recv[:, 0],
                         graph_states.inputs["world"]["actuator"].data.state_estimate.mean.th[:, 0], label="actuator")
        axes[0].set(ylabel="th")
        axes[0].legend()
        thdot_world = graph_states.state["world"].thdot
        axes[1].plot(ts_world, thdot_world, label="world")
        axes[1].set(ylabel="thdot", ylim=[-30, 30])
        fig.suptitle(f"{identifier}(th, thdot)")
        # plt.show()
        return figs

    # Initialize solver
    fitness_kwargs = dict(maximize=False, centered_rank=True, z_score=False, w_decay=0.0)
    strategies = {
        "OpenES": dict(popsize=300, use_antithetic_sampling=True, opt_name="adam",
                       lrate_init=0.27, lrate_decay=1.0, lrate_limit=0.27,
                       sigma_init=0.14400870142450048, sigma_decay=1.0, sigma_limit=0.14400870142450048, mean_decay=0.0),
        "CMA_ES": dict(popsize=200, elite_ratio=0.1, sigma_init=0.5, mean_decay=0.),
    }

    trans = base.Identity.init()
    solver = evo.EvoSolver.init(u_min, u_max, strategy, strategies[strategy], fitness_kwargs)
    task = Task.init(graph=graph, solver=solver, max_steps=300, description="Control Task",
                     init_params=init_params, trans=trans, loss_filter=loss_filter, rollout=rollout_fn, plot=plot_fn)
    return task


def create_sysid_task(graph: Graph, graph_state: base.GraphState, strategy: str = "CMA_ES"):
    MIN, MAX = 0.5, 1.5

    # Create an empty filter
    base_states = {name: s for name, s in graph_state.state.items()}
    loss_filter = jax.tree_util.tree_map(lambda x: False, base_states)
    # loss_filter["world"] = loss_filter["world"].replace(loss_th=True, loss_thdot=True)  # Estimator loss
    # loss_filter["world"] = loss_filter["world"].replace(loss_task=True)  # Task loss
    loss_filter["world"] = loss_filter["world"].replace(loss_ts=True)  # Reconstruction loss
    # loss_filter["sensor"] = loss_filter["sensor"].replace(loss_th=True, loss_thdot=True)  # Reconstruction loss
    loss_filter["camera"] = loss_filter["camera"].replace(loss_th=True)  # Reconstruction loss
    # loss_filter["world"] = loss_filter["world"].replace(loss_task=True)  # Controller loss

    # Create an empty skeleton of the params
    base_params = {name: p for name, p in graph_state.params.items()}  # Get base structure for params
    init_params = jax.tree_util.tree_map(lambda x: None, base_params)  # Create an empty skeleton of the params
    u_min = jax.tree_util.tree_map(lambda x: None, base_params)
    u_max = jax.tree_util.tree_map(lambda x: None, base_params)
    shared = []  # Holds the list of shared params

    # World
    init_params["world"] = base_params["world"].replace(max_speed=None, actuator_delay=None)
    u_min["world"] = jax.tree_util.tree_map(lambda x: x * MIN, init_params["world"])
    u_max["world"] = jax.tree_util.tree_map(lambda x: x * MAX, init_params["world"])
    shared.append(base.Shared.init(where=lambda _p: _p["world"].actuator_delay,
                                   replace_fn=lambda _p: _p["actuator"].actuator_delay))

    # Supervisor
    # init_params["supervisor"] = base_params["supervisor"].replace(max_th=None, min_th=None, max_thdot=None, min_thdot=None)
    # init_params["supervisor"] = init_params["supervisor"].replace(init_states=None)  # Optimize for initial states.
    # u_min["supervisor"] = jax.tree_util.tree_map(lambda x: x * MIN, init_params["supervisor"])
    # u_max["supervisor"] = jax.tree_util.tree_map(lambda x: x * MAX, init_params["supervisor"])

    # Sensor
    init_params["sensor"] = base_params["sensor"].replace()
    u_min["sensor"] = jax.tree_util.tree_map(lambda x: x * MIN, init_params["sensor"])
    u_max["sensor"] = jax.tree_util.tree_map(lambda x: x * MAX, init_params["sensor"])
    u_min["sensor"] = eqx.tree_at(lambda _min: _min.sensor_delay.alpha, u_min["sensor"], 0.)
    u_max["sensor"] = eqx.tree_at(lambda _max: _max.sensor_delay.alpha, u_max["sensor"], 1.)

    # Detector
    camera = graph.nodes["camera"]
    detector = base_params["camera"].detector
    median = onp.array(camera._outputs.median).reshape(-1, 2)
    a, b, x0, y0, phi = detector.estimate_ellipse(median)
    print(f"Initial guess: a={a}, b={b}, x0={x0}, y0={y0}, phi={phi}")
    init_detector = detector.replace()
    init_detector = init_detector.replace(a=a, b=b, x0=x0, y0=y0, phi=phi)  # Transformation to pendulum angle
    init_detector = init_detector.replace(min_max_threshold=None, lower_bgr=None, upper_bgr=None,
                                          sigma=None, binarization_threshold=None)  # Image processing
    # init_detector = init_detector.replace(a=None, b=None, x0=None, y0=None, phi=None, theta_offset=None)  # Transformation to pendulum angle
    init_detector = init_detector.replace(wn=None)  # Low-pass filter
    u_min_detector = jax.tree_util.tree_map(lambda x: x * 0.9, init_detector)  # todo: Use MIN?
    u_max_detector = jax.tree_util.tree_map(lambda x: x * 1.1, init_detector)  # todo: use MAX?
    u_min_detector = eqx.tree_at(lambda _min: _min.phi, u_min_detector, -2 * onp.pi)
    u_max_detector = eqx.tree_at(lambda _max: _max.phi, u_max_detector, 2 * onp.pi)
    u_min_detector = eqx.tree_at(lambda _min: _min.theta_offset, u_min_detector, -2 * onp.pi)
    u_max_detector = eqx.tree_at(lambda _max: _max.theta_offset, u_max_detector, 2 * onp.pi)

    # Camera
    # todo: add measurement noise to camera & connect with estimator
    init_params["camera"] = base_params["camera"].replace(detector=None)  # Add detector after setting other camera params
    u_min["camera"] = jax.tree_util.tree_map(lambda x: x * MIN, init_params["camera"])
    u_max["camera"] = jax.tree_util.tree_map(lambda x: x * MAX, init_params["camera"])
    u_min["camera"] = eqx.tree_at(lambda _min: _min.sensor_delay.alpha, u_min["camera"], 0.)
    u_max["camera"] = eqx.tree_at(lambda _max: _max.sensor_delay.alpha, u_max["camera"], 1.)
    # Add detector
    init_params["camera"] = init_params["camera"].replace(detector=init_detector)
    u_min["camera"] = u_min["camera"].replace(detector=u_min_detector)
    u_max["camera"] = u_max["camera"].replace(detector=u_max_detector)

    # Estimator
    from envs.pendulum.estimator import UKFOde
    init_params["estimator"] = base_params["estimator"].replace(dt_future=0.0, ode=None)
    u_min["estimator"] = jax.tree_util.tree_map(lambda x: x * MIN, init_params["estimator"])
    u_max["estimator"] = jax.tree_util.tree_map(lambda x: x * MAX, init_params["estimator"])
    u_min["estimator"] = eqx.tree_at(lambda _min: _min.dt_future, u_min["estimator"], 0.)
    u_max["estimator"] = eqx.tree_at(lambda _max: _max.dt_future, u_max["estimator"], 0.1)
    replace_fn = lambda _p: UKFOde(**_p["world"].__dict__).replace(actuator_delay=None)
    shared.append(base.Shared.init(where=lambda _p: _p["estimator"].ode, replace_fn=replace_fn))

    # Controller
    # init_params["controller"] = base_params["controller"].replace(max_torque=None, min_torque=None)
    # u_min["controller"] = jax.tree_util.tree_map(lambda x: x * MIN, init_params["controller"])
    # u_max["controller"] = jax.tree_util.tree_map(lambda x: x * MAX, init_params["controller"])

    # Actuator
    init_params["actuator"] = base_params["actuator"].replace()
    u_min["actuator"] = jax.tree_util.tree_map(lambda x: x * MIN, init_params["actuator"])
    u_max["actuator"] = jax.tree_util.tree_map(lambda x: x * MAX, init_params["actuator"])
    u_min["actuator"] = eqx.tree_at(lambda _min: _min.actuator_delay.alpha, u_min["actuator"], 0.)
    u_max["actuator"] = eqx.tree_at(lambda _max: _max.actuator_delay.alpha, u_max["actuator"], 1.)

    def rollout_fn(graph: Graph, params: Dict[str, base.Params], rng: jax.Array = None) -> base.GraphState:
        # Initialize graph state
        gs = graph.init(rng=rng, params=params, order=("supervisor", "actuator"))

        # Rollout graph
        final_gs = graph.rollout(gs, carry_only=True)
        return final_gs

    def plot_fn(graph_states: base.GraphState, identifier: str = "") -> Any:
        # Plot best result
        figs = []
        fig, axes = plt.subplots(nrows=2)
        figs.append(fig)
        ts_sensor = graph_states.inputs["estimator"]["sensor"].ts_sent[:, 0]
        ts_world = graph_states.ts["world"]
        ts_camera = graph_states.state["camera"].tsn_1
        ts_estimator = graph_states.inputs["controller"]["estimator"].data.ts[:, -1]

        # wrap_unwrap = lambda x: onp.unwrap((x + onp.pi) % (2 * onp.pi) - onp.pi, discont=onp.pi)
        wrap_unwrap = lambda x: onp.sin(x)
        th_sensor = graph_states.inputs["estimator"]["sensor"].data.th[:, 0]
        th_sensor = wrap_unwrap(th_sensor)
        th_world = graph_states.state["world"].th
        th_world = wrap_unwrap(th_world)
        th_camera = jnp.unwrap(graph_states.state["camera"].thn_1)
        th_camera = wrap_unwrap(th_camera)
        th_estimator = graph_states.inputs["controller"]["estimator"].data.mean.th[:, 0]
        th_estimator = wrap_unwrap(th_estimator)
        axes[0].plot(ts_world, th_world, label="world")
        axes[0].plot(ts_sensor, th_sensor, label="sensor")
        axes[0].plot(ts_camera, th_camera, label="camera")
        axes[0].plot(ts_estimator, th_estimator, label="estimator")
        axes[0].set(ylabel="th")
        axes[0].legend()
        thdot_sensor = graph_states.inputs["estimator"]["sensor"].data.thdot[:, 0]
        thdot_world = graph_states.state["world"].thdot
        thdot_camera = graph_states.state["camera"].yn_1
        thdot_estimator = graph_states.inputs["controller"]["estimator"].data.mean.thdot[:, 0]
        axes[1].plot(ts_world, thdot_world, label="world")
        axes[1].plot(ts_sensor, thdot_sensor, label="sensor")
        axes[1].plot(ts_camera, thdot_camera, label="camera")
        axes[1].plot(ts_estimator, thdot_estimator, label="estimator")
        # axes[1].plot(outputs_sysid["camera"].ts[EPS_IDX], outputs_sysid["camera"].thdot[EPS_IDX], label="camera (data)")
        axes[1].set(ylabel="thdot", ylim=[-30, 30])
        axes[1].legend()
        fig.suptitle(f"{identifier}(th, thdot)")

        # Plot sin cos
        # fig, axes = plt.subplots(nrows=2)
        # ts_world = graph_states.nodes["world"].ts
        # th_world = graph_states.nodes["world"].state.th
        # ts_camera = graph_states.nodes["camera"].state.tsn_1
        # th_camera = jnp.unwrap(graph_states.nodes["camera"].state.thn_1)
        # axes[0].plot(ts_world, jnp.sin(th_world), label="world")
        # axes[0].plot(ts_sensor, jnp.sin(th_sensor), label="sensor")
        # axes[0].plot(ts_camera, jnp.sin(th_camera), label="camera")
        # axes[1].plot(ts_world, jnp.cos(th_world), label="world")
        # axes[1].plot(ts_sensor, jnp.cos(th_sensor), label="sensor")
        # axes[1].plot(ts_camera, jnp.cos(th_camera), label="camera")
        # axes[0].legend()
        # fig.suptitle(f"{identifier}(sin, cos)")
        figs.append(fig)
        return figs

    # Initialize solver
    strategies = {
        # todo: optimize OpenES
        "OpenES": dict(popsize=200, use_antithetic_sampling=True, opt_name="adam",
                       lrate_init=0.125, lrate_decay=0.999, lrate_limit=0.001,
                       sigma_init=0.05, sigma_decay=0.999, sigma_limit=0.01, mean_decay=0.0),
        "CMA_ES": dict(popsize=200, elite_ratio=0.1, sigma_init=0.25, mean_decay=0.),
    }

    denorm = base.Denormalize.init(u_min, u_max)
    trans = base.Chain.init(denorm, *shared)
    solver = evo.EvoSolver.init(denorm.normalize(u_min), denorm.normalize(u_max), strategy, strategies[strategy])
    task = Task.init(graph=graph, solver=solver, max_steps=300, description="System Identification",
                     init_params=init_params, trans=trans, loss_filter=loss_filter, rollout=rollout_fn, plot=plot_fn)
    return task


def create_estimator_task(graph: Graph, graph_state: base.GraphState, strategy: str = "CMA_ES"):
    MIN, MAX = 0.5, 1.5

    # Create an empty filter
    base_states = {name: s for name, s in graph_state.state.items()}
    loss_filter = jax.tree_util.tree_map(lambda x: False, base_states)
    loss_filter["world"] = loss_filter["world"].replace(loss_th=True, loss_thdot=True)  # Estimator loss
    # loss_filter["world"] = loss_filter["world"].replace(loss_task=True)  # Task loss
    # loss_filter["world"] = loss_filter["world"].replace(loss_ts=True)  # Reconstruction loss
    # loss_filter["sensor"] = loss_filter["sensor"].replace(loss_th=True, loss_thdot=True)  # Reconstruction loss
    # loss_filter["detector"] = loss_filter["detector"].replace(loss_th=True)  # Reconstruction loss
    # loss_filter["world"] = loss_filter["world"].replace(loss_task=True)  # Controller loss

    # Create an empty skeleton of the params
    base_params = {name: p for name, p in graph_state.params.items()}  # Get base structure for params
    init_params = jax.tree_util.tree_map(lambda x: None, base_params)  # Create an empty skeleton of the params
    u_min = jax.tree_util.tree_map(lambda x: None, base_params)
    u_max = jax.tree_util.tree_map(lambda x: None, base_params)

    # World
    # init_params["world"] = base_params["world"].replace(max_speed=None)
    # u_min["world"] = jax.tree_util.tree_map(lambda x: x * MIN, init_params["world"])
    # u_max["world"] = jax.tree_util.tree_map(lambda x: x * MAX, init_params["world"])
    # u_min["world"] = eqx.tree_at(lambda _min: _min.actuator_delay.alpha, u_min["world"], 0.)
    # u_max["world"] = eqx.tree_at(lambda _max: _max.actuator_delay.alpha, u_max["world"], 1.)

    # Supervisor
    # init_params["supervisor"] = base_params["supervisor"].replace(max_th=None, min_th=None, max_thdot=None, min_thdot=None)
    # init_params["supervisor"] = init_params["supervisor"].replace(init_states=None)  # Optimize for initial states.
    # u_min["supervisor"] = jax.tree_util.tree_map(lambda x: x * MIN, init_params["supervisor"])
    # u_max["supervisor"] = jax.tree_util.tree_map(lambda x: x * MAX, init_params["supervisor"])

    # Camera
    # init_params["camera"] = base_params["camera"].replace()
    # u_min["camera"] = jax.tree_util.tree_map(lambda x: x * MIN, init_params["camera"])
    # u_max["camera"] = jax.tree_util.tree_map(lambda x: x * MAX, init_params["camera"])
    # u_min["camera"] = eqx.tree_at(lambda _min: _min.sensor_delay.alpha, u_min["camera"], 0.)
    # u_max["camera"] = eqx.tree_at(lambda _max: _max.sensor_delay.alpha, u_max["camera"], 1.)

    # Sensor
    # init_params["sensor"] = base_params["sensor"].replace()
    # u_min["sensor"] = jax.tree_util.tree_map(lambda x: x * MIN, init_params["sensor"])
    # u_max["sensor"] = jax.tree_util.tree_map(lambda x: x * MAX, init_params["sensor"])
    # u_min["sensor"] = eqx.tree_at(lambda _min: _min.sensor_delay.alpha, u_min["sensor"], 0.)
    # u_max["sensor"] = eqx.tree_at(lambda _max: _max.sensor_delay.alpha, u_max["sensor"], 1.)

    # Detection (detection)
    init_params["detector"] = init_params["detector"].replace(wn=5)# Low-pass filter
    u_min["detector"] = init_params["detector"].replace(wn=0.1)
    u_max["detector"] = init_params["detector"].replace(wn=20)

    # Estimator
    init_params["estimator"] = base_params["estimator"].replace(dt_future=None)
    u_min["estimator"] = jax.tree_util.tree_map(lambda x: x * MIN, init_params["estimator"])
    u_max["estimator"] = jax.tree_util.tree_map(lambda x: x * MAX, init_params["estimator"])

    # Controller
    # init_params["controller"] = base_params["controller"].replace(max_torque=None, min_torque=None)
    # u_min["controller"] = jax.tree_util.tree_map(lambda x: x * MIN, init_params["controller"])
    # u_max["controller"] = jax.tree_util.tree_map(lambda x: x * MAX, init_params["controller"])

    # Actuator
    # init_params["actuator"] = base_params["actuator"].replace()
    # u_min["actuator"] = jax.tree_util.tree_map(lambda x: x * MIN, init_params["actuator"])
    # u_max["actuator"] = jax.tree_util.tree_map(lambda x: x * MAX, init_params["actuator"])

    def rollout_fn(graph: Graph, params: Dict[str, base.Params], rng: jax.Array = None) -> base.GraphState:
        # Initialize graph state
        gs = graph.init(rng=rng, params=params, order=("supervisor", "actuator"))

        # Rollout graph
        final_gs = graph.rollout(gs, carry_only=True)
        return final_gs

    def plot_fn(graph_states: base.GraphState, identifier: str = "") -> Any:
        # Plot best result
        figs = []
        fig, axes = plt.subplots(nrows=2)
        figs.append(fig)
        ts_sensor = graph_states.inputs["estimator"]["sensor"].ts_sent[:, 0]
        ts_world = graph_states.ts["world"]
        ts_detector = graph_states.state["detector"].tsn_1
        th_sensor = graph_states.inputs["estimator"]["sensor"].data.th[:, 0] - graph_states.inputs["estimator"]["sensor"].data.th[1, 0]
        th_world = graph_states.state["world"].th
        th_detector = jnp.unwrap(graph_states.state["detector"].thn_1)
        axes[0].plot(ts_world, th_world, label="world")
        axes[0].plot(ts_sensor, th_sensor, label="sensor")
        axes[0].plot(ts_detector, th_detector, label="detector")
        axes[0].set(ylabel="th")
        axes[0].legend()
        thdot_sensor = graph_states.inputs["estimator"]["sensor"].data.thdot[:, 0]
        thdot_world = graph_states.state["world"].thdot
        thdot_detector = graph_states.state["detector"].yn_1
        axes[1].plot(ts_world, thdot_world, label="world")
        axes[1].plot(ts_sensor, thdot_sensor, label="sensor")
        axes[1].plot(ts_detector, thdot_detector, label="detector (low-pass)")
        # axes[1].plot(outputs_sysid["detector"].ts[EPS_IDX], outputs_sysid["detector"].thdot[EPS_IDX], label="detector (data)")
        axes[1].set(ylabel="thdot", ylim=[-30, 30])
        axes[1].legend()
        fig.suptitle(f"{identifier}(th, thdot)")

        # Plot sin cos
        # fig, axes = plt.subplots(nrows=2)
        # ts_world = graph_states.nodes["world"].ts
        # th_world = graph_states.nodes["world"].state.th
        # ts_detector = graph_states.nodes["detector"].state.tsn_1
        # th_detector = jnp.unwrap(graph_states.nodes["detector"].state.thn_1)
        # axes[0].plot(ts_world, jnp.sin(th_world), label="world")
        # axes[0].plot(ts_sensor, jnp.sin(th_sensor), label="sensor")
        # axes[0].plot(ts_detector, jnp.sin(th_detector), label="detector")
        # axes[1].plot(ts_world, jnp.cos(th_world), label="world")
        # axes[1].plot(ts_sensor, jnp.cos(th_sensor), label="sensor")
        # axes[1].plot(ts_detector, jnp.cos(th_detector), label="detector")
        # axes[0].legend()
        # fig.suptitle(f"{identifier}(sin, cos)")
        figs.append(fig)
        return figs

    # Initialize solver
    strategies = {
        "OpenES": dict(popsize=200, use_antithetic_sampling=True, opt_name="adam",
                       lrate_init=0.125, lrate_decay=0.999, lrate_limit=0.001,
                       sigma_init=0.05, sigma_decay=0.999, sigma_limit=0.01, mean_decay=0.0),
        "CMA_ES": dict(popsize=200, elite_ratio=0.1, sigma_init=0.5, mean_decay=0.),
    }
    denorm = base.Denormalize.init(u_min, u_max)
    solver = evo.EvoSolver.init(denorm.normalize(u_min), denorm.normalize(u_max), strategy, strategies[strategy])
    task = Task.init(graph=graph, solver=solver, max_steps=10, description="Low-pass filter",
                     init_params=init_params, trans=denorm, loss_filter=loss_filter, rollout=rollout_fn, plot=plot_fn)
    return task
