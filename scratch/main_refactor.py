import functools
import jax
import tqdm
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
import supergraph
import distrax

# Check if we have a GPU
try:
    gpu = jax.devices("gpu")
    print("GPU found!")
except RuntimeError:
    print("Warning: No GPU found, falling back to CPU. Speedups will be less pronounced.")
    print("Hint: if you are using Google Colab, try to change the runtime to GPU: "
          "Runtime -> Change runtime type -> Hardware accelerator -> GPU.")

import rexv2
from rexv2 import utils
from rexv2 import base
from rexv2.constants import Clock, RealTimeFactor, LogLevel, Scheduling, Jitter, Supergraph

if __name__ == "__main__":
    # todo: steps
    #   [DONE] Batch graphs of multiple episodes (utils.to_networkx_graph skip seq=-1).
    #   [DONE] (lightweight) Logging
    #   [DONE] GMM delay identification
    #   [DONE] Augment recorded graphs with simulation nodes (how to deal with delay windows?)
    #   [DONE] Update generate_graphs
    #       [DONE] raise error if unsupported connection (blocking)
    #       [DONE] raise error if TrainableDelay is used? Or warn?
    #       [DONE] Extract computation, communication, phase from node definitions.
    #       [DONE] For trainable delays, convert to Deterministic(loc=delay_min).
    #   [DONE] Modify utils.apply_window to take delay window into account
    #   [DONE] How to increase window of step_state.inputs when simulating delays?
    #   [DONE] Apply trainable window to step_state.inputs in partition_runner.py
    #   [DONE] Use inputs.delay_dist in AsyncGraph instead of Connection.delay_dist?
    #   [DONE] Split step_states in graph_state into params, states, inputs, seq, rng, ts, etc...
    #   [DONE] graph.init(params: Dict[str, Params]) instead of graph.init(step_states: Dict[str, StepState])
    #   [DONE] Refactor PPO, evo, CEM, rl.py
    #   [DONE] Replace delay_dist in node.init_inputs.
    #   [DONE] Add Pendulum (ode, real)
    #   - Refactor tfd to distrax
    #   - Check weaktypes and recompilation & how to compile step function --> leads to more latency (maybe not if jit compiled)?
    #       - Check new graph_state format (especially in AsyncGraph updates of step_states)
    #   - Add Crazyflie (ode, real)
    #   [DONE] flax.serialization.to_bytes does not save static fields. Use pickle instead?
    #   [DONE] Delay identification
    #       [DONE] Change delay_dist to base.StaticDelay
    #       [DONE] Add delay_dist to logging
    #       [DONE] Add delay_dist to InputState
    #       [DONE] Find convenient way to update alphas with trainable parameters in the InputState.delay_dist --> init_inputs.
    #       [DONE] Delay sys_id can only be used for async connections in compiled simulation.
    #       [DONE] When augmenting a graph, we should augment it with the minimum delay, so as to leave room for enlarging the window when compiling.
    #       [DONE] Everywhere we have a trainable distribution, we need to enlarge the window and apply the delay in compilation
    #       [DONE] In simulation, use trainable delay as a conventional distribution

    import rexv2.pendulum as pendulum

    # `color` and `order` arguments are merely for visualization purposes.
    world = pendulum.nodes.OdeWorld(name="world", rate=100, color="grape", order=0,
                                    delay=0.99 / 100, delay_dist=base.StaticDist.create(
            distrax.Deterministic(loc=1 / 100)))  # Uses a set of ODE's to simulate the pendulum
    sensor = pendulum.nodes.Sensor(name="sensor", rate=80, color="pink", order=1,
                                   delay=0.5 / 80, delay_dist=base.StaticDist.create(
            distrax.Normal(loc=0.5 / 80, scale=0.1 / 80)))  # Sensor that reads the pendulum's angle and angular velocity
    agent = pendulum.nodes.RandomAgent(name="agent", rate=20, color="teal", order=3,
                                       delay=0.5 / 20, delay_dist=base.StaticDist.create(
            distrax.Normal(loc=0.5 / 20, scale=0.05 / 20)))  # Agent that generates random actions
    actuator = pendulum.nodes.Actuator(name="actuator", rate=20, color="orange", order=2,
                                       delay=0.5 / 20, delay_dist=base.StaticDist.create(
            distrax.Normal(loc=0.5 / 20, scale=0.05 / 20)))  # Actuator that applies the action to the pendulum
    nodes = dict(world=world, sensor=sensor, agent=agent, actuator=actuator)

    # Connect nodes
    world.connect(actuator, window=1, name="actuator", skip=True,
                  # Skip resolves potential circular dependencies in the artificially generated computation graphs
                  delay=1 / 100, delay_dist=base.StaticDist.create(distrax.Deterministic(loc=0.1 / 100)))
    sensor.connect(world, window=1, name="world",
                   delay=1 / 100, delay_dist=base.StaticDist.create(distrax.Deterministic(loc=0.1 / 100)))
    actuator.connect(agent, window=1, name="agent",
                     delay=1 / 100, delay_dist=base.StaticDist.create(distrax.Deterministic(loc=0.1 / 100)))
    agent.connect(sensor, window=3, name="sensor",
                  delay=1 / 100, delay_dist=base.StaticDist.create(distrax.Deterministic(loc=0.1 / 100)))

    # Set log levels
    utils.set_log_level(LogLevel.WARN)
    utils.set_log_level(LogLevel.DEBUG, world, "blue")
    utils.set_log_level(LogLevel.SILENT, sensor, "blue")
    utils.set_log_level(LogLevel.SILENT, actuator, "blue")
    utils.set_log_level(LogLevel.DEBUG, agent, "cyan")

    # Create the graph
    from rexv2.asynchronous import AsyncGraph

    graph = AsyncGraph(nodes, agent, clock=Clock.SIMULATED, real_time_factor=RealTimeFactor.FAST_AS_POSSIBLE)

    # Set record settings
    world.set_record_settings(params=True, rng=True, inputs=True, state=True, output=True)

    # Get predefined params
    ss = world.init_step_state()
    params = world.init_params()

    # Test graph API
    episodes = []
    gs = graph.init(params={"world": params})
    for j in range(2):
        if j == 0:  # Run API
            for _ in range(10 + j):
                gs = graph.run(gs)
        else:  # Reset/Step API
            gs, _ = graph.reset(gs)
            for _ in range(10 + j):
                gs, _ = graph.step(gs)
        graph.stop()

        # Get records
        r = graph.get_record()
        episodes.append(r)
    record = base.ExperimentRecord(episodes=episodes)

    # Fit the delay distributions
    # from rexv2.gmm_estimator import GMMEstimator
    # gmm = GMMEstimator(record.episodes[0].nodes["sensor"].steps.delay, name="sensor")
    # gmm.fit()
    # dist = gmm.get_dist()
    # anim_step = gmm.animate_training()
    # anim_step.save("gmm_step.mp4")
    # gmm.plot_hist()
    # gmm.plot_loss()
    # gmm.plot_normalized_weights()
    # plt.show()

    # Convert to graph
    graphs_raw = record.to_graph()

    # Add dummy sensor
    sim = pendulum.nodes.Sensor(name="sim", rate=80, color="indigo", order=4, delay=0.5 / 80,
                                delay_dist=base.StaticDist.create(
                                    distrax.Normal(loc=0.5 / 80, scale=0.1 / 80)))
    delay_dist = base.StaticDist.create(distrax.Normal(loc=0.1 / 100, scale=0.05 / 100))
    sim.connect(world, window=1, name="world", delay=1 / 100, delay_dist=delay_dist)
    min_delay, max_delay = jnp.clip(delay_dist.quantile(0.01), 0., None), delay_dist.quantile(0.99)
    delay_dist = base.TrainableDist.create(alpha=0.5, min=min_delay, max=max_delay)
    agent.connect(sim, window=1, name="sim", delay=1 / 100, delay_dist=delay_dist)
    nodes["sim"] = sim

    # Augment the graphs with simulation nodes
    from rexv2 import artificial

    graphs_aug = artificial.augment_graphs(graphs_raw, nodes)

    # Generate the graphs with simulation nodes
    # graphs_gen = artificial.generate_graphs(nodes, ts_max=1.0, num_episodes=2)

    # Compile the graph
    graph = rexv2.graph.Graph(nodes, agent, graphs_aug, supergraph=Supergraph.MCS, progress_bar=True)
    # graph.init = jax.jit(graph.init, static_argnames=("order",))  # Compile the init function
    graph.reset = jax.jit(graph.reset)  # Compile the reset function
    graph.step = jax.jit(graph.step)  # Compile the step function
    graph.run = jax.jit(graph.run)  # Compile the run function

    # Test the graph
    gs = graph.init(jax.random.PRNGKey(0), randomize_eps=True, order=("world",))
    gs, ss = graph.reset(gs)  # Reset the graph state, returns the initial observation (in ss)
    rollout = [gs]
    pbar = tqdm.tqdm(range(graph.max_steps))
    for _ in pbar:  # Simulate for a number of steps
        # Access the last sensor message of the input buffer
        # -1 is the most recent message, -2 the second most recent, etc. up until the window size
        sensor_msg = ss.inputs["sensor"][-1].data  # .data grabs the pytree message object

        # Devise an action, prepare the output message, and update the step_state.
        rng_next, rng_action = jax.random.split(ss.rng, num=2)
        action = jax.random.uniform(rng_action, shape=(1,), minval=-2.0, maxval=2.0)
        output = pendulum.nodes.ActuatorOutput(action=action)  # Define output
        next_ss = ss.replace(rng=rng_next)  # Update the step state

        # Print the current time, sensor reading, and action
        pbar.set_postfix_str(
            f"step: {ss.seq}, ts_start: {ss.ts:.2f}, th: {sensor_msg.th:.2f}, thdot: {sensor_msg.thdot:.2f}, Action: {action[0]:.2f}")

        # Step the graph (i.e., executes the next time step by sending the output message to the actuator node)
        gs, ss = graph.step(gs, step_state=next_ss, output=output)  # Step the graph
        rollout.append(gs)
    pbar.close()

    # However, now we will repeatedly call the run() method, which will call the step() method of the agent node.
    # In our case, the agent node is a random agent, so it will also generate random actions.
    rollout = [gs]
    pbar = tqdm.tqdm(range(graph.max_steps))
    for _ in pbar:
        gs = graph.run(gs)  # Run the graph (incl. the agent's step() method)
        rollout.append(gs)
        # We can access the agent's state directly (this is the state *after* the step() method was called)
        ss = gs.step_state["agent"]
        # Print the current time, sensor reading, and action
        pbar.set_postfix_str(f"step: {ss.seq}, ts_start: {ss.ts:.2f}")
    pbar.close()

    # @title Vectorized rollouts
    # We may also perform many rollouts in parallel by using jax.vmap
    num_rollouts = 500
    rngs = jax.random.split(jax.random.PRNGKey(2), num=num_rollouts)

    # Vectorized graph initialization
    graph_init = functools.partial(graph.init, randomize_eps=True, order=("world",))
    graph_init_jv = jax.jit(jax.vmap(graph_init, in_axes=0))

    # Vectorized graph rollout with .rollout() convenience function
    graph_rollout_jv = jax.jit(jax.vmap(graph.rollout, in_axes=0))

    # Check if we have a GPU
    try:
        gpu = jax.devices("gpu")
    except RuntimeError:
        print("Warning: No GPU found, falling back to CPU. Speedups will be less pronounced.")
        print("Hint: if you are using Google Colab, try to change the runtime to GPU: "
              "Runtime -> Change runtime type -> Hardware accelerator -> GPU.")

    # During the first call, the two functions are compiled
    from supergraph.evaluate import timer

    with timer("Jit-compilation[graph.init]"):
        gs = graph_init_jv(rngs)
    with timer("Jit-compilation[graph.rollout]"):
        rollout = graph_rollout_jv(gs, eps=gs.eps)

    # Subsequent calls are much faster
    # @markdown  Note that the speedups require a GPU. On a CPU, the speedups for the vectorized rollout are less pronounced.
    t = timer(f"Vectorized rollout of {num_rollouts} rollouts", repeat=10)
    with t:
        for _ in range(10):
            gs = graph_init_jv(rngs)
            rollout = graph_rollout_jv(gs, eps=gs.eps)
    fps = (graph.max_steps * num_rollouts * t.repeat) / t.duration
    print(f"\nSimulation speed: {fps:.0f} steps/second (depends on hardware)\n")

    # Visualize rollouts
    num_viz = min(100, num_rollouts)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    axs[0].plot(rollout.ts["agent"][:num_viz, :].T, rollout.inputs["agent"]["sensor"].data.th[:num_viz, :, -1].T)
    axs[0].set_title("Angle")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Angle [rad]")
    axs[1].plot(rollout.ts["agent"][:num_viz, :].T,
                rollout.inputs["agent"]["sensor"].data.thdot[:num_viz, :, -1].T)
    axs[1].set_title("Angular velocity")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Angular velocity [rad/s]")
    axs[2].plot(rollout.ts["actuator"][:num_viz, :].T,
                rollout.inputs["actuator"]["agent"].data.action[:num_viz, :, -1, 0].T)
    axs[2].set_title("Action")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Action")

    # Visualize the graph
    Gs = graph.Gs
    # Gs = [utils.to_networkx_graph(graphs_aug[i], nodes=nodes, validate=True) for i in range(2)]
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    for i, G in enumerate(Gs):
        supergraph.plot_graph(G, max_x=10, ax=axs[i])
        axs[i].set_title(f"Episode {i}")
        axs[i].set_xlabel("Time [s]")
    plt.show()

    exit()
    # Define the phase
    phase = dict(world=distrax.Deterministic(loc=0.),
                 sensor=distrax.TruncatedNormal(loc=0.5 / sensor.rate, scale=0.5 / sensor.rate, low=0., high=1 / sensor.rate),
                 agent=distrax.TruncatedNormal(loc=0.5 / agent.rate, scale=0.5 / agent.rate, low=0., high=1 / agent.rate),
                 actuator=distrax.TruncatedNormal(loc=0.5 / actuator.rate, scale=0.5 / actuator.rate, low=0.,
                                                  high=1 / actuator.rate), )

    # Define the expected computation delay distributions of each node's .step() method
    computation_delays = dict(world=distrax.Deterministic(loc=1 / world.rate),
                              sensor=distrax.TruncatedNormal(loc=0.5 / sensor.rate, scale=0.1 / sensor.rate, low=1e-6,
                                                             high=1e6),
                              agent=distrax.TruncatedNormal(loc=0.5 / agent.rate, scale=0.05 / agent.rate, low=1e-6, high=1e6),
                              actuator=distrax.TruncatedNormal(loc=0.5 / actuator.rate, scale=0.05 / actuator.rate, low=1e-6,
                                                               high=1e6))

    # Define the expected communication delay distributions of sending messages for each connection
    communication_delays = dict()
    for n in [world, sensor, agent, actuator]:
        for c in n.outputs.values():
            communication_delays[(c.output_node.name, c.input_node.name)] = distrax.TruncatedNormal(loc=1 / 100,
                                                                                                    scale=0.1 * 1 / 100,
                                                                                                    low=1e-6,
                                                                                                    high=1e6)

    # Artificially generate graphs
    rng = jax.random.PRNGKey(0)
    t_final = 5.0  # Total time of each episode
    num_episodes = 10  # Number of episodes
    graphs_raw = rexv2.artificial.generate_graphs(nodes, computation_delays, communication_delays, rng=rng,
                                                  t_final=t_final, phase=phase, num_episodes=num_episodes)
    # Visualize the first second of two generated computation graphs
    # Notice the slight differences in the graphs due to random delays and phase shifts
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    Gs = [rexv2.utils.to_networkx_graph(graphs_raw[i], nodes=nodes) for i in range(2)]
    for i, G in enumerate(Gs):
        supergraph.plot_graph(G, max_x=0.3, ax=axs[i])
        axs[i].set_title(f"Episode {i}")
        axs[i].set_xlabel("Time [s]")

    supergraph_method = "MCS"  # Other options "MCS", "topological", or "generational"
    graph = rexv2.graph.Graph(nodes, agent, graphs_raw, supergraph="MCS", progress_bar=True)
    # graph.init = jax.jit(graph.init, static_argnames=("order",))  # Compile the init function
    graph.reset = jax.jit(graph.reset)  # Compile the reset function
    graph.step = jax.jit(graph.step)  # Compile the step function
    graph.run = jax.jit(graph.run)  # Compile the run function

    # Next, we visualize the identified supergraph that will be run every step of the simulation.
    # @markdown Notice how the agent is the supervisor, meaning there can only be one agent vertex.
    # @markdown Moreover, it must be a root of the supergraph.
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    supergraph.plot_graph(graph.S, ax=ax)
    ax.set_title(f"Supergraph `{supergraph_method}`")
    ax.set_xlabel("Topological generation")

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # The raw computation graph, as provided by the compiler (same as above)
    supergraph.plot_graph(Gs[0], max_x=0.3, ax=axs[0])
    axs[0].set_title(f"Episode 0 (raw)")
    axs[0].set_xlabel("Time [s]")

    # The one that is used for actual computation, which takes into account the window sizes of the connections
    supergraph.plot_graph(graph.Gs[0], max_x=0.3, ax=axs[1])
    axs[1].set_title(f"Episode 0 (incl. window)")
    axs[1].set_xlabel("Time [s]")

    # The graph state contains the state of each node, but also what episode graph is currently being executed, and
    # the current time within the episode, that is, what partitions have already been executed.
    # As we will see later in the node definition, the state of a node is divided into params, states, and input buffers.
    rng = jax.random.PRNGKey(1)

    # Initialize the graph state
    # The order can be important, as it defines the order in which the nodes must be initialized.
    # That is, some node initialization procedures may depend on the result of others.
    gs = graph.init(rng, randomize_eps=True, order=("world",))

    # Gym-API like interface, where the user calls reset() and step() methods
    gs, ss = graph.reset(gs)  # Reset the graph state, returns the initial observation (in ss)
    rollout = [gs]
    pbar = tqdm.tqdm(range(graph.max_steps))
    for _ in pbar:  # Simulate for a number of steps
        # Access the last sensor message of the input buffer
        # -1 is the most recent message, -2 the second most recent, etc. up until the window size
        sensor_msg = ss.inputs["sensor"][-1].data  # .data grabs the pytree message object

        # Devise an action, prepare the output message, and update the step_state.
        rng_next, rng_action = jax.random.split(ss.rng, num=2)
        action = jax.random.uniform(rng_action, shape=(1,), minval=-2.0, maxval=2.0)
        output = pendulum.nodes.ActuatorOutput(action=action)  # Define output
        next_ss = ss.replace(rng=rng_next)  # Update the step state

        # Print the current time, sensor reading, and action
        pbar.set_postfix_str(
            f"step: {ss.seq}, ts_start: {ss.ts:.2f}, th: {sensor_msg.th:.2f}, thdot: {sensor_msg.thdot:.2f}, Action: {action[0]:.2f}")

        # Step the graph (i.e., executes the next time step by sending the output message to the actuator node)
        gs, ss = graph.step(gs, step_state=next_ss, output=output)  # Step the graph
        rollout.append(gs)
    pbar.close()

    # The previous example follows the Gym-API like interface, but completely ignores
    # the step() method of the supervisor node (agent).
    # In some cases, it may be convenient to run the graph including the step() method
    # of the supervisor node (agent).
    # For this, we expose an alternative interface using the .run() method.

    # Again, we initialize the graph state
    gs = graph.init(rng, randomize_eps=True, order=("world",))

    # However, now we will repeatedly call the run() method, which will call the step() method of the agent node.
    # In our case, the agent node is a random agent, so it will also generate random actions.
    rollout = [gs]
    pbar = tqdm.tqdm(range(graph.max_steps))
    for _ in pbar:
        gs = graph.run(gs)  # Run the graph (incl. the agent's step() method)
        rollout.append(gs)
        # We can access the agent's state directly (this is the state *after* the step() method was called)
        ss = gs.nodes["agent"]
        # Print the current time, sensor reading, and action
        pbar.set_postfix_str(f"step: {ss.seq}, ts_start: {ss.ts:.2f}")
    pbar.close()

    # @title Vectorized rollouts
    # We may also perform many rollouts in parallel by using jax.vmap
    num_rollouts = 500
    rngs = jax.random.split(jax.random.PRNGKey(2), num=num_rollouts)

    # Vectorized graph initialization
    graph_init = functools.partial(graph.init, randomize_eps=True, order=("world",))
    graph_init_jv = jax.jit(jax.vmap(graph_init, in_axes=0))

    # Vectorized graph rollout with .rollout() convenience function
    graph_rollout_jv = jax.jit(jax.vmap(graph.rollout, in_axes=0))

    # Check if we have a GPU
    try:
        gpu = jax.devices("gpu")
    except RuntimeError:
        print("Warning: No GPU found, falling back to CPU. Speedups will be less pronounced.")
        print("Hint: if you are using Google Colab, try to change the runtime to GPU: "
              "Runtime -> Change runtime type -> Hardware accelerator -> GPU.")

    # During the first call, the two functions are compiled
    from supergraph.evaluate import timer

    with timer("Jit-compilation[graph.init]"):
        gs = graph_init_jv(rngs)
    with timer("Jit-compilation[graph.rollout]"):
        rollout = graph_rollout_jv(gs, eps=gs.eps)

    # Subsequent calls are much faster
    # @markdown  Note that the speedups require a GPU. On a CPU, the speedups for the vectorized rollout are less pronounced.
    t = timer(f"Vectorized rollout of {num_rollouts} rollouts", repeat=10)
    with t:
        for _ in range(10):
            gs = graph_init_jv(rngs)
            rollout = graph_rollout_jv(gs, eps=gs.eps)
    fps = (graph.max_steps * num_rollouts * t.repeat) / t.duration
    print(f"\nSimulation speed: {fps:.0f} steps/second (depends on hardware)\n")

    # Visualize rollouts
    num_viz = min(100, num_rollouts)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    axs[0].plot(rollout.nodes["agent"].ts[:num_viz, :].T, rollout.nodes["agent"].inputs["sensor"].data.th[:num_viz, :, -1].T)
    axs[0].set_title("Angle")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Angle [rad]")
    axs[1].plot(rollout.nodes["agent"].ts[:num_viz, :].T,
                rollout.nodes["agent"].inputs["sensor"].data.thdot[:num_viz, :, -1].T)
    axs[1].set_title("Angular velocity")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Angular velocity [rad/s]")
    axs[2].plot(rollout.nodes["actuator"].ts[:num_viz, :].T,
                rollout.nodes["actuator"].inputs["agent"].data.action[:num_viz, :, -1, 0].T)
    axs[2].set_title("Action")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Action")
