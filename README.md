<h1 align="center">Rex: Robotic Environments with jaX</h1>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="license">
  </a>
  <a href="https://pep8.org/">
    <img src="https://img.shields.io/badge/code%20style-pep8-000000.svg" alt="PEP8">
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="codestyle">
  </a>
</p>

Rex is a [JAX](https://github.com/google/jax)-powered framework for sim-to-real robotics.

Key features:

- **Graph-based design**: Model asynchronous systems with nodes for sensing, actuation, and computation.
- **Latency-aware modeling**: Simulate delay effects for hardware, computation, and communication channels.
- **Real-time and parallelized runtimes**: Run real-world experiments or accelerated parallelized simulations.
- **Seamless integration with JAX**: Utilize JAX's autodiff, JIT compilation, and GPU/TPU acceleration.
- **System identification tools**: Estimate dynamics and delays directly from real-world data.
- **Modular and extensible**: Compatible with various simulation engines (e.g., [Brax](https://github.com/google/brax), [MuJoCo](https://mujoco.readthedocs.io/en/stable/mjx.html)).
- **Unified sim2real pipeline**: Train delay-aware policies in simulation and deploy them on real-world systems.

## Sim-to-Real Workflow

1. **Interface Real Systems**: Define nodes for sensors, actuators, and computation to represent real-world systems.
2. **Build Simulation**: Swap real-world nodes with simulated ones (e.g., physics engines, motor dynamics).
3. **System Identification**: Estimate system dynamics and delays from real-world data.
4. **Policy Training**: Train delay-aware policies in simulation, accounting for realistic dynamics and delays.
5. **Evaluation**: Evaluate trained policies on the real-world system, and iterate on the design.

## Installation

```bash
pip install rex-lib
```

Requires Python 3.9+ and JAX 0.4.30+.

## Documentation
Available at [https://bheijden.github.io/rex/](https://bheijden.github.io/rex/).

## Quick example

Here's a simple example of a pendulum system. 
The real-world system is defined with nodes interfacing hardware for sensing, actuation:
```python
from rex.asynchronous import AsyncGraph
from rex.examples.pendulum import Actuator, Agent, Sensor

sensor = Sensor(rate=50)        # 50 Hz sampling rate
agent = Agent(rate=30)          # 30 Hz policy execution rate
actuator = Actuator(rate=50)    # 50 Hz control rate
nodes = dict(sensor=sensor, agent=agent, actuator=actuator)

agent.connect(sensor)       # Agent receives sensor data
actuator.connect(agent)     # Actuator receives agent commands
graph = AsyncGraph(nodes, agent) # Graph for real-world execution

graph_state = graph.init()  # Initial states of all nodes
graph.warmup(graph_state)   # Jit-compiles the graph (only once).
for _ in range(100):        # Run the graph for 100 steps
    graph_state = graph.run(graph_state) # Run for one step
graph.stop()                # Stop asynchronous nodes
data = graph.get_record()   # Get recorded data from the graph
```
In simulation, we replace the hardware-interfacing nodes with simulated ones, add delay models, and add a physics simulation node:
```python
from distrax import Normal
from rex.constants import Clock, RealTimeFactor
from rex.asynchronous import AsyncGraph
from rex.examples.pendulum import SimActuator, Agent, SimSensor, BraxWorld

sensor = SimSensor(rate=50, delay_dist=Normal(0.01, 0.001))     # Process delay
agent = Agent(rate=30, delay_dist=Normal(0.02, 0.005))          # Computational delay
actuator = SimActuator(rate=50, delay_dist=Normal(0.01, 0.001)) # Process delay
world = BraxWorld(rate=100)  # 100 Hz physics simulation
nodes = dict(sensor=sensor, agent=agent, actuator=actuator, world=world)

sensor.connect(world, delay_dist=Normal(0.001, 0.001)) # Sensor delay
agent.connect(sensor, delay_dist=Normal(0.001, 0.001)) # Communication delay
actuator.connect(agent, delay_dist=Normal(0.001, 0.001)) # Communication delay
world.connect(actuator, delay_dist=Normal(0.001, 0.001), # Actuator delay
              skip=True) # Breaks algebraic loop in the graph
graph = AsyncGraph(nodes, agent,
                   clock=Clock.SIMULATED, # Simulates based on delay_dist
                   real_time_factor=RealTimeFactor.FAST_AS_POSSIBLE)

graph_state = graph.init()  # Initial states of all nodes
graph.warmup(graph_state)   # Jit-compiles the graph
for _ in range(100):        # Run the graph for 100 steps
    graph_state = graph.run(graph_state) # Run for one step
graph.stop()                # Stop asynchronous nodes
data = graph.get_record()   # Get recorded data from the graph
```
Nodes are defined using JAX's [PyTrees](https://jax.readthedocs.io/en/latest/pytrees.html):
```python
from rex.node import BaseNode

class Agent(BaseNode):
    def init_params(self, rng=None, graph_state=None):
        return SomePyTree(a=..., b=...)

    def init_state(self, rng=None, graph_state=None):
        return SomePyTree(x1=..., x2=...)

    def init_output(self, rng=None, graph_state=None):
        return SomePyTree(y1=..., y2=...)
    
    # Jit-compiled via graph.warmup for faster execution
    def step(self, step_state): # Called at Node's rate
        ss = step_state  # Shorten name
        # Read params, and current state
        params, state = ss.params, ss.state
        # Current episode, sequence, timestamp
        eps, seq, ts = ss.eps, ss.seq, ss.ts
        # Grab the data, and I/O timestamps
        cam = ss.inputs["sensor"] # Received messages 
        cam.data, cam.ts_send, cam.ts_recv
        ... # Some computation for new_state, output
        new_state = SomePyTree(x1=..., x2=...)
        output = SomePyTree(y1=..., y2=...)
        # Update step_state for next step call
        new_ss = ss.replace(state=new_state)
        return new_ss, output # Sends output
```

## Next steps
If this quick start has got you interested, then have a look at the [sim2real.ipynb](https://github.com/bheijden/rex/blob/master/examples/sim2real.ipynb) notebook for an example of a sim-to-real workflow using Rex.

## Citation

If you are using rex for your scientific publications, please cite:

```bibtex
@article{heijden2024rex,
  title={{REX: GPU-Accelerated Sim2Real Framework with Delay and Dynamics Estimation}},
  author={van der Heijden, Bas and Kober, Jens and Babuska, Robert and Ferranti, Laura},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2025}
}
```