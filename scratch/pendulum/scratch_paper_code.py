# Simulation nodes & connections
cam = SimCam(rate=60, delay=Gauss(0.05, 0.01))
agent = Agent(rate=30, delay=Gauss(0.02, 0.01))
motor = SimMotor(rate=50, delay=Gauss(0.04, 0.01))
brax = Brax(rate=100, delay=Deterministic(1 / 100))
nodes = [brax, cam, agent, motor]
brax.connect(motor, delay=Gauss(0.01, 0.01))
cam.connect(brax, delay=Gauss(0.01, 0.01))
agent.connect(cam, delay=Gauss(0.01, 0.01))
motor.connect(agent, delay=Gauss(0.01, 0.01),
              window=2, block=True)
# Runtime: SIMULATED (no throttling)
graph = Graph(agent, nodes, Clock.SIMULATED,
              RealTimeFactor.FAST_AS_POSSIBLE)
graph.warmup(devices=...)  # JIT compilation
gs = graph.init()  # Graph state
for i in range(100):  # Simulates 100 steps
    gs = graph.run(gs)
graph.stop()  # Halts all nodes
# Simulated data flow to computation graph
cg = graph.get_record().to_graph().augment(nodes)
# Runtime: COMPILED (1000 parallel rollouts)
graph = CompiledGraph(agent, nodes, cg)
rngs = jax.split(jax.random.PRNGKey(0), num=1000)
gss = jax.vmap(graph.init)(rngs)  # 1000 states
rollout = jax.vmap(graph.rollout)(gss, rngs)  # run
