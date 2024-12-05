import itertools

import jax
import jax.numpy as jnp


if __name__ == "__main__":
    # Check if we have a GPU
    try:
        gpu = jax.devices("gpu")
        gpu = gpu[0] if len(gpu) > 0 else None
        print("GPU found!")
    except RuntimeError:
        print("Warning: No GPU found, falling back to CPU. Speedups will be less pronounced.")
        print(
            "Hint: if you are using Google Colab, try to change the runtime to GPU: "
            "Runtime -> Change runtime type -> Hardware accelerator -> GPU."
        )
        gpu = None

    # Check the number of available CPU cores
    print(f"CPU cores available: {len(jax.devices('cpu'))}")
    cpus = itertools.cycle(jax.devices("cpu"))

    # Create a Brax world
    from rex.pendulum.brax import BraxWorld

    world = BraxWorld(name="world", rate=50, color="grape", order=0)  # Brax world that simulates the pendulum

    # Repeat a dummy state
    state = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None], 10, axis=0), world.init_state())

    from rex.pendulum.render import render

    render(state, dt=float(1 / world.rate))
