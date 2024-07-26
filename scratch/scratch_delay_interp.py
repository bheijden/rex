import jax
import numpy as onp
import jax.numpy as jnp

from rexv2 import base


zoh = base.TrainableDist(alpha=0.5, min=0.0, max=0.4, interp="zoh")
lin = base.TrainableDist(alpha=0.5, min=0.0, max=0.4, interp="linear")

# Prepare input
seq = onp.array([-1, -1, 0, 1, 2, 3])
ts_sent = onp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3])
# ts_recv = onp.array([0.0, 0.0, 0.0, 0.11, 0.21, 0.31])
ts_recv = ts_sent + lin.mean()
data = onp.array([-100, -100, 0.0, 1.0, 2.0, 3.0])
i_zoh = base.InputState.from_outputs(seq=seq, ts_sent=ts_sent, ts_recv=ts_recv, outputs=data, delay_dist=zoh, is_data=True)
i_lin = i_zoh.replace(delay_dist=lin)

# Test
rate_out = 10
ts_start = 0.0
print("window:", i_zoh.delay_dist.window(rate_out))
interp_lin = i_lin.delay_dist.apply_delay(rate_out, i_lin, ts_start + lin.mean())
print("interp_lin:", interp_lin)
# interp_lin = jax.jit(i_lin.delay_dist.apply_delay, static_argnums=(0,))(rate_out, i_lin, ts_start + lin.mean())
interp_zoh = i_zoh.delay_dist.apply_delay(rate_out, i_zoh, ts_start + zoh.mean())
print("interp_zoh:", interp_zoh)






