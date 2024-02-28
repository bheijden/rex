import jax.numpy as jnp
import jax
from flax import struct
from rex.base import InputState, Delay


@struct.dataclass
class SensorOutput:
    out: jax.Array

    @classmethod
    def init(cls, i):
        return cls(out=jnp.array(i))


def test_delay_sysid():
    # Define delay
    rate_out = 10
    window = 3
    delay_sysid = Delay(alpha=0., min=0.1, max=0.35)

    # Define input_state
    cum_window = window + delay_sysid.delay_window(rate_out)
    seqs = jnp.arange(0, cum_window)
    ts_sent = seqs * 1/rate_out
    ts_recv = ts_sent + delay_sysid.min
    outputs = [SensorOutput.init(i) for i in range(cum_window)]
    input_state = InputState.from_outputs(seqs, ts_sent, ts_recv, outputs, delay_sysid=delay_sysid, rate=rate_out)

    # Define test_cases ts_step
    min_ts_step = ts_recv[-1]
    max_ts_step = ts_recv[-1] + 20

    # Define test_cases alpha
    min_alpha = 0.0
    max_alpha = 1.0

    test_cases = [{"alpha": min_alpha, "ts_step": min_ts_step},
                  {"alpha": max_alpha, "ts_step": min_ts_step},
                  {"alpha": max_alpha, "ts_step": max_ts_step},
                  {"alpha": min_alpha, "ts_step": max_ts_step},
                  ]
    for case in test_cases:
        case_delay_sysid = delay_sysid.replace(alpha=case["alpha"])
        _input_state = input_state.replace(delay_sysid=case_delay_sysid)
        case_input_state = _input_state.apply_delay(case["ts_step"])
        assert jnp.all(case_input_state.ts_recv <= case["ts_step"]), "ts_recv should be less than or equal to ts_sent"

        # Check if the next ts_recv that was excluded is greater than ts_step
        max_seq = jnp.max(case_input_state.seq)
        idx_max_seq = jnp.where(_input_state.seq == max_seq)[0][0]
        if idx_max_seq < len(_input_state.seq)-1:
            idx_next = idx_max_seq + 1
            ts_recv_next = _input_state.ts_sent[idx_next] + case_delay_sysid.value
            assert ts_recv_next > case["ts_step"], "ts_recv should be greater than ts_step"






