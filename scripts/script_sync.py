from math import floor, ceil
from termcolor import cprint, colored


def expected_inputs(N_out: int, rate_in: float, rate_out: float, delay: float, skip: bool):
    # In case of skip=True and the output rate is an exact multiple of the input rate, we need to add a slight offset
    eps = 1e-9

    # Constants
    offset = int(skip) * (int((rate_out - eps) / rate_in) if rate_out > rate_in else -1)

    # Alternative numerically unstable implementation
    # dt_out, dt_in = 1 / rate_out, 1 / rate_in
    # t_prev = dt_out * (N_out - 1 + offset) - delay
    # t = dt_out * (N_out + offset) - delay
    # N_in_prev = int(t_prev / dt_in)
    # N_in = int(t / dt_in)
    # delta = N_in - N_in_prev
    # j = (delta - 1 + int(not skip))
    # T = dt_out * N_out - delay - j * dt_in
    # correction = ceil(-T / dt_in)

    # Numerically stable implementation
    N_in_prev = int((rate_in * (N_out + offset - 1) - rate_out * rate_in * delay) // rate_out)
    N_in = int((rate_in * (N_out + offset) - rate_out * rate_in * delay) // rate_out)

    # Alternative (iterative) delay correction
    # delta = N_in - N_in_prev
    # for i in range(int(not skip), delta + int(not skip)):
    #     # Alternative numerically unstable implementation
    #     # t_in_delayed = dt_out * N_out - delay - i * dt_in
    #     # Numerically stable implementation
    #     t_in_delayed = (N_out * rate_in - delay * rate_out * rate_in - i * rate_out) / rate_in
    #     if t_in_delayed < 0:
    #         delta -= 1

    # Alternative delay correction
    delta = N_in - N_in_prev
    j = (delta - 1 + int(not skip))
    # Numerically stable implementation
    T = (rate_in * N_out - rate_out * rate_in * delay - rate_out * j) / (rate_out * rate_in)
    correction = -floor(T * rate_in)
    corrected = delta - min(delta, max(0, correction))  # limits as follows: 0 < correction < delta

    # Overwrite t=0 dependencies, because there are none when skipping.
    num_est = corrected if N_out > 0 else int(not skip)
    return num_est


# Definition: engine rate >= node rate

node = ("node", 0.0, 0.0, 1, 1.25, False, [1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1])
node_v2 = ("node_v2", 0.0, 0.0, 1, 2.5, False, [1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3])
engine = ("engine", 0.0, 0.0, 1.25, 1, False, [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1])
engine_v2 = ("engine_v2", 0.0, 0.0, 2.50, 1, False, [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1])
equal = ("equal", 0.0, 0.0, 1, 1, False, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
double_in = ("double_in", 0.0, 0.0, 1, 2, False, [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
double_out = ("double_out", 0.0, 0.0, 2, 1, False, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
triple_out = ("triple_out", 0.0, 0.0, 3, 1, False, [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0])

skip_node = ("skip_node", 0.0, 0.0, 1, 1.25, True, [0, 2, 1, 1, 1, 2, 1, 1, 1, 2]) # node skips first engine tick
skip_node_v2 = ("skip_node_v2", 0.0, 0.0, 1, 2.5, True, [0, 3, 2, 3, 2, 3, 2, 3, 2, 3])
skip_engine = ("skip_engine", 0.0, 0.0, 1.25, 1, True, [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1])
skip_engine_v2 = ("skip_engine_v2", 0.0, 0.0, 2.50, 1, True, [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1])
skip_equal = ("skip_equal", 0.0, 0.0, 1, 1, True, [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

skip_double_in = ("skip_double_in", 0.0, 0.0, 1, 2, True, [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
skip_double_out = ("skip_double_out", 0.0, 0.0, 2, 1, True, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
skip_triple_out = ("skip_triple_out", 0.0, 0.0, 3, 1, True, [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1])

delay_engine_v2 = ("delay_engine_v2", 0.6, 0.2, 2.50, 1, False, [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
delay_node_v2_multiple = ("delay_node_v2_multiple", 0.6, 0.2, 1, 2.50, False, [1, 1, 3, 2, 3, 2, 3, 2])
delay_node_v2 = ("delay_node_v2", 1.6, 0.2, 1, 2.50, False, [1, 0, 1, 3, 2, 3, 2, 3])
delay_large_equal = ("delay_large_equal", 10.2, 0.2, 1.5, 1.5, False, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

delay_skip_node_multiple = ("delay_skip_node_multiple", 0.2, 0.6, 1, 2.5, True, [0, 2, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3])
delay_skip_node = ("delay_skip_node", 0.2, 1.6, 1, 2.5, True, [0, 0, 2, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3])
delay_skip_engine = ("delay_skip_engine", 0.2, 1.6, 2.5, 1, True, [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1])
delay_skip_engine_multiple = ("delay_skip_engine_multiple", 0.2, 0.6, 2.5, 1, True, [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1])

# ALL
runs = [
        skip_node, skip_node_v2, skip_engine, skip_engine_v2, skip_equal, skip_double_in, skip_double_out, skip_triple_out,
        node, node_v2, engine, engine_v2, equal, double_in, double_out, triple_out,
        delay_engine_v2, delay_node_v2_multiple, delay_node_v2, delay_large_equal,
        delay_skip_node_multiple, delay_skip_node, delay_skip_engine, delay_skip_engine_multiple,
]

# NORMAL
# runs = [skip_node]
# runs = [skip_node_v2]
# runs = [skip_engine]
# runs = [skip_engine_v2]
# runs = [skip_equal]
# runs = [skip_double_in]
# runs = [skip_double_out]
# runs = [skip_triple_out]
# runs = [node]
# runs = [node_v2]
# runs = [engine]
# runs = [engine_v2]
# runs = [equal]
# runs = [double_in]
# runs = [double_out]
# runs = [triple_out]

# runs = [delay_engine_v2]
# runs = [delay_node_v2_multiple]
# runs = [delay_node_v2]
# runs = [delay_skip_node_multiple]
# runs = [delay_skip_node]
# runs = [delay_skip_engine]
# runs = [delay_skip_engine_multiple]
# runs = [delay_large_equal]


def add(N, t_idx, t_in, s_dict, s_list, f):
    f += 1
    s_list[N] += 1
    s_dict[t_idx]["num"] += 1
    s_dict[t_idx]["t_in"].append(t_in)


def verify(name, rate, s_dict, skip: bool, raise_exception: bool = False):
    for N, (t_idx, grouped) in enumerate(s_dict.items()):
        t_idx_min = round(t_idx-1/rate, 6)
        num, t_in = grouped["num"], grouped["t_in"]
        text_t = []
        flag = False
        for t in t_in:
            color = "green"
            if skip:
                if t >= t_idx or (t < t_idx_min and N != 0):
                    color = "red"
                    flag = True
            else:
                if t > t_idx or (t <= t_idx_min and N != 0):
                    color = "red"
                    flag = True
            text_t.append(colored(str(t), color))
        op_high = "<" if skip else "<="
        op_low = "<=" if skip else "<"
        if flag and raise_exception:
            print(f"{name} | skip={skip} | N={str(N).ljust(2)} | {str(t_idx_min).ljust(5)}{op_low} [{', '.join(text_t)}] {op_high}{str(t_idx).ljust(5)}")
            raise Exception()
        else:
            print(f"{name} | skip={skip} | N={str(N).ljust(2)} | {str(t_idx_min).ljust(5)}{op_low} [{', '.join(text_t)}] {op_high}{str(t_idx).ljust(5)}")


for run in runs:
    name, phase_node, phase_in, rate_node, rate_in, skip, seq = run
    t_in = [round(i/rate_in + phase_in, 6) for i in range(0, 100)]
    t_node = [round(i/rate_node + phase_node, 6) for i in range(0, 100) if round(i/rate_node, 6) < 15]
    seq_list = len(t_node) * [0]
    seq_dict = {t: dict(num=0, t_in=[]) for t in t_node}
    for t in t_in:
        flag = 0
        for N_node, (t_low, t_high) in enumerate(zip(t_node[0:-1], t_node[1:])):
            if N_node == 0:
                if t <= t_low and not skip:
                    add(N_node, t_low, t, seq_dict, seq_list, flag)
                elif t < t_low and skip:
                    add(N_node, t_low, t, seq_dict, seq_list, flag)
            if t_low < t <= t_high and not skip:
                add(N_node+1, t_high, t, seq_dict, seq_list, flag)
            elif t_low <= t < t_high and skip:
                add(N_node+1, t_high, t, seq_dict, seq_list, flag)
        assert flag < 2

    # Verify generated sequences
    verify(name, rate_node, seq_dict, skip, raise_exception=True)

    # old sequences --> they are incorrect with delays (based on implementation where delays were implemented differently).
    assert seq == seq_list[:len(seq)] or phase_in != phase_node, name

    seq = seq_list

    for N_node, num in enumerate(seq):
        # In case of skip=True and the output rate is an exact multiple of the input rate, we need to add a slight offset
        # eps = 1e-9

        # Alternative numerically unstable implementation
        dt_node, dt_in = 1 / rate_node, 1 / rate_in
        t_high = dt_node * N_node + phase_node
        t_low = dt_node * (N_node - 1) + phase_node
        t_high = round(t_high, 6)
        t_low = round(t_low, 6)

        # Determine starting t_in
        # todo: find numerically stable (and fast) implementation.
        i = int((t_low - phase_in) // dt_in) if N_node > 0 else 0

        text_t = []
        t = round(i / rate_in + phase_in, 6)
        while not t > t_high:
            flag = 0
            if not t < phase_in:
                if N_node == 0:
                    if t <= t_low and not skip:
                        text_t.append(str(t))
                        flag += 1
                    elif t < t_low and skip:
                        text_t.append(str(t))
                        flag += 1
                if t_low < t <= t_high and not skip:
                    text_t.append(str(t))
                    flag += 1
                elif t_low <= t < t_high and skip:
                    text_t.append(str(t))
                    flag += 1
            assert flag < 2
            i += 1
            t = round(i / rate_in + phase_in, 6)

        num_est = len(text_t)
        op_high = "<" if skip else "<="
        op_low = "<=" if skip else "<"
        color = "green" if num_est == num else "red"
        cprint(f"{name} | skip={skip} | N={str(N_node).ljust(2)} | {str(t_low).ljust(5)}{op_low} [{', '.join(text_t)}] {op_high}{str(t_high).ljust(5)}", color)
        # cprint(f"{name} | N_node={str(N_node).ljust(2)} | t={str(N_node / rate_node).ljust(5)} | num={num}/{num_est}", color)

# if skip:
#     # Phase difference (phase_in <= phase_node if skip=False else phase_in >= phase_node)
#     phase = round(phase_node - phase_in, 6)
#
#     # Alternative numerically unstable implementation
#     dt_node, dt_in = 1 / rate_node, 1 / rate_in
#     t_low = dt_node * (N_node - 1) + phase
#     t_high = dt_node * N_node + phase
#
#     num_est = 0
#     t_in = round(max(-dt_in, dt_in * int(t_low // dt_in)), 6)
#     while not t_in >= t_high:
#         if t_in >= 0 and t_in >= t_low and t_in < t_high:
#             num_est += 1
#         t_in = round(t_in + dt_in, 6)
# else:
#     # Phase difference (phase_in <= phase_node if skip=False else phase_in >= phase_node)
#     phase = round(phase_node - phase_in, 6)
#
#     # Alternative numerically unstable implementation
#     dt_node, dt_in = 1 / rate_node, 1 / rate_in
#     t_high = dt_node * N_node + phase
#     t_low = dt_node * (N_node - 1)
#     if N_node > 0:  # Required for initial tick
#         t_low += phase
#
#     t_high = round(t_high, 6)
#     t_low = round(t_low, 6)
#
#     num_est = 0
#     t_in = round(max(-dt_in, dt_in * int(t_low // dt_in)), 6)
#     while not t_in > t_high:
#         if t_in >= 0 and t_in > t_low and t_in <= t_high:
#             num_est += 1
#         t_in = round(t_in + dt_in, 6)