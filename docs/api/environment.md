# Environment and Wrappers

::: rex.rl.BaseEnv

---

::: rex.rl.BaseWrapper
        options:
            members:
                - __init__
                - __getattr__

---

::: rex.rl.AutoResetWrapper
        options:
            members:
                - __init__
                - reset
                - step

---

::: rex.rl.LogWrapper
        options:
            members:
                - __init__
                - reset
                - step

::: rex.rl.LogState
        options:
            members:
                - 

---

::: rex.rl.SquashActionWrapper
        options:
            members:
                - __init__
                - reset
                - step
                - action_space

::: rex.rl.SquashState
        options:
            members:
                - scale
                - unsquash
                - action_space

---

::: rex.rl.ClipActionWrapper
        options:
            members:
                - step

---

::: rex.rl.VecEnvWrapper
        options:
            members:
                - __init__
---

::: rex.rl.NormalizeVecObservationWrapper
        options:
            members:
                - __init__
                - reset
                - step

::: rex.rl.NormalizeVecReward
        options:
            members:
                - __init__
                - reset
                - step

::: rex.rl.NormalizeVec
        options:
            members:
                - normalize
                - denormalize
