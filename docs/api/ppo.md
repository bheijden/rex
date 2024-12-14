# Proximal Policy Optimization

::: rex.ppo.train

---

::: rex.ppo.Config
        options:
            members:
                - EVAL_METRICS_JAX_CB
                - EVAL_METRICS_HOST_CB

---

::: rex.ppo.PPOResult
        options:
            members:
                - obs_scaling
                - act_scaling
                - rwd_Scaling
                - policy

---

::: rex.ppo.Policy
        options:
            members:
                - apply_actor
                - get_action

---

::: rex.ppo.RunnerState
        options:
            members:
                - 