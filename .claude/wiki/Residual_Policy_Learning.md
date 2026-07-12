---
type: algorithm_reference
tags: [algorithm, reinforcement-learning, residual-policy, prior-controller, pytorch, concept]
source: https://arxiv.org/abs/1812.06298
date_ingested: 2026-07-09
---

# Residual_Policy_Learning

A method for improving an existing imperfect (possibly non-differentiable) base controller by learning an additive residual correction with model-free RL, rather than training a policy from scratch.

**Reference:** Silver, Allen, Tenenbaum, Kaelbling. "Residual Policy Learning." arXiv:1812.06298, 2018.

## Formulation

Given a fixed base policy `π_0(s)` (hand-designed, MPC, or any prior controller), learn a residual policy `π_θ(s)` using standard RL:

```
a_t  =  π_0(s_t)  +  π_θ(s_t)
```

`π_θ` is a standard neural actor (MLP). The base policy is **frozen** and need not be differentiable. All gradient updates go to `π_θ` via a standard policy-gradient objective (PPO, SAC, etc.). Action clipping is applied to `a_t` after addition.

Optional scaling (ariel extension — not in original paper):

```
a_t  =  π_0(s_t)  +  α · π_θ(s_t)          # α ∈ (0, 1]
```

Splitting `α` per-task (e.g. hover 0.10, trajectory 0.40) lets the prior dominate on tasks where it is near-optimal and gives the residual more authority on tasks that require departing from the prior behaviour (banking, lateral racing).

## Parameters

| Name | Default (paper) | Ariel value | Role |
|------|----------------|-------------|------|
| `α` (residual scale) | 1.0 (implicit) | 0.10 hover / 0.40 racing | Weight on residual action before addition |
| π_θ architecture | MLP (shallow) | MLP [256,256] SiLU | Actor hidden layers |
| Base RL algorithm | PPO / TRPO | PPO (SB3) | Policy-gradient algorithm training π_θ |
| Action clipping | [-1, 1] | [-1, 1] | Applied to `a_t` sum, not residual alone |

## Implementation Notes

**Why this works:**
- The base policy handles most of the control signal; RL exploration only needs to search for small corrections.
- Drastically reduces the effective exploration burden vs. from-scratch training.
- Works even with sparse rewards because the base policy provides a reasonable return floor from step 0.

**Prior-fighting failure mode:**
- If `π_0` actively resists the motion the residual is trying to produce (e.g. attitude-levelling gains fighting deliberate banking), the residual may learn to saturate against the prior.
- **Mitigation (ariel):** `TASK_PRIOR_GAIN_SCALE` in `ResidualDroneEnv` scales down `k_tilt` to 0.3 and rate damping to 0.5 for trajectory tasks so the hover prior's stability terms no longer fight deliberate banking. Only altitude-tracking gains (`k_alt_p`, `k_alt_d`) are kept at full strength.

**Residual magnitude monitoring:**
- Log mean `|π_θ(s_t)|` during training. If consistently near zero, the residual has collapsed (the prior is sufficient or `α` is too small). If consistently near 1, the prior contributes nothing and the system is training from scratch.
- For hover task in ariel, `α = 0.10` with a near-perfect prior is expected to produce small but non-zero residual magnitudes.

**Action space:**
- The original paper uses continuous action spaces. The addition is element-wise.
- In ariel, actions are per-motor throttle commands in `[-1, 1]^N` (N motors). The prior outputs motor commands and the residual adds corrections.

**Critic independence:**
- The critic (value function) can receive additional inputs not given to the actor, including information about `π_0`'s quality (e.g. CMA hover score, `cmaes_params`). This tightens advantage estimates without giving the actor the ability to "undo" the prior selectively.

## When to Use

- **Use RPL when:** A reasonable base controller exists (hand-tuned, MPC, analytical, CMA-tuned) and RL from scratch is too sample-inefficient or tends to fail on sparse rewards.
- **Especially suited for:** Morphology-conditional control where a different analytical prior can be computed cheaply per body (e.g. CMA-ES hover tuning in ~60s per morphology).
- **Less useful when:** The base policy is actively wrong (reward at step 0 is negative); better to weaken the prior's gains first. Or when the action space dimension is very large relative to the prior's expressiveness.

## In Ariel

**Implementation:** `examples/spear/library/`

| File | Role |
|------|------|
| `prior_controller.py` | `HoverPrior` — PyTorch implementation of the analytical PD + yaw-damping controller. Single source of truth for mixer sign convention. Takes `(state, cmaes_params)` → motor commands. |
| `envs/residual_drone_env.py` | `ResidualDroneEnv` — wraps `TorchDroneGateEnv`. Applies `action_total = clamp(prior + α·residual)` internally so PPO sees only the residual action space. Per-task `TASK_ALPHA` and `TASK_PRIOR_GAIN_SCALE` class attributes. |
| `37_train_residual_mtrl.py` | Stage-3 MTRL-PPO trainer. Obs = `(gate_drone_obs[26], task_oh[5], morph_features[22]) = 53d`. Per-task reward normalisation via `_PerTaskRewardNormalizer`. |
| `35c_hover_cmaes_minimal.py` | CMA-ES tuner that produces the 11-d `cmaes_params` vector stored in the hex library for each morphology. |
| `36_build_hover_library.py` | Stage-1 script: samples 100 hex morphs, runs 35c per morph, saves `__data__/hex_library/v1/library.npz`. |

**Prior parameterisation in ariel:**
The base policy is not a learned NN — it is a closed-form PD + yaw-damping controller:

```
cmaes_params = [trim_0, ..., trim_5,    # per-motor trim offsets (6)
                k_alt_p, k_alt_d,       # altitude PD gains (2)
                k_tilt,                 # attitude levelling gain (1)
                k_rate,                 # rate damping gain (1)
                k_yaw_rate]             # yaw rate damping gain (1)
               # total: 11 dimensions
```

These are fit per-morphology by 35c in ~60s (CMA-ES budget=400, λ=128). They are stored in `library.npz` and looked up at env construction — the residual PPO never sees them.

**Observation space extension vs. original paper:**
The original paper conditions `π_θ` on `s_t` only. Ariel extends this to `(s_t, task_one_hot[5], morph_features[22])` so a single residual policy covers 5 tasks across 100 morphologies. The `morph_features` are a hand-crafted 22-d permutation-invariant descriptor (arm-length stats, TWR, azimuth-gap stats, etc.) from `morphology_features.py`.

## From: Johannink et al. 2019 — Residual RL for Robot Control (arXiv:1812.03201, ICRA 2019)

**Authors:** Johannink, Bahl, Nair, Luo, Kumar, Loskyll, Aparicio Ojea, Solowjow, Levine.

A concurrent, independently developed formulation of the same residual RL idea, validated on a **real Kuka IIWA arm** performing precision block insertion (gear-on-shaft). Key technical differences from Silver et al.:

### Formulation (Johannink)

```
τ_t  =  τ_fb(s_t)  +  π_θ(s_t)          # torque-space superposition
```

`τ_fb` is a Cartesian impedance / position feedback controller operating in torque space. The residual `π_θ` outputs bounded torque corrections. Final torques are **clipped** to protect physical hardware.

### RL Algorithm: SAC (not PPO)

Johannink et al. use **Soft Actor-Critic (SAC)** with automatic entropy tuning — off-policy and significantly more sample-efficient than PPO. This matters for real-robot deployment where environment interactions are expensive.

| Hyperparameter | Value |
|---|---|
| Discount γ | 0.99 |
| Replay buffer size | 1,000,000 transitions |
| Batch size | 256 |
| Actor/Critic hidden layers | [256, 256] |
| Activation | ReLU |
| Target entropy | automatic (`-dim(A)`) |

### Key empirical findings

1. **Feedback controller alone fails** on contact/friction tasks — brittle to model misspecification.
2. **RL alone fails** — exploration is too hard without the base controller providing near-success states from the start.
3. **Combination succeeds** — the base controller narrows exploration to the relevant region of state space.
4. **Residual action bounding is critical** — an unclipped residual destabilises the feedback controller and can damage the robot.

### Distinctions from Silver et al. (1812.06298)

| Aspect | Silver et al. | Johannink et al. |
|---|---|---|
| Environment | Simulation (MuJoCo) | Real robot (Kuka IIWA) + sim |
| RL algorithm | PPO (on-policy) | SAC (off-policy) |
| Action domain | Position / velocity | Torque |
| Task type | Long-horizon manipulation | Precision contact insertion |
| Residual clipping | Implicit via action space | Explicit hardware safety clip |

### Ariel relevance

- Confirms that **SAC is a viable alternative to PPO** for the residual in ariel's `37_train_residual_mtrl.py`, especially if training moves to real hardware where sample efficiency matters more than wall-clock throughput.
- Reinforces the **action clipping design** already present in `ResidualDroneEnv`: `action_total = clamp(prior + α·residual)`.
- The "residual narrows exploration" finding directly explains why hover reward at step 0 should already be high in ariel — the prior provides near-optimal returns before any RL training, so the critic has a useful gradient from the very first rollout.

## See Also

- [[CMA-ES_Algorithm]] — used to fit `cmaes_params` per morphology (the prior)
- [[CMA-ES_Parameters]] — budget / λ choices for `35c`
- [[competing_conventions]] — mixer sign conventions; must be correct for the prior to stabilise rather than destabilise

## Practical Notes (ariel-specific)

1. **Step-0 sanity check:** With `α` of the residual policy and a zero-initialised actor, the prior alone should yield high hover reward. If not, `HoverPrior` is mis-wired.
2. **~20% of library morphs have fragile priors:** `median_score` p10 ≈ 298 vs. `score` p10 ≈ 589, meaning some morphs have high-variance CMA solutions. These are the most likely failure points. Either re-run with 3× budget or down-weight them in training rotation.
3. **k-NN prior for OOD morphs:** At eval time, warm-start CMA-ES for an unseen morph from the k=5 nearest-library-neighbour mean in `morph_features` space. Trim components are per-motor and order-sensitive — sort by azimuth before interpolating.
