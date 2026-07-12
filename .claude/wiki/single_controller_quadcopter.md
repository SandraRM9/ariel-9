---
type: algorithm_reference
tags: [algorithm, reinforcement-learning, morphology, drone, adaptation, multitask, concept]
source: https://arxiv.org/abs/2209.09232
date_ingested: 2026-07-09
---

# single_controller_quadcopter

A two-phase teacher-student RL framework that trains a **single near-hover position controller** deployable on quadcopters with 4.5× mass and 2.9× arm-length differences — zero-shot, without sim-to-real fine-tuning.

**Reference:** Zhang, Loquercio, Wu, Kumar, Malik, Mueller. "Learning a Single Near-Hover Position Controller for Vastly Different Quadcopters." arXiv:2209.09232, ICRA 2023.

**This is the closest published baseline to ariel's generalist hex-drone controller project.** See [[Residual_Policy_Learning]] for the architectural complement (residual-on-prior vs. pure adaptation). See [[morphology_conditioned_control]] for the general concept.

## Formulation

### Two-network architecture (RMA-style)

```
# Phase 1 — Teacher policy (trained with privileged info)
z_t       = μ(e_t)           # environment encoder: 18-d params → 6-d latent
a_t       = π(x_t, z_t)     # base policy: state + latent → motor commands

# Phase 2 — Adaptation module (trained supervised, policy frozen)
ẑ_t       = φ({x_{t-k:t}, a_{t-k:t}})    # estimate z from 400-step history
a_t_test  = π(x_t, ẑ_t)    # deployment: replace μ(e_t) with φ(history)
```

`e_t` (18-d privileged environment parameters) is available during training but **not at deployment**. The adaptation module φ learns to recover `z_t` from observable history alone.

### Observation space (x_t, 23-d)

| Component | Dim | Notes |
|---|---|---|
| Position | 3 | World frame |
| Velocity | 3 | World frame |
| Rotation matrix (rows 1,2) | 6 | Attitude (not full 9 — 2 rows sufficient) |
| Mass-normalised thrust command | 1 | Commanded c_des |
| Angular velocity | 3 | Body frame |
| Commanded total thrust | 1 | High-level |
| Commanded angular velocity | 3 | High-level setpoint |
| (padding / extra) | 3 | Total 23 |

### Action space (a_t, 4-d)
Individual motor speed commands for a quadcopter. Clipped to feasible range.

### Environment parameters (e_t, 18-d — teacher only)

| Parameter | Training range |
|---|---|
| Mass | [0.142, 0.950] kg |
| Arm length | [0.046, 0.200] m |
| Inertia Ixx, Iyy | [7.42e-5, 5.60e-3] kg·m² |
| Inertia Izz | [1.20e-4, 8.80e-3] kg·m² |
| Propeller κ (torque-to-thrust ratio) | [0.0041, 0.0168] m |
| Motor constant | [1.15e-7, 7.64e-6] |
| Body drag coefficient (3-d) | [0, 0.62] |
| Max motor speed | [707, 4895] rad/s |
| Payload mass | 10–50% of vehicle mass |
| Payload location | ±10% of arm length |

### Reward (Phase 1 PPO)

```
r_t = 1.0 * δt                              # survival (step size)
    - 0.01 * ‖ω_t - ω_des_t‖               # angular velocity tracking
    - 0.02 * ‖c_t - c_des_t‖               # mass-normalised thrust tracking
    - 0.0002 * ‖m_t - m_{t-1}‖             # output oscillation penalty
```

Note: reward is **not** direct position tracking — a high-level P-controller converts position error to desired angular velocity + thrust commands outside the RL agent. The RL agent tracks those inner-loop setpoints.

## Parameters

| Name | Value | Role |
|---|---|---|
| Latent z dim | 6 | Bottleneck between env params and policy |
| History window k | 400 steps | Adaptation module input length |
| Base policy MLP | [256, 256, 256] | 3 hidden layers |
| Env encoder μ MLP | [128, 128] | 2 hidden layers |
| Adaptation φ — CNN layers | [32→32 k=8 s=4], [32→32 k=5 s=1], [32→32 k=5 s=1] | 1D-CNN on history |
| Phase 1 algorithm | PPO | On-policy, 100M steps, ~2h on 1 GPU |
| Phase 2 algorithm | Adam + MSE | Supervised, 10M steps, ~20 min |
| Phase 2 training data | 1M most recent rollout steps | From Phase 1 replay |
| Control frequency | 500 Hz | Sim step 2 ms |
| Observation latency | 10 ms | Modelled in sim |
| Episode length | 5 s max | Early termination at h < 2 cm |
| High-level ω_n | 2 rad/s | Natural frequency |
| High-level ζ | 0.7 | Damping ratio |

## Implementation Notes

**Two-phase training:**
1. **Phase 1 (teacher):** PPO with ground-truth `e_t` available. Policy + encoder train together. Policy learns to exploit `z_t` for morph-conditional control.
2. **Phase 2 (student):** Policy is frozen. Adaptation module φ trained with supervised loss `‖z_t - ẑ_t‖²` to recover the latent from observable history. The separation keeps Phase 2 lightweight (20 min vs 2h).

**Why 6-d latent and not 18-d?**
The encoder μ compresses e_t to 6-d. The adaptation module φ must then predict 6-d not 18-d — much easier from noisy sensor history. A direct 18-d SysID baseline performed worse (tested as ablation).

**Adaptation speed:** ~2 seconds wall-clock to detect and adapt to sudden 290 g payload addition (~36% of body mass). This latency is determined by the history window length (400 steps × 2 ms = 800 ms nominal, but convergence takes ~2.5× window).

**High-level / low-level split:**
The RL agent is an **inner-loop** (body-rate + thrust tracking) controller. A classical P-controller outer loop converts position errors to desired angular velocity + thrust. The RL agent never sees raw position goals — only inner-loop setpoints. This makes the problem easier to learn and is architecturally cleaner.

## When to Use

- **Single policy, multiple morphologies** where physical parameters vary continuously (mass, inertia, motor strength) — exactly this paper's setting.
- **No prior analytical controller available** for the prior — pure RMA without a residual base.
- **Near-hover tasks only** — authors explicitly note it fails on aggressive trajectory tracking (stated limitation).
- For **sim-to-real** where domain randomization + adaptation module can bridge the gap.

## Key Results

| Scenario | Proposed | Best competitor (ℒ₁) |
|---|---|---|
| In-dist success rate | 66% | 59% |
| In-dist height error | 0.09 m | 0.17 m |
| OOD ext. force success | 49% | 42% |
| OOD ext. force height error | 0.09 m | 0.30 m |
| OOD motor failure success | 38% | 33% |
| Real-robot (inertia board) success | 100% | 0% (LTF, PID both fail) |

Zero-shot cross-platform: tested on two physical quadcopters (4.5× mass difference) without any modification or re-tuning.

## Comparison to Ariel's Generalist Controller

| Aspect | Zhang et al. (this paper) | Ariel (spear/library) |
|---|---|---|
| Morphology variation | Physical params only (mass, inertia, motors) | Full hex geometry + spin pattern + prop size |
| Base controller | None (pure RL) | Analytical CMA-tuned hover prior (`35c`) |
| Morph representation | Learned 6-d latent via online adaptation | Hand-crafted 22-d `morph_features` (feed-forward) |
| Adaptation at deploy time | Online (TCN over 400-step history) | Offline (60s CMA) + optional 200k PPO fine-tune |
| Tasks | Near-hover only | 5 tasks (hover + 4 trajectory) |
| RL algorithm | PPO (Phase 1) | PPO (SB3) |
| Action space | 4 motors (quadcopter) | 6 motors (hexacopter) |

**Key design choice difference:** Ariel uses **feed-forward morph_features** rather than **online adaptation**. This is faster at deploy time but requires the morph to be known at construction. If morph_features prove insufficient for OOD morphs, the Zhang et al. adaptation module (1D-CNN over history) is the fallback architecture — see §Fallback below.

**Fallback:** If ariel's Stage-4 held-out morph eval shows the residual generalises poorly from morph_features alone, replace / augment morph_features with an online adaptation module φ predicting the 22-d (or a compressed k-d) morph latent from the last k state-action pairs. This is the RMA-to-ariel translation.

## Successor Paper

[[extreme_adapt_quadcopter]] (Zhang et al. 2025, IEEE T-RO, arXiv:2409.12949) extends this work with:
- **BC + RL hybrid training** (behaviour cloning from a model-based expert with decaying weight)
- **Torque-tracking reward** instead of angular velocity tracking — denser gradient for 500 Hz controller
- **Design-informed domain randomization** (physically correlated via size factor c)
- **Wider env parameter space**: 34-d (vs 18-d here); 8-d latent (vs 6-d here)
- **Shorter history window**: 100 steps (vs 400 here)
- Achieves 16× OOD generalisation beyond training range; motor constants differing 100× in hardware tests

## See Also

- [[Residual_Policy_Learning]] — complementary architecture (prior + residual) rather than pure adaptation
- [[morphology_conditioned_control]] — general concept page

## Practical Notes

1. The 6-d latent bottleneck is key: don't expose 18-d raw params to the policy. The ariel analog is `morph_features (22-d) → consider compressing to 6–8-d via linear projection` if the actor's input space becomes unwieldy.
2. Phase-2 supervised training is cheap (20 min). If ariel needs online adaptation, Phase 2 can be added without retraining Phase 1 — the base policy only needs to use a latent.
3. History window of 400 steps (at 500 Hz = 800 ms) is tailored to the adaptation latency requirement. For ariel's 100 Hz control, 400 steps = 4 s — may need to shorten.
4. The high-level / low-level split (outer P-loop + inner RL) is worth adopting for ariel's trajectory tasks: let RL track inner-loop setpoints, not raw waypoints.
