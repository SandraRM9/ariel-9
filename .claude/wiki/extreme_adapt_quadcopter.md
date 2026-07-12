---
type: algorithm_reference
tags: [algorithm, reinforcement-learning, morphology, drone, adaptation, imitation-learning, multitask]
source: https://hiperlab.berkeley.edu/wp-content/uploads/2024/09/2024_Zhang_ExtremeAdapt.pdf
date_ingested: 2026-07-09
---

# extreme_adapt_quadcopter

A learning-based low-level quadcopter controller combining PPO reinforcement learning with behavior cloning from a model-based expert, achieving **zero-shot adaptation across quadcopters with 3.7× mass difference and motor constants differing by 100×**. Extends [[single_controller_quadcopter]] (Zhang et al. 2023 / arXiv:2209.09232) with three key innovations: dual BC+RL training, torque-tracking reward, and design-informed domain randomization.

**Reference:** Zhang, Loquercio, Tang, Wang, Malik, Mueller. "A Learning-based Quadcopter Controller with Extreme Adaptation." IEEE Transactions on Robotics, 2025. (arXiv:2409.12949; PDF: hiperlab.berkeley.edu)

**Code:** https://github.com/muellerlab/xadapt_ctrl

## Formulation

Same two-phase RMA architecture as [[single_controller_quadcopter]], with an extended environmental parameter space and BC+RL hybrid objective:

```
# Phase 1 — Teacher (privileged training, BC + RL combined)
z_t  = µ(e_t)            # intrinsics encoder: 34-d params → 8-d latent
a_t  = π(x_t, z_t)      # base policy: state(17d) + latent(8d) → motors(4d)

# BC+RL loss (adaptive α schedule)
α     = exp(-0.001 · t_epoch)                    # decays from 1 → 0
R(π)  = (1 - α) · R_RL(π) - α · L_BC(π)
L_BC  = ‖a_expert - a‖²                         # MSE vs. model-based expert
R_RL  = E[Σ γ^t r_t]                            # PPO objective

# Phase 2 — Adaptation module (supervised, policy frozen)
ẑ_t  = ϕ({x_{t-100:t}, a_{t-100:t}})           # TCN over 100-step history
a_t  = π(x_t, ẑ_t)                              # deployment
```

The expert for BC is a **model-based low-level controller** with access to the ground-truth parameters of the randomized quadcopter at each episode — it adapts per episode, so the policy learns from a dynamically appropriate expert rather than a single fixed one.

## Parameters

### State vector x_t (17-d)
| Component | Dim | Notes |
|---|---|---|
| Rotation matrix (rows 1, 2) | 9 | Full SO(3) attitude — more than 2023 paper |
| Mass-normalised thrust | 1 | Commanded c_P |
| Angular velocity ω | 3 | Body frame |
| Commanded total thrust | 1 | From high-level |
| Commanded angular velocity | 3 | From high-level |

### Environmental parameters e_t (34-d — teacher only, extended from 18-d in 2023)
| Parameter | Training range | Testing range |
|---|---|---|
| Mass (kg) | [0.226, 0.950] | [0.205, 1.841] |
| Arm length (m) | [0.046, 0.200] | [0.040, 0.220] |
| MMOI Ixx, Iyy (kg·m²) | [1.93e-4, 5.40e-3] | [1.73e-5, 2.27e-2] |
| MMOI Izz (kg·m²) | [2.42e-4, 8.51e-3] | [2.10e-4, 3.40e-2] |
| Prop κ (torque-to-thrust, m) | [0.0069, 0.0161] | [0.0051, 0.0170] |
| Prop C_F (thrust-to-ω², kg/rad²) | [3.88e-8, 8.40e-6] | [3.24e-9, 1.02e-4] |
| Body drag coefficient | [0, 0.74] | [0, 1.15] |
| Max motor speed (rad/s) | [800, 8044] | [400, 10021] |
| Motor effectiveness factor (×4) | [0.7, 1.3] | [0.7, 1.3] |
| Payload (% of mass) | [18, 40] | [18, 40] |
| Payload CoM offset (% arm) | [-50, 50] | [-50, 50] |
| Motor time constant (s) | 0.01 | 0.01 |
| Mixer matrix | 4×4 | — |
| External torque (3-d) | randomised | — |

### Latent z_t (8-d)
8-d bottleneck, determined by binary search on latent dimension optimising learning performance. Larger than the 6-d used in the 2023 paper due to the wider 34-d env parameter space.

### Action a_t (4-d)
Individual motor speed commands (rad/s), clipped to per-motor max.

### Reward function (Phase 1)
```
r_t = - ‖a_t - a_{t-1}‖              # output smoothing penalty
    + δt                               # survival reward (step size)
    - ‖c_P^t - c_P^t_des‖            # mass-normalised thrust tracking
    - ‖τ_t - τ_des^t‖                # torque tracking (KEY CHANGE from 2023)
```

Commanded torque from rate controller:
```
τ_des = J · K · (ω_des - ω) + ω × (J · ω)
K = diag(20, 20, 4)  s⁻¹             # roll/pitch gain 20, yaw 4
```

**Why torque not angular velocity?** Torque responds immediately to motor speed commands; angular velocity requires temporal integration. Torque reward gives denser, more direct gradient for a high-bandwidth (500 Hz) low-level controller. Ablation showed torque tracking reward significantly outperforms angular velocity reward in both speed and stability of training convergence.

## Design-Informed Domain Randomization

The paper's key contribution is physically correlated domain randomization — not independent parameter sampling. A single **size factor c ∈ [0,1]** drives correlated scaling:

```
l   = c(l_max - l_min) + l_min                # arm length (linear in c)
m   = c_m(m_max - m_min) + m_min              # c_m = (l³ - l_min³)/(l_max³ - l_min³)
J   = c_J(J_max - J_min) + J_min              # c_J = (l⁵ - l_min⁵)/(...)
C_d = c_Cd(Cd_max - Cd_min) + Cd_min         # c_Cd = (l² - l_min²)/(...)
C_F = C_Fmin · (C_Fmax/C_Fmin)^c             # exponential in c (motor thrust)
```

Remaining parameters (max motor speed, prop constant, etc.) scale linearly with c. After scaling, **±20% uniform noise** is added to all parameters for flexibility. Additionally, **per-rotor motor efficiency** [0.7, 1.3] models hardware variation and partial failures.

This approach ensures physically plausible quadcopters (e.g., prevents small-light bodies with impossibly powerful motors) and reduces the likelihood of degenerate samples that would destabilise training.

## Network Architectures

| Module | Architecture | I/O |
|---|---|---|
| Base policy π | 3-layer MLP, 256 hidden | (17 + 8) → 4 |
| Intrinsics encoder µ | 2-layer MLP, 128 hidden | 34 → 8 |
| Adaptation module ϕ | 3-layer 1D-CNN | 100 × (17+4) → 8 |

1D-CNN layers for ϕ:
```
Layer 1: [32 in, 32 out, kernel=8, stride=4]
Layer 2: [32 in, 32 out, kernel=5, stride=1]
Layer 3: [32 in, 32 out, kernel=5, stride=1]
→ flatten → linear projection → 8-d ẑ_t
```

History window: **100 steps** (vs 400 in 2023 paper). At 500 Hz, 100 steps = 200 ms. The shorter window forces faster adaptation and reduces memory.

## Training Hyperparameters

| Parameter | Value |
|---|---|
| Phase 1 algorithm | PPO (PyTorch) |
| Phase 1 steps | 100M |
| Phase 1 wall-clock | ~1.5h on 1 GPU |
| Phase 2 algorithm | ADAM + MSE |
| Phase 2 steps | 10M |
| Phase 2 wall-clock | ~20 min |
| Phase 2 training data | last 1M rollout steps |
| Control frequency | 500 Hz |
| Sim step | 2 ms |
| Observation latency | 5 ms (reduced from 10ms in 2023) |
| Episode max duration | 5 s |
| Early termination: height | < 2 cm |
| Early termination: body rate | > 10 rad/s |
| BC weight α | exp(−0.001 · t_epoch), 1→0 |
| Discount γ | standard PPO default |
| Deployment inference | MNN (Mobile Neural Network) format |

## Key Results

### Simulation (Table III — in-distribution test range)
| Method | Success rate | Pos RMSE (m) | Vel RMSE (m/s) |
|---|---|---|---|
| PID-PDn (nominal model only) | 22% | 0.510±0.372 | 0.845±1.066 |
| L1-PDn (adaptive high-level) | 62% | 0.186±0.167 | 0.278±0.392 |
| PID-L1 (adaptive low-level) | 77% | 0.221±0.242 | 0.357±0.481 |
| **PID-Ours** | **100%** | **0.154±0.079** | **0.117±0.068** |
| PID-PD* (ground-truth params) | 100% | 0.061±0.057 | 0.059±0.050 |

### OOD generalisation (δ = size-factor deviation from nominal)
- **Training range:** δ ≤ 0.5 (λ ∈ [0,1])
- **δ = 8 test (16× wider than training):** >95% success rate until δ=8, drops to 84%
- Pos RMSE increase from δ=0 to δ=8: **+4.6%** (ours) vs **+481%** (best baseline PID-L1)

### Real hardware
- 100% success, zero-shot, on quadcopters with **3.7× mass and 3.1× arm-length difference**
- Comparable to platform-specific PID-PD* controller that was in-flight tuned
- Motor constants differing by **>100×** between platforms

## Comparison to ariel's Generalist Controller

| Aspect | Zhang et al. 2025 (this paper) | Ariel (spear/library) |
|---|---|---|
| Morphology variation | Physical params (mass, inertia, motors) — continuous | Hex geometry + spin + prop — discrete morphology set |
| Training signal | BC + RL (BC dominant early, RL dominant late) | RL only (PPO) |
| Expert for BC | Adaptive model-based controller (per-episode params) | N/A — analytical prior is used as env offset, not BC target |
| Morph representation | Learned 8-d latent, online adaptation (100 steps) | Hand-crafted 22-d feed-forward + 11-d CMA prior |
| Tasks | Trajectory tracking (agile) | 5 tasks: hover + 4 trajectory |
| Domain randomization | Design-informed (correlated via size factor) | Stratified sampling (hex_sampler.py) |
| Action space | 4 motors | 6 motors |
| Sim-to-real gap | Pearson r=0.65 (validated on real hardware) | Sim only (no real hardware yet) |

**Key idea to adopt in ariel:**

1. **BC from the CMA prior.** Instead of using the prior only as an additive env offset (current approach), add a BC loss term: `L_BC = ‖prior_action - total_action‖²` with decaying weight. This would accelerate early training by explicitly imitating the prior's hover behaviour rather than relying on the residual to discover it via reward alone.

2. **Design-informed domain randomization for hex.** `hex_sampler.py` currently uses independent ranges per parameter. Adapting the size-factor correlation approach (mass ∝ arm_length³, inertia ∝ arm_length⁵) would produce more physically plausible hex morphologies and reduce degenerate samples in the training rotation.

3. **Shorter history window for adaptation module.** If ariel adds an online adaptation module (see [[single_controller_quadcopter]] §Fallback), use 100 steps (200 ms at 500 Hz) rather than 400, forcing faster latent convergence.

4. **Torque tracking reward.** For trajectory tasks where inner-loop rate control matters, rewarding torque rather than angular velocity provides denser gradient (same argument applies to hex 6-motor case).

## See Also

- [[single_controller_quadcopter]] — predecessor (2023); same RMA architecture, fewer innovations
- [[Residual_Policy_Learning]] — complementary architecture (prior + additive residual vs. pure adaptation)
- [[morphology_conditioned_control]] — taxonomy of morphology-conditioned control approaches
