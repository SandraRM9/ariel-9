---
type: algorithm_reference
tags: [algorithm, reinforcement-learning, drone, sim-to-real, ppo, perception, racing]
source: https://www.nature.com/articles/s41586-023-06419-4
date_ingested: 2026-07-09
---

# Swift_Drone_Racing

Deep RL system that achieves world-champion-level FPV drone racing by combining PPO training in simulation with empirical residual models for sim-to-real transfer. Distinct from [[single_controller_quadcopter]] and [[extreme_adapt_quadcopter]] in that it targets **aggressive high-speed flight** (not near-hover adaptation) and uses **empirical data-driven residual models** rather than domain randomization.

**Reference:** Kaufmann, Bauersfeld, Loquercio, Müller, Koltun, Scaramuzza. "Champion-level drone racing using deep reinforcement learning." Nature 620:982–987, 2023. DOI: 10.1038/s41586-023-06419-4

## Formulation

### Control Policy

```python
# Two-layer MLP; input is 31-d observation vector
obs = concat([
    platform_state,   # 15-d: position(3) + velocity(3) + rotation_matrix(9)
    gate_corners,     # 12-d: relative positions of 4 gate corners (3d each)
    prev_action,      # 4-d: previous mass-normalised thrust + body rates
])  # total: 31-d

action = policy(obs)  # 4-d: [collective_thrust_normalised, body_rates(3)]
```

### Quadrotor Dynamics

Core state:
```
ẋ = [ṗ_WB, q̇_WB, v̇_W, ω̇_B, Ω̇]^T

p_WB : position (world)
q_WB : attitude quaternion
v_W  : inertial velocity
ω_B  : body rates
Ω    : motor speeds
```

Propeller model (quadratic in motor speed):
```
f_i(Ω_i) = [0, 0, c_l · Ω_i²]^T     # lift
τ_i(Ω_i) = [0, 0, c_d · Ω_i²]^T     # drag torque
```

Aerodynamic forces (grey-box polynomial in body-frame velocity v_B and avg motor speed Ω_avg):
```
f_x  ∝ v_x + v_x|v_x| + Ω² + v_x·Ω²
f_y  ∝ v_y + v_y|v_y| + Ω² + v_y·Ω²
f_z  ∝ v_z + v_z|v_z| + v_xy + v²_xy + v_xy·Ω² + v_z·Ω² + v_xy·v_z·Ω²
τ_x  ∝ v_y + v_y|v_y| + Ω² + v_y·Ω² + v_y|v_y|·Ω²
τ_y  ∝ v_x + v_x|v_x| + Ω² + v_x·Ω² + v_x|v_x|·Ω²
τ_z  ∝ v_x + v_y
```
Coefficients identified from motion-capture flight data.

### Reward Function

```
r_t = r_prog + r_perc + r_cmd - r_crash

r_prog = λ₁ · (d_{t-1}^Gate - d_t^Gate)         # progress toward next gate

r_perc = λ₂ · exp(λ₃ · δ_cam⁴)                  # gate visible in camera
         # δ_cam: angle between camera axis and gate center

r_cmd  = λ₄·‖a_t^ω‖ + λ₅·‖a_t - a_{t-1}‖²      # penalise jerky commands

r_crash = 5.0 if (p_z < 0 or gate collision) else 0   # terminates episode
```

`r_perc` is the key innovation: explicitly rewarding keeping the next gate in the camera field-of-view stabilises the VIO drift-correction Kalman filter during high-speed flight. Without it, perception degrades in high-G manoeuvres.

## Parameters

### Policy Network
| Name | Value | Role |
|---|---|---|
| Architecture | 2-layer MLP | Actor and critic (separate weights) |
| Hidden width | 128 | Per layer |
| Activation | LeakyReLU (slope 0.2) | |
| Input dim | 31 | Observation vector |
| Output dim | 4 | Collective thrust + body rates |
| Value network | Same arch | Critic; sees privileged state during training |

### PPO Training
| Name | Value | Role |
|---|---|---|
| Algorithm | PPO | On-policy |
| Parallel agents | 100 | Simultaneously simulated |
| Episode length | 1,500 steps | |
| Total interactions (Phase 1) | 1 × 10⁸ | |
| Fine-tuning interactions | 2 × 10⁷ | |
| Optimizer | Adam | |
| Learning rate | 3 × 10⁻⁴ | Both policy and value |
| Training time (Phase 1) | ~50 min | i9-12900K + RTX 3090 |
| Domain randomization | None | Replaced by empirical residual models |

### Hardware (Autonomous Drone)
| Component | Spec |
|---|---|
| Frame | Armattan Chameleon 6″ |
| Motors | T-Motor Velox 2306 |
| Propellers | 5″ three-bladed |
| Total weight | 870 g |
| Max static thrust | ~35 N |
| Thrust-to-weight ratio | 4.1 |
| Onboard compute | NVIDIA Jetson TX2 (6-core @ 2GHz, 256 CUDA cores) |
| State estimation | Intel RealSense T265 VIO @ 100 Hz |
| Flight controller | STM32 running Betaflight |
| Sensorimotor latency | 40 ms |

### Gate Detection Network
| Parameter | Value |
|---|---|
| Architecture | U-Net, 6-level encoder-decoder |
| Filter sizes | (8, 16, 16, 16, 16, 16) + 12-filter output |
| Input resolution | 384 × 384 px |
| Inference time | 40 ms (Jetson TX2, TensorRT FP16) |
| Activation | LeakyReLU (α = 0.01) |
| Output | Gate corner pixel coordinates |

## Implementation Notes

### Three-Phase Training Pipeline

```
Phase 1: PPO in idealized sim (perfect state, idealized dynamics)
         → policy_0

Phase 2: Deploy policy_0 on real track (~3 rollouts, ~50s each)
         Collect (state_estimated, state_groundtruth, action) tuples
         Fit GP residual (perception) + kNN residual (dynamics)
         → augmented_simulator

Phase 3: Fine-tune with PPO in augmented_simulator (2×10⁷ steps)
         → policy_final
```

See [[empirical_sim_to_real]] for the GP+kNN residual model details.

### Kalman Filter for VIO Drift Correction

State: `x = [p_drift, v_drift]^T ∈ R^6`

Measurement update triggered whenever gate detector finds corners → IPPE pose → fused with VIO. Process noise: σ_pos = 0.05, σ_vel = 0.1. The filter corrects VIO drift mid-flight using detected gate poses as absolute reference anchors.

### Low-Level Controller (Betaflight)

Policy outputs **mass-normalised collective thrust + body rate setpoints** (4-d). Betaflight's PID tracks body rates at 500 Hz on the STM32. This clean interface decouples the RL policy from low-level motor mixing — the same interface used in [[single_controller_quadcopter]] and [[extreme_adapt_quadcopter]].

Key Betaflight quirks in identification:
- D-term reference: zero (pure rate damping, no trajectory derivative)
- I-term resets on throttle cut
- Motor saturation prioritises body rate tracking over thrust

## When to Use

- **Aggressive trajectory tasks** (racing, aerobatics) where near-hover assumptions break down — distinct from the RMA papers which explicitly target near-hover.
- **Sim-to-real without domain randomization**: when real flight data (~3 rollouts) is available and domain randomization produces conservative/slow policies.
- **Perception-in-the-loop control**: when the onboard camera contributes to state estimation — `r_perc` is the template for any reward that incentivises keeping a visual target in frame.

## Key Results

| Metric | Value |
|---|---|
| Races won vs. Vanover (DRL world champion) | 5/9 |
| Races won vs. Bitmatta (MultiGP champion) | 4/7 |
| Races won vs. Schaepper (Swiss champion) | 6/9 |
| Swift's fastest lap vs. Vanover's best | +0.5 s faster |
| Losses attributable to being slower | 20% |
| Losses due to gate collision | 40% |
| Losses due to opponent collision | 40% |

Swift advantages: 120 ms faster start reaction, tighter Split-S turning radius, more consistent lap times. Human advantages: strategic pace management, earlier gate orientation, faster in certain individual segments.

## See Also

- [[empirical_sim_to_real]] — GP + kNN residual model methodology (concept page)
- [[single_controller_quadcopter]] — near-hover adaptation; same action space convention
- [[extreme_adapt_quadcopter]] — morphology-adaptive version of the same low-level interface
- [[Residual_Policy_Learning]] — residual *policy* (different concept: additive on a prior, not sim-to-real data correction)
