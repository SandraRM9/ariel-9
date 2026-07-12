---
type: algorithm_reference
tags: [algorithm, reinforcement-learning, drone, td3, sim-to-real, curriculum, end-to-end, rpm-control]
source: https://arxiv.org/abs/2311.13081
date_ingested: 2026-07-09
---

# Learning_to_Fly_in_Seconds

End-to-end quadrotor control framework that trains a direct RPM policy using TD3 + asymmetric actor-critic + curriculum learning, achieving Sim2Real transfer in **18 seconds of wall-clock training** on a consumer laptop GPU. No domain randomization required. Trains on only 0.3×10⁶ environment interactions — ~330× fewer than prior RPM-control work.

**Reference:** Eschmann, Albani, Loianno. "Learning to Fly in Seconds." IEEE RA-L, accepted April 2024. arXiv:2311.13081

**Codebase:** https://github.com/rl-tools/learning-to-fly (C++/CUDA, RLtools framework)

## Formulation

### State and Observation Spaces

**MDP state** (17-d, internal to simulator):
```
s = {p, q, v, ω, ω_m}
p     : position (3-d)
q     : orientation quaternion (4-d)
v     : linear velocity (3-d)
ω     : angular velocity (3-d)
ω_m   : motor speeds (4-d)
```

**Critic observations** (28-d, training only — privileged):
```
o_c = {p, R, v, ω, ω_m, f_r, τ_r}
R     : rotation matrix from q (9-d, replaces 4-d quaternion)
f_r   : random force disturbance, sampled per episode (3-d)
τ_r   : random torque disturbance, sampled per episode (3-d)
```

**Actor observations** (18 + N_H·4 dimensional, deployed on hardware):
```
o_a = {p, R, v, ω, H}
H : action history of last N_H RPM commands (N_H × 4-d)
```
Observation noise added to actor inputs (position σ=0.001, orientation σ=0.001, linear vel σ=0.002, angular vel σ=0.002). Quaternion→rotation matrix conversion used for both actor and critic to remove double-cover ambiguity.

**Why action history?** Motors have first-order low-pass dynamics (time constant τ_motor ≈ 0.15 s). At 100 Hz, this means actions only affect state after 5–25 control steps — severe partial observability. The history H provides the actor a proprioceptive proxy for the unobservable motor speeds.

### Action Space

Direct RPM setpoints (4-d):
```
a = {ω_sp1, ω_sp2, ω_sp3, ω_sp4}   # RPM setpoints for each motor
```
Motors modeled as first-order low-pass filters with τ = 0.15 s (empirically identified on Crazyflie).

### Reward Function

```
r(s, a, s') = -C_rp·‖p‖₂² - C_rq·(1 - q_w²) - C_rv·‖v‖₂² 
              - C_rω·‖ω‖₂² - C_ra·‖a - C_rab‖₂² + C_rs

p      : position error (from target)
q_w    : quaternion w component (= 1 when upright)
v      : velocity (penalises speed)
ω      : angular velocity (penalises spinning)
a      : current action
C_rab  : action baseline (= 0.334, hover equilibrium RPM normalised)
C_rs   : survival bonus (positive constant)
```

No explicit termination penalty term; instead, crashes end the episode and forfeit all future survival bonuses (the "learning to terminate" anti-pattern is mitigated by the survival bonus). Exact coefficient values are in the supplementary (`parameters.pdf` linked from GitHub repo).

### Curriculum Learning

Coefficients are annealed from `C_init,*` (permissive) toward `C_target,*` (strict) every 100,000 training steps by multiplying by `C_p,*` factors:

```
Every 100k steps:
  for each coefficient C_* in {C_rp, C_rq, C_rv, C_rω, C_ra}:
    C_* ← min(C_* × C_p,*, C_target,*)
```

Exploration noise is decayed on the same exponential schedule. Effect: early in training the agent faces easy rewards (small penalties, high survival bonus) allowing it to learn basic stability; later it must also track position precisely and use smooth commands.

## Parameters

### Training
| Name | Value | Role |
|---|---|---|
| RL algorithm | TD3 (off-policy) | Actor-critic; better sample complexity than PPO for continuous control |
| Parallel environments | 8,192 (64 blocks × 128 threads) | GPU parallelism |
| Simulation frequency | 100 Hz | Control loop rate |
| Integration timestep | 0.02 s (50 Hz physics) | Dynamics integration |
| Training steps (fast) | 300,000 | 18 s wall-clock; sufficient for real deployment |
| Training steps (full) | 3,000,000 | Marginal improvement over 300k |
| Curriculum update interval | 100,000 steps | |
| Simulator throughput | 1,284 M steps/s | NVIDIA T2000 GPU |
| Simulator speed | ~5 months flight/s | vs. 200k steps/s for Flightmare (~6420× faster) |
| Action history length N_H | (see supplementary) | Compensates motor delay |
| Motor time constant τ | 0.15 s | First-order low-pass; empirically identified on Crazyflie |

### Observation Noise (actor inputs)
| Component | Std |
|---|---|
| Position | 0.001 |
| Orientation | 0.001 |
| Linear velocity | 0.002 |
| Angular velocity | 0.002 |

### Physical Parameters (reference platform — MRS hex, not Crazyflie)
| Parameter | Value |
|---|---|
| Mass | 0.73 kg |
| Inertia J_xx, J_yy | 0.00791 kg·m² |
| Inertia J_zz | 0.01231 kg·m² |
| Arm length | 0.1202 m |
| Torque coefficient | 0.016 |
| Motor time constant | 0.04 s |
| Thrust model | quadratic: [-1.769, 0.00384, 1.33e-6] |

Crazyflie (27 g nano quadrotor) parameters differ; see manufacturer specs and repo `dynamics/` headers.

## Implementation Notes

### Asymmetric Actor-Critic Architecture

```
Training:
  critic  ←  o_c (28-d privileged: incl. motor speeds, disturbances)
  actor   ←  o_a (18+NH·4 d: no motor speeds, + action history)
  TD3 update on both critic and actor

Deployment:
  only actor is deployed; critic is discarded
  real sensors: VIO/mocap for p, v; IMU for ω; onboard rotation matrix
  motor speeds NOT measured — history H substitutes
```

### Sim2Real Transfer (no domain randomization)

The key insight: domain randomization forces policies to hedge against all randomized parameter values, reducing aggressiveness. Instead:
1. Match simulator physics carefully (grey-box motor model, motor delay, battery model)
2. Add random episode-level disturbances (f_r, τ_r) seen by critic but not actor → critic learns to compensate implicitly, actor learns robust behaviour
3. Add observation noise to actor inputs to simulate sensor imperfections

This produces aggressive policies tuned for the *actual* hardware characteristics. Contrast with [[empirical_sim_to_real]] (Swift/Kaufmann 2023) which fits GP+kNN residuals on real flight data — an alternative when grey-box identification is insufficient.

### Control Abstraction Level (Paper's Taxonomy)

The paper introduces a 6-level taxonomy from high-level to low-level:
```
Level 0: Position commands
Level 1: Velocity commands
Level 2.1: Attitude + thrust (domain params: mass)
Level 3.1: Body rates + thrust / CTBR (domain params: mass, rotational kinematics)
Level 4.2: Individual rotor thrusts (domain params: inertia, geometry)
Level 4.3: RPMs (domain params: thrust/torque curves)
Level 5.1: Motor RPM setpoints [THIS PAPER] (domain params: motor delay τ)
Level 5.2: Motor effort (domain params: battery level)
```

Each lower level adds non-linearities and domain parameters. [[Swift_Drone_Racing]] operates at Level 3.1 (CTBR) — this paper operates at Level 5.1, facing the full stack.

## When to Use

- **When direct RPM control is required** (no lower-level controller on hardware, or Crazyflie-style platforms with ESC-only interfaces).
- **When training budget is severely limited**: 18 s wall-clock with only 300k steps — dramatically lower barrier to entry than PPO-based approaches.
- **When hardware matches simulator closely**: Grey-box motor model must be identified. If platform is poorly characterised, use [[empirical_sim_to_real]] to fit residuals.
- **Off-policy preferred over PPO**: TD3 reuses transitions via replay buffer, better sample efficiency. All prior RPM-control RL work used PPO.

## Key Results

### Ablation (Table II) — full setup vs. ablations on Crazyflie
| Component removed | Real flights success | Notes |
|---|---|---|
| None (full setup) | 10/10 | Baseline |
| Observation noise | 9/10 | Marginal degradation |
| Reward recalculation in buffer | 9/10 | Minor |
| Exploration noise decay | 9/10 | Minor |
| Random disturbances (f_r, τ_r) | 9/10 | |
| Action history H | 7/10 | Motor delay unhandled |
| Asymmetric actor-critic | 2/10 | Critical component |
| Curriculum learning | 0/10 | Critical component |
| Motor delay simulation | 0/10 | Critical component |

### Trajectory Tracking (Table III) — figure-eight Lissajous, Crazyflie
| Controller | Slow (T=15s) | Normal (T=5.5s) | Fast (T=3.5s) |
|---|---|---|---|
| Geometric (min-snap) | **best** | competitive | poor |
| This work | competitive | competitive | **best** |
| Molchanov et al. 2019 (SRT, 84M steps) | oscillations | oscillations | oscillations |
| Gronauer et al. 2022 (SRT, 16M steps) | competitive | 7/10 crash | 10/10 crash |

Max tracked speed: 3 m/s, max acceleration: 0.9 g (agile trajectory).

## Limitations

- Exact coefficient values in supplementary only (not reproduced in paper body)
- No morphology adaptation — single platform (Crazyflie). No equivalent of [[single_controller_quadcopter]]'s cross-platform generalization
- Near-hover implicit assumption partially relaxed (agile trajectories work), but no explicit recovery from tumbling
- Battery-level adaptation not addressed (Level 5.2 not handled)
- C++ / RLtools framework — no Python bindings; porting to PyTorch/SB3 requires re-implementing TD3 + curriculum

## See Also

- [[Swift_Drone_Racing]] — same drone domain, Level 3.1 (CTBR), PPO, world-champion results
- [[empirical_sim_to_real]] — alternative Sim2Real approach (GP+kNN residuals vs. grey-box matching here)
- [[single_controller_quadcopter]] — cross-morphology Level 3.1 controller; uses domain randomization + RMA adaptation
- [[multirotor_control_taxonomy]] — the 6-level taxonomy introduced in this paper (concept page)
