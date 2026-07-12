---
type: algorithm_reference
tags: [algorithm, drone, control, geometric-control, se3, classical-control]
source: https://arxiv.org/abs/1411.2986
date_ingested: 2026-07-09
---

# Lee_Geometric_Control_SE3

A nonlinear geometric tracking controller for quadrotor UAVs that operates directly on the SE(3) manifold (ℝ³ × SO(3)), avoiding singularities inherent in Euler angle or quaternion representations. Provides the classical-control upper bound against which ariel's RL-based generalist controller is compared.

**Primary reference (non-adaptive):** Lee, Leok, McClamroch. "Geometric Tracking Control of a Quadrotor UAV on SE(3)." CDC 2010. arXiv:1003.2005.

**This page covers the adaptive extension:** Goodarzi, Lee, Lee. "Geometric Adaptive Tracking Control of a Quadrotor UAV on SE(3) for Agile Maneuvers." arXiv:1411.2986, 2014.

**Ariel implementation:** `src/ariel/simulation/drone/controllers/lee_control/lee_controller.py` — `LeeGeometricControl` class.

## System Dynamics

Configuration space: SE(3) = ℝ³ ⋊ SO(3).

```
# State variables:
x ∈ ℝ³          position (world frame)
v = ẋ ∈ ℝ³      velocity (world frame)
R ∈ SO(3)        rotation matrix (body → world)
Ω ∈ ℝ³          angular velocity (body frame)

# Equations of motion:
ẋ   = v
m·v̇ = m·g·e₃ - f·R·e₃ + W_x(x,v,R,Ω)·θ_x
Ṙ   = R·Ω̂
J·Ω̇ + Ω × J·Ω = M + W_R(x,v,R,Ω)·θ_R
```

Where `f ∈ ℝ` is total thrust, `M ∈ ℝ³` is total moment, `W_x, W_R` are uncertainty basis functions with unknown bounded parameters `θ_x, θ_R` (adaptive terms — omit for the non-adaptive version).

`Ω̂` denotes the hat map: `Ω̂·v = Ω × v` for all v.

## Error Definitions

```
# Tracking error (attitude):
Ψ(R, R_d) = ½·tr[I - R_dᵀ·R]          # error function ∈ [0, 2)
e_R = ½·(R_dᵀ·R - Rᵀ·R_d)∨           # attitude error vector (∨ = vee map)
e_Ω = Ω - Rᵀ·R_d·Ω_d                  # angular velocity error

# Tracking error (position/velocity):
e_x = x - x_d                           # position error
e_v = v - ẋ_d                           # velocity error
```

The attitude error function `Ψ ∈ [0, 2)` is zero only when `R = R_d` and avoids the ±2π wrapping of Euler angles. Critically, `Ψ < 1` (< 90° error) is required for convergence — if the vehicle flips beyond 90° from desired, the controller must be restarted.

## Control Laws

### Thrust command

```
f = (k_x·e_x + k_v·e_v + m·g·e₃ - m·ẍ_d) · R·e₃
```

The inner product with `R·e₃` projects the required force onto the current thrust direction.

### Desired attitude (computed from position error)

```
b₃_c = -(k_x·e_x + k_v·e_v + m·g·e₃ - m·ẍ_d) / ‖·‖    # desired thrust direction
b₁_c = b₁_d - (b₁_d·b₃_c)·b₃_c / ‖·‖                   # desired heading, projected
R_c  = [b₁_c×b₃_c / ‖·‖, b₁_c×b₃_c × b₃_c, b₃_c]        # = [b₂_c | −b₁_c×b₃_c | b₃_c]
```

where `b₁_d` is the desired heading direction (from yaw setpoint).

### Moment command

```
M = -k_R·e_R - k_Ω·e_Ω
  - W_R·θ̄_R                                          # adaptive compensation (optional)
  + (Rᵀ·R_c·Ω_c)∧·J·Rᵀ·R_c·Ω_c + J·Rᵀ·R_c·Ω̇_c   # feedforward terms
```

For the non-adaptive version (ariel's implementation), the `W_R·θ̄_R` term is omitted.

### Adaptive laws (Goodarzi et al. extension)

```
# Attitude adaptation:
θ̄̇_R = γ_R · W_Rᵀ · (e_Ω + c₂·e_R)

# Position adaptation (with projection to stay bounded):
θ̄̇_x = γ_x · W_xᵀ · (e_v + c₁·e_x)       if ‖θ̄_x‖ < B_θ
       = γ_x · (I - θ̄_x·θ̄_xᵀ/‖θ̄_x‖²) · W_xᵀ · (e_v + c₁·e_x)   otherwise
```

## Gain Parameters (ariel implementation)

| Gain | Symbol | Default (heavy ~1 kg) | auto_scale_gains | Role |
|---|---|---|---|---|
| `pos_P_gain` | k_x | [2.0, 2.0, 3.0] | — | Position PD stiffness |
| `vel_P_gain` | k_v | [3.0, 3.0, 4.0] | — | Velocity damping |
| `att_P_gain` | k_R | [0.3, 0.3, 0.1] | `I·ω_n²`, ω_n=12 rad/s | Attitude stiffness |
| `rate_P_gain` | k_Ω | [0.05, 0.05, 0.03] | `2·I·ω_n` | Angular rate damping |

**`auto_scale_gains=True` (recommended for hex drones):** Derives att/rate gains from the drone's inertia tensor `I_diag = diag(J)` and a natural frequency `ω_n = 12 rad/s` (critically damped second-order attitude response). Required for small drones (Izz ≈ 1e-4) where the 1 kg defaults saturate motor allocation immediately.

```python
att_P_gain  = I_diag * omega_n**2    # K_rot = I·ω_n²
rate_P_gain = 2.0 * I_diag * omega_n  # K_angvel = 2·I·ω_n  (critical damping)
```

## Implementation Notes

**Motor allocation:** The controller outputs `(f, M)` → motor speed commands via the thrust/torque allocation matrix (pseudo-inverse of the mixing matrix). For hexacopters, the system is over-actuated (6 motors, 4 DOF) so a minimum-norm allocation is used.

**NED vs ENU:** Ariel's validated path uses NED convention. The ENU branch in `_wrench_to_motor_commands` has a sign error in thrust allocation (drives `w_cmd` to motor floor). Until patched, use `orient="NED"` exclusively.

**Kinematic playback in `14_mujoco_lee_figure8.py`:** Because of the ENU sign bug, the Lee controller runs in the Python simulator (NED) and records the trajectory `(t, pos_NED, quat_NED)`. MuJoCo plays back the result kinematically (no physics) after NED→ENU conversion. This is a workaround, not the intended architecture.

**Convergence prerequisite:** `Ψ(R(0), R_d(0)) < 1` — the initial attitude error must be less than 90°. Manoeuvres that require crossing 90° tilt (e.g. aerobatics) require attitude mode switching.

## Stability Guarantees

**Attitude-only (Proposition 1):**
```
Given gains satisfying:
  c₂ < min{√(k_R·λ_min(J)) / λ_max(J),
            4k_Ω / (8k_R·λ_max(J) + (k_Ω + B₂)²)}
→ (e_R, e_Ω) → (0, 0) asymptotically.
```

**Full position+attitude (Proposition 2):**
```
Given Ψ(R(0), R_c(0)) < ψ₁ < 1:
→ (e_x, e_v, e_R, e_Ω) → (0, 0, 0, 0) asymptotically.
```

Region of attraction: all initial conditions with attitude error < 90°.

## In Ariel

### Location
`src/ariel/simulation/drone/controllers/lee_control/`

### Usage (from `14_mujoco_lee_figure8.py`)
```python
from ariel.simulation.drone.controllers.lee_control.lee_controller import LeeGeometricControl

ctrl = LeeGeometricControl(
    quad,
    yawType=1,          # 1 = yaw control enabled
    orient="NED",
    auto_scale_gains=True,
    pos_P_gain=np.array([args.pos_gain] * 3),
)
```

### Role as classical-control baseline

The Lee controller is the **analytically optimal upper bound** for trajectory tracking on a single known morphology. In the generalist-controller project (Stage 3), the Lee controller represents the performance ceiling that the prior + residual system should approach:

| Scenario | Expected performance |
|---|---|
| Known morphology, hover | CMA prior alone ≈ Lee hover |
| Known morphology, figure8 | Lee controller > prior+residual (baseline to close gap) |
| Unknown morphology, any task | Lee fails (requires exact params); prior+residual adapts |

The gap between Lee performance and prior+residual performance on trajectory tasks is the primary metric for evaluating whether the residual PPO is learning useful corrections beyond the hover prior.

### Comparison to RL-based control

| Property | Lee geometric | CMA prior (35c) | Prior + residual (37) |
|---|---|---|---|
| Requires known params? | ✅ exact mass, inertia, motors | ✅ (fitted via 60s CMA) | ✅ CMA + morph_features |
| Singularity-free? | ✅ (SE(3)) | ✅ (quaternion sim) | ✅ |
| Generalises to unseen morphs? | ❌ | ❌ (re-fit needed) | ✅ (morph_features) |
| Handles agile flight? | ✅ | ❌ (hover only) | Partial (trajectory tasks) |
| Works with non-quadrotor? | Limited | ✅ (any N motors) | ✅ (any N motors) |

## Experimental Validation (Goodarzi et al.)

Platform: m=0.755 kg, d=0.169 m rotor spacing, J ≈ diag(~2e-3, ~2e-3, ~4e-3) kg·m².

Demonstrated: 360° flip in <0.4 s, Lissajous 3D path tracking. Tracking errors bounded despite ~0.15s system latency.

## See Also

- [[Residual_Policy_Learning]] — the prior+residual architecture in ariel; Lee is the classical baseline
- [[morphology_conditioned_control]] — why Lee fails for generalisation across morphologies
- [[competing_conventions]] — NED/ENU sign conventions; critical for the Lee controller in ariel
