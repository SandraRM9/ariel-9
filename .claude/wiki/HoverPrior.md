---
type: ariel_reference
tags: [ariel, residual-policy, drone, analytical-controller, prior]
source: ariel project source code
date_ingested: 2026-07-10
---

# HoverPrior

Analytical morphology-conditional hover prior implementing closed-form PD + yaw-damping control for drone altitude and attitude stabilization.

## Location

`examples/spear/library/prior_controller.py`

## Purpose

Provides a fixed, non-learned analytical controller that:
- Tracks altitude at a target NED height
- Damps roll/pitch angles and body rates
- Damps yaw rate
- Can be composed with a residual policy for multi-task learning

All calculations are closed-form (no gradients needed) and vectorized over batch dimensions. Exposes two compositional levels:

1. **`prior_effort(state, params)`** — raw per-motor effort (compose with residual here)
2. **`prior_action(state, params)`** — fully clamped motor commands (ready for dynamics)

## Signature

```python
class HoverPrior:
    def __init__(
        self,
        propellers: Sequence[dict],
        params: dict,
        target_ned: Sequence[float],
        *,
        gravity: float = 9.81,
        action_scale: float = 0.4,
        twr: float | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        ...

    def prior_effort(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def effort_to_action(self, effort: torch.Tensor) -> torch.Tensor:
        ...

    def prior_action(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        ...

    def default_init_params(
        self,
        *,
        mass: float | None = None,
        inertia: np.ndarray | None = None,
    ) -> np.ndarray:
        ...
```

## Parameters (Constructor)

| Name | Type | Description |
|------|------|-------------|
| `propellers` | `Sequence[dict]` | List of propeller dicts with keys `loc` (xyz position), `dir` (direction + descriptor tuple), `propsize` (int). Determines mixer sign convention and motor count. |
| `params` | `dict` | Output of `ariel.simulation.drone.dynamics_params.derive_reference_params()` for this morphology. Provides `k_w`, `w_min`, `w_max`, `k` (squared term constant). |
| `target_ned` | `Sequence[float]` | 3-vector hover target in NED coords (z negative for altitude above ground). Typically `(0.0, 0.0, -1.5)`. |
| `gravity` | `float` | m/s² (default 9.81). |
| `action_scale` | `float` | Multiplier applied to effort before adding `u_hover`. Default 0.4 (35c/residual env convention). Controls feedback gain magnitude. |
| `twr` | `float \| None` | Thrust-to-weight ratio. If provided, auto-reduces `action_scale` by `(TWR_REF / twr)²` (clamped ≤ 1). Prevents high-TWR morphs (prop-6/7 hexes) from over-correcting and crashing. |
| `device` | `torch.device \| str` | PyTorch device (`"cpu"`, `"cuda:0"`, etc.). Tensors are precomputed and cached on this device. |
| `dtype` | `torch.dtype` | Tensor dtype (default `torch.float32`). |

## Properties

| Name | Type | Returns | Description |
|------|------|---------|-------------|
| `param_dim` | property | `int` | N (motors) + 5 (gains) = N + 5. Always 11 for hex (6 + 5). |
| `state_dim_min` | property | `int` | 12 (position, velocity, euler, body rates; motor speeds beyond dim 12 are ignored). |
| `n_motors` | attribute | `int` | Number of motors (from propellers length). |

## Reference Constants

```python
N_GAINS = 5  # k_alt_p, k_alt_d, k_tilt, k_rate, k_yaw_rate
GRAVITY_DEFAULT = 9.81  # m/s²
REF_MASS = 0.20         # kg (reference morph for gain warm-start)
REF_IYY = 1.5e-3        # kg·m² (reference pitch inertia)
REF_IZZ = 3.0e-3        # kg·m² (reference yaw inertia)
```

## Parameter Vector Layout

The `params` (cmaes_params) vector is 11-dimensional for hex:

```
params[0:6]   = trim        per-motor static thrust bias (N values)
params[6]     = k_alt_p     P gain on altitude error (NED z), scales with mass
params[7]     = k_alt_d     D gain on vertical velocity, scales with mass
params[8]     = k_tilt      P gain on roll/pitch angle, scales with inertia
params[9]     = k_rate      D gain on roll/pitch body rates, scales with inertia
params[10]    = k_yaw_rate  D gain on body yaw rate, scales with yaw inertia
```

The `trim` component is fitted by `_analytical_trim()` to null static roll/pitch moments; the gains are optimized by CMA-ES in scripts like `35c_hover_cmaes_minimal.py`.

## State Layout (Expected Input)

```python
state[0:3]   = pos_ned     (x, y, z; z↓ positive, NED frame)
state[3:6]   = vel_world   (vx, vy, vz, world frame)
state[6:9]   = euler       (phi=roll, theta=pitch, psi=yaw) ZYX intrinsic
state[9:12]  = body_rate   (p, q, r) body-frame angular velocity
state[12:]   = motor_w     (optional; not used by prior but may be present)
```

## Methods

### `prior_effort(state: torch.Tensor, params: torch.Tensor) -> torch.Tensor`

Compute per-motor raw effort before clamp/scale transformation.

**Signature:**
```python
@torch.no_grad()
def prior_effort(
    self,
    state: torch.Tensor,  # (B, 12+) or (12+,)
    params: torch.Tensor,  # (B, N+5) or (N+5,)
) -> torch.Tensor:  # (B, N) or (N,)
```

**Behavior:**

Computes four control terms that sum into effort:

```
alt_cmd  = k_alt_p * (z - target_z) - k_alt_d * vz
att_cmd  = k_tilt * (mix_pitch * theta + mix_roll * phi)
rate_cmd = k_rate * (mix_pitch * q + mix_roll * p)
yaw_cmd  = k_yaw_rate * yaw_mix * r

effort = trim + alt_cmd + att_cmd + rate_cmd + yaw_cmd
```

Where `mix_*` are precomputed per-motor mixer matrices from `tilt_mixer()` and `yaw_mixer()`.

**Parameters:**
- `state`: `(B, state_dim)` or `(state_dim,)`. Batch dimension optional.
- `params`: `(B, N+5)` or `(N+5,)`. Must have matching batch dimension with state (or be automatically broadcast).

**Returns:**
- `(B, N)` or `(N,)` raw effort tensor, ready for composition with residual.

**Usage:**
```python
effort = prior.prior_effort(state, params)
total_effort = effort + alpha * residual_action
action = prior.effort_to_action(total_effort)
```

### `effort_to_action(effort: torch.Tensor) -> torch.Tensor`

Transform raw effort to clamped motor commands via 35c's clamp+scale+u_hover pipeline.

**Signature:**
```python
def effort_to_action(self, effort: torch.Tensor) -> torch.Tensor:
```

**Formula:**
```
action = clamp(u_hover + clamp(effort, ±1) · action_scale, ±1)
```

Where:
- `u_hover` is precomputed analytically at __init__
- `action_scale` is the feedback gain magnitude (default 0.4, auto-reduced for high-TWR morphs)
- Inner clamp prevents saturation blow-up
- Outer clamp ensures valid motor commands in [-1, 1]

**Parameters:**
- `effort`: `(B, N)` or `(N,)` raw effort from `prior_effort()` or composed with residual

**Returns:**
- Motor commands in `[-1, 1]^N` ready for dynamics

### `prior_action(state: torch.Tensor, params: torch.Tensor) -> torch.Tensor`

Convenience: fully composed prior action (equivalent to `effort_to_action(prior_effort(...))`).

**Signature:**
```python
@torch.no_grad()
def prior_action(
    self,
    state: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
```

**Returns:**
- Motor commands in `[-1, 1]^N`, ready to feed to drone dynamics

**Usage:** For baseline (non-residual) rollouts.

### `default_init_params(mass: float | None = None, inertia: np.ndarray | None = None) -> np.ndarray`

Generate warm-started CMA-ES initial parameters matched to this morph's mass and inertia.

**Signature:**
```python
def default_init_params(
    self,
    *,
    mass: float | None = None,
    inertia: np.ndarray | None = None,
) -> np.ndarray:
```

**Behavior:**

1. **Trim:** Calls `_analytical_trim()` to compute per-motor offsets that null static roll/pitch moments from asymmetric arm lengths.
2. **Gains:** Hand-picked reference gains are **scaled by morph's mass/inertia** to maintain closed-loop bandwidth across morphologies:
   - `k_alt_p, k_alt_d` scale with mass
   - `k_tilt, k_rate` scale with Iyy (pitch/roll inertia)
   - `k_yaw_rate` scales with Izz (yaw inertia)

**Parameters:**
- `mass`: Float (kg). If provided, gains are scaled proportionally from REF_MASS. If `None`, all gains use reference values.
- `inertia`: `(3, 3)` array (kg·m²). If provided, rotational gains are scaled from REF_IYY / REF_IZZ.

**Returns:**
- NumPy array of shape `(N+5,)` ready to be fed into `ng.p.Array(init=...)`

**Example:**
```python
init_params = prior.default_init_params(mass=0.25, inertia=drone.inertia)
# Scales gains to compensate for 0.25 kg mass vs. 0.20 kg reference
```

## Helper Functions

### `tilt_mixer(propellers: Sequence[dict]) -> np.ndarray`

Generate per-motor mixer matrix for roll/pitch feedback.

**Returns:** `(N, 2)` array where column 0 = pitch contribution, column 1 = roll

**Sign convention:**
- **Roll:** `+sin(phi)` — right-side motor (+y) contributes positively to roll feedback
- **Pitch:** `-cos(phi)` — front motor (+x) contributes negatively to pitch feedback (the minus is critical; wrong sign causes positive feedback and CMA drives k_tilt→0)

**Verification:** Mixer signs are audited against `derive_reference_params` and `_dynamics_body` in `torch_drone_gate_env.py`. Test cases in `test_prior_controller.py` guard against regression.

### `yaw_mixer(propellers: Sequence[dict]) -> np.ndarray`

Generate per-motor mixer for body-yaw-rate feedback.

**Returns:** `(N,)` array

**Sign convention:** Bakes in `-spin` so learned `k_yaw_rate` stays positive. For CCW motors (spin=+1), the mixer is -1; for CW (spin=-1), it is +1.

### `compute_u_hover(params: dict, n_motors: int, gravity: float = 9.81) -> float`

Closed-form hover throttle matching 35c.

**Returns:** Motor command in `[-1, 1]` that, applied uniformly to all motors, exactly cancels gravity at hover.

**Used by:** `__init__` to precompute `self.u_hover`, and by `_analytical_trim()` to map thrust ratios back to action space.

## Notes

### Single Source of Truth

This module is the canonical implementation of the hover controller. All consumers import from here:
- `35c_hover_cmaes_minimal.py` — CMA-ES training
- `35d_replay_cmaes_minimal.py` — rollout replay
- `ResidualDroneEnv` — residual env wrapper
- Tests in `test_prior_controller.py` defend sign conventions for all consumers

### Mixer Sign Convention Audit

The pitch mixer sign (`-cos(phi)`) was corrected after the initial implementation showed positive feedback (CMA drove `k_tilt → 0`). The fix was verified end-to-end:
- Pre-fix: `k_tilt ≈ 0`, hover unstable
- Post-fix: `k_tilt ≈ +1.1`, hover time tripled

All sign tests in `test_prior_controller.py` guard against regression.

### Motor Initialization in ResidualDroneEnv

The residual env uses `_w_hover_norm` (hover-equivalent normalized motor speed) at reset instead of `w=0`. Without this, high-TWR morphs (prop-6/7) crash instantly at t=0 because `w=0` means mid-throttle, not zero thrust. This single fix increased library pass rate from ~40% to ~100%.

### Trimming and Analytical Solution

The `_analytical_trim()` method solves a 3×N least-norm linear system to null roll and pitch moments from asymmetric arm magnitudes (sampler jitter ±30% per motor). The solution is mapped back through the throttle curve to get per-motor trim offsets in action space.

Without analytical trim, high-thrust low-inertia morphs flip in <0.1s before CMA can refine gains.

### TWR Auto-Scaling

High-TWR morphs (large prop, light frame) tend to overshoot with the reference `action_scale=0.4`. The constructor auto-reduces it via:

```python
action_scale *= (TWR_REF / twr) ** 2  # TWR_REF = 32.0
```

This prevents control authority from exceeding the morphology's inertial damping capability.

## Relationship to [[Residual_Policy_Learning]]

`HoverPrior` is the base policy π₀ in the residual RL formulation:

```
a_t = prior_action(s_t) + α · π_θ(s_t)
```

or at the effort level:

```
effort = prior_effort(s_t) + α · residual_action
a_t = effort_to_action(effort)
```

The decomposition into `prior_effort` and `effort_to_action` enables clean composition with a learned residual without touching the clamp/scale logic.

## See Also

- [[ResidualDroneEnv]] — wrapper that uses this prior
- [[Residual_Policy_Learning]] — the conceptual framework
- `35c_hover_cmaes_minimal.py` — CMA-ES tuner that optimizes cmaes_params
- `test_prior_controller.py` — unit tests verifying mixer signs and shapes
