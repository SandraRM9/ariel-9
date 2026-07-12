---
type: ariel_reference
tags: [ariel, drone, environment, reinforcement-learning, pytorch, vecenv]
source: ariel project source code
date_ingested: 2026-07-10
---

# TorchDroneGateEnv

GPU-accelerated Gymnasium-compatible VecEnv for multi-gate drone trajectory tasks with hand-written PyTorch dynamics.

## Location

`src/ariel/simulation/tasks/torch_drone_gate_env.py`

## Purpose

Drop-in replacement for `DroneGateEnv` (NumPy/SymPy-based) with all simulation on device (CPU or CUDA). Implements a gymnasium VecEnv supporting:
- Multi-agent parallel simulation (batch dimension `E`)
- Variable morphologies (N motors)
- Gate-passing reward structure
- Customizable reward shaping (upright bonus, velocity reward, altitude floor penalty)
- Obstacle avoidance (cylindrical obstacles + ray-casting)
- NED frame dynamics with Euler angles (ZYX intrinsic)

## Signature

```python
class TorchDroneGateEnv(VecEnv):
    def __init__(
        self,
        num_envs: int,
        propellers=None,
        individual=None,
        gates_pos=None,
        gate_yaw=None,
        start_pos=None,
        x_bounds=(-5, 5),
        y_bounds=(-5, 5),
        z_bounds=(-5, 5),
        gates_ahead: int = 2,
        motor_limit: float = 1.0,
        initialize_at_random_gates: bool = True,
        seed=None,
        render_mode=None,
        device="cpu",
        dt: float = 0.01,
        max_steps: int = 1200,
        action_filter_alpha: float = 1.0,
        gate_reward: float = 1.0,
        pause_if_collision: bool = False,
        num_state_history: int = 0,
        num_action_history: int = 0,
        history_step_size: int = 1,
        upright_bonus: float = 0.0,
        tilt_terminate_cos: float = 0.0,
        extra_yaw_rate_pen: float = 0.0,
        obstacle_cyl_pos: np.ndarray | None = None,
        obstacle_cyl_r: np.ndarray | None = None,
        num_rays: int = 0,
        ray_max_range: float = 5.0,
        body_radius: float = 0.2,
        collision_penalty: float = 10.0,
        clearance_pen_coef: float = 0.0,
        velocity_reward_coef: float = 0.0,
        altitude_floor_z: float = -0.5,
        altitude_floor_coef: float = 0.0,
    ) -> None:
        ...
```

## Constructor Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `num_envs` | `int` | — | Batch size (parallel environments) |
| `propellers` | `Sequence[dict]` | `None` | List of propeller dicts (loc, dir, propsize). If `None` and `individual` is also `None`, uses standard quad. |
| `individual` | object | `None` | Ariel Individual genotype (alternative to `propellers`); converted via `DroneGateEnv._convert_individual_to_propellers()`. |
| `gates_pos` | `np.ndarray` | `None` | Gate positions (G, 3) in NED. If `None`, uses default 8-gate figure-8 track. |
| `gate_yaw` | `np.ndarray` | `None` | Gate yaw angles (G,) in radians. If `None`, uses default. |
| `start_pos` | `np.ndarray` | `None` | Starting position (3,) in NED. If `None`, derives from first gate. |
| `x_bounds`, `y_bounds`, `z_bounds` | tuple[float, float] | `(-5, 5)` | World bounds for out-of-bounds termination. |
| `gates_ahead` | `int` | `2` | Number of future gate offsets to include in observation (for lookahead). |
| `motor_limit` | `float` | `1.0` | Action clipping: final action in `[-1, 2*motor_limit - 1]`. Default `1.0` → `[-1, 1]`. |
| `initialize_at_random_gates` | `bool` | `True` | If `True`, reset at random gates with random velocities. If `False`, always reset at `start_pos`. |
| `seed` | `int \| None` | `None` | Random seed for reproducibility. |
| `device` | `str` | `"cpu"` | PyTorch device (`"cpu"`, `"cuda:0"`, etc.). |
| `dt` | `float` | `0.01` | Simulation timestep in seconds. |
| `max_steps` | `int` | `1200` | Episode length before auto-termination. |
| `action_filter_alpha` | `float` | `1.0` | Action smoothing (IIR filter): `a_filt = α·a + (1-α)·a_filt_prev`. `1.0` = no filtering. |
| `gate_reward` | `float` | `1.0` | Reward per gate passed (gate-crossing bonus). |
| `upright_bonus` | `float` | `0.0` | Per-step bonus: `+α · cos(roll)·cos(pitch)` clamped ≥ 0. Incentivises staying level. |
| `tilt_terminate_cos` | `float` | `0.0` | If > 0, terminate episode when `cos(roll)·cos(pitch) < threshold` (tilt too extreme). Disabled if `0.0`. |
| `extra_yaw_rate_pen` | `float` | `0.0` | Extra penalty: `-α · |ω_z|` on top of legacy `0.001 · ‖ω‖` rate penalty. |
| `velocity_reward_coef` | `float` | `0.0` | Per-step velocity reward: `+α · max(0, v·ê_toward_gate)`. Incentivises forward speed toward active gate. |
| `altitude_floor_z` | `float` | `-0.5` | Floor altitude in NED (higher = closer to ground). Used only if `altitude_floor_coef > 0`. |
| `altitude_floor_coef` | `float` | `0.0` | If > 0, apply soft penalty: `-α · max(0, z - floor_z)`. Encourages altitude above floor. |
| `obstacle_cyl_pos` | `np.ndarray \| None` | `None` | Obstacle cylinder xy positions (C, 2) in NED. |
| `obstacle_cyl_r` | `np.ndarray \| None` | `None` | Obstacle cylinder radii (C,). Must match `obstacle_cyl_pos` length. |
| `num_rays` | `int` | `0` | Number of ray-cast distance sensors (body-frame, evenly spaced over 2π). Appended to observation. |
| `ray_max_range` | `float` | `5.0` | Max range for ray-cast distances (m). |
| `body_radius` | `float` | `0.2` | Drone body radius for collision detection (m). |
| `collision_penalty` | `float` | `10.0` | Reward penalty for collision: `-collision_penalty`. |
| `clearance_pen_coef` | `float` | `0.0` | Soft obstacle penalty (before contact): `-α · max(0, (body_r + 0.3 - clearance) / safety_dist)`. |

## State and Observation Layout

### World State (Internal, `world_states`)

```
state[0:3]   = pos_ned     (x, y, z; NED frame, z↓ positive = altitude)
state[3:6]   = vel_world   (vx, vy, vz; world frame)
state[6:9]   = euler       (phi=roll, theta=pitch, psi=yaw) ZYX intrinsic
state[9:12]  = body_rate   (p, q, r; body-frame angular velocity)
state[12:12+N] = motor_w   (normalised motor speeds ∈ [-1, 1])
# total: (12+N, E) where N = num_motors, E = num_envs
```

### Observation (Gate-Relative, `_obs_t`)

```
obs[0:3]      = pos_rel (xyz relative to active gate, rotated into gate frame)
obs[3:6]      = vel_rel (xyz velocity relative to gate frame)
obs[6:7]      = roll    (absolute roll angle)
obs[7:8]      = pitch   (absolute pitch angle)
obs[8:9]      = yaw_rel (relative yaw: world_yaw - gate_yaw)
obs[9:12]     = body_rates (p, q, r; absolute)
obs[12:12+N]  = motor_w (normalised motor speeds)
obs[12+N : 12+N+4*gates_ahead] = future_gate_offsets
                # Each future gate i contributes 4 dims:
                # [dpos_rel(3), dyaw_rel(1)] relative to prior gate
obs[...+4*gates_ahead : ...+4*gates_ahead+num_rays] = ray_distances
                # Per-motor ray-cast distance sensors if num_rays > 0
# total: (state_len,) = 12 + N + 4*gates_ahead + num_rays
```

## Reward Structure

Per-step reward (all components sum):

```
r = r_distance + r_rate_pen + r_upright + r_velocity + r_altitude_floor + r_gate + r_oob + r_tilt + r_collision
```

### Component Breakdown

| Component | Formula | Activation |
|-----------|---------|------------|
| **Distance** | `(old_dist - new_dist)` | Always active |
| **Rate penalty** | `-0.001 · ‖ω‖` | Always active |
| **Extra yaw penalty** | `-extra_yaw_rate_pen · \|ω_z\|` | If `extra_yaw_rate_pen > 0` |
| **Upright bonus** | `+upright_bonus · max(0, cos(roll)·cos(pitch))` | If `upright_bonus > 0` |
| **Velocity reward** | `+velocity_reward_coef · max(0, v·ê_toward_gate)` | If `velocity_reward_coef > 0` |
| **Altitude floor** | `-altitude_floor_coef · max(0, z - altitude_floor_z)` | If `altitude_floor_coef > 0` |
| **Gate crossing** | `+gate_reward` per gate, `+10.0` for final gate | Per crossing (binary event) |
| **Out-of-bounds** | `-10.0` | If oob or diverged |
| **Tilt termination** | `-10.0` | If tilted (if `tilt_terminate_cos > 0` and exceeded) |
| **Collision** | `-collision_penalty` | If touched obstacle |

### Key Formulas (Per-Step)

**Distance reward:** Encourages closing distance to active gate
```python
r_dist = (pos_old - gate_pos).norm() - (pos_new - gate_pos).norm()
```

**Gate crossing:** Detects crossing the gate plane in the forward direction AND being inside the gate box
```python
proj_old = (pos_old - gate_pos) · gate_normal  # gate_normal = [cos(yaw), sin(yaw)]
proj_new = (pos_new - gate_pos) · gate_normal
crossed = (proj_old < 0) & (proj_new > 0)
in_gate = max(abs(dpos_xyz)) < gate_size / 2  # amax over xyz
gate_passed = crossed & in_gate
```

**Velocity reward:** Bonus for velocity component toward gate
```python
if velocity_reward_coef > 0:
    r_vel = velocity_reward_coef · max(0, v · (gate_pos - pos_new).normalize())
```

**Altitude floor:** Soft penalty below floor (not a hard wall)
```python
if altitude_floor_coef > 0:
    r_floor = -altitude_floor_coef · max(0, z - altitude_floor_z)
```

## Episode Termination Conditions

Episode ends (sets `done=True`) when any of:

| Condition | Penalty | Trigger |
|-----------|---------|---------|
| **Max steps reached** | None (truncation, not failure) | `step_count >= max_steps` |
| **Out-of-bounds** | `-10.0` | Position outside `(x_bounds, y_bounds, z_bounds)` |
| **Divergence** | `-10.0` | NaN/inf in state or `|state| > 1e6` |
| **Tilt limit** | `-10.0` | `cos(roll)·cos(pitch) < tilt_terminate_cos` (if enabled) |
| **Collision** | `-collision_penalty` | Drone xy within `body_radius` of obstacle center |

## Dynamics

The core dynamics are implemented in the module-level function `_dynamics_body()` and compiled via `torch.compile` on CUDA for performance. Per-step integration:

1. **Motor model:** Inverse of the commanded throttle curve (quadratic motor response)
2. **Forces:** Thrust (vertical), aerodynamic drag (body-frame velocity dependent)
3. **Moments:** Roll/pitch (per-motor thrust imbalance), yaw (motor spin and rate damping)
4. **Euler angle kinematics** (ZYX convention)
5. **Integration:** Forward Euler, `state_new = state_old + dt · state_dot`

## Usage Example

```python
from src.ariel.simulation.tasks.torch_drone_gate_env import TorchDroneGateEnv

# Create env with 20 parallel agents, with velocity + altitude shaping
env = TorchDroneGateEnv(
    num_envs=20,
    propellers=morph["propellers"],  # from library
    gates_pos=gates,
    gate_yaw=gate_yaws,
    device="cuda:0",
    dt=0.01,
    max_steps=1000,
    # Reward shaping for trajectory tasks:
    upright_bonus=0.01,           # encourage level flight
    velocity_reward_coef=0.005,   # encourage forward progress
    altitude_floor_z=-0.5,
    altitude_floor_coef=0.5,      # soft penalty for low altitude
)

obs = env.reset()  # (20, state_len) numpy array
for step in range(1000):
    actions = policy(obs)  # (20, N) actions
    env.step_async(actions)
    obs, rewards, dones, infos = env.step_wait()
    # rewards: (20,) per-step shaped reward
    # dones: (20,) episode termination flags
    # infos: list of 20 dicts with gate_passed, num_gates_passed, etc.
```

## Notes

### Reward Shaping Pitfalls

The distance reward `r_dist = old_dist - new_dist` is **telescoping**: it doesn't favour speed, only net progress. At hover or orbit equilibrium, the residual policy gets paid ~0 by this term alone. To make gate-seeking an attractive trajectory:

- **Add `velocity_reward_coef > 0`**: Explicitly rewards velocity component toward the gate (per `Swift_Drone_Racing.md` logic).
- **Avoid `upright_bonus` overkill**: A large bonus (~0.10+) can outweigh gate distance, making "level drifting" locally optimal (as observed in the user's 500k-step hover-only prior run).

The defaults are:
```python
upright_bonus=0.0, velocity_reward_coef=0.0
```
meaning no dense shaping by default — only distance + rate damping. This is conservative but makes gate-seeking gradient-sparse for long-horizon tasks.

### Gate Detection

Gate crossing requires TWO conditions:
1. **Crossing the plane:** The plane normal is `[cos(gate_yaw), sin(gate_yaw)]` (gate's forward direction).
2. **Inside the box:** All three dimensions (xyz) within `gate_size / 2` of the gate center.

A drone can pass through the plane but miss the gate (box too small, or drone trajectory tangential). Collision with gate geometry is **not** a failure — only missing the box is.

### Observation Wrapping

The observation is rotated into each gate's frame so the policy doesn't need to learn yaw-invariance. The rotation uses `cg·dxy[0] + sg·dxy[1]` where `cg, sg = cos(gate_yaw), sin(gate_yaw)`. This assumes the policy benefits from a stable "forward = x, right = y" frame. For some tasks (circles, figure-8s), learning yaw-invariance is essential — the observation doesn't make this automatic.

### Compiled Dynamics

On CUDA, `_dynamics_body` is compiled once per motor-count via `torch.compile(..., mode="reduce-overhead")`. This traces a single graph reused across all individuals with the same motor count, massively speeding batch evaluation. CPU uses uncompiled `_dynamics_body` (torch.compile overhead > benefit on CPU).

### Motor Normalization

Motor speeds `w` are stored normalised ∈ `[-1, 1]`, linearly mapped to physical rad/s:

```python
W = (w + 1.0) * (0.5 * W_RANGE) + w_min  # W_RANGE = 3000 rad/s (W_MAX_N - W_MIN_N)
```

The action `u ∈ [-1, 1]` commands a throttle that is inverted through the motor's quadratic throttle curve to get target speed `W_c`, then integrated via `dW = (W_c - W) / tau`.

## See Also

- [[ResidualDroneEnv]] — wrapper that applies prior + residual on top of this env
- [[HoverPrior]] — analytical controller used in residual learning
- `DroneGateEnv` (NumPy version) — the original, non-GPU implementation
- `DroneSimulator` — drone parameter and propeller management
