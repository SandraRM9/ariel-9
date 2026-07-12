---
type: source_summary
tags: [source, ariel, environment, reinforcement-learning]
source: src/ariel/simulation/tasks/torch_drone_gate_env.py
author: Ariel project
date_ingested: 2026-07-10
---

# Source - TorchDroneGateEnv (ariel)

Documentation of the GPU-accelerated gate-tracking environment used in Stage 2/3 of the SPEAR residual learning pipeline.

## Entity Pages Created

- [[TorchDroneGateEnv]] — The VecEnv class providing GPU-accelerated drone simulation for multi-gate trajectories. Documents the full reward structure (distance + rate damping + upright bonus + velocity reward + altitude floor penalty + gate crossing bonus), episode termination conditions (max steps, out-of-bounds, divergence, tilt limit, collision), gate crossing detection logic (plane crossing + box containment), and the PyTorch dynamics engine (motor model, forces, moments, Euler integration).

## Context

TorchDroneGateEnv is the base environment wrapped by [[ResidualDroneEnv]] for Stage 3 MTRL training. The reward structure is critical for understanding the gradient landscape:

- **Distance reward** (`old_dist - new_dist`): Telescoping, favours progress not speed
- **Upright bonus** (default 0.0, configurable): Can dominate trajectory learning if set too high
- **Velocity reward** (default 0.0, configurable): Must be enabled to incentivise forward speed toward gates
- **Gate-crossing bonus** (sparse event): `+gate_reward` per crossing, `+10.0` for final gate

The user's observation (zero gates passed with +20 reward on figure8 despite hover-only prior) is explained by the upright bonus + velocity reward structure: the policy was paid for level flight and drift, not gate-seeking. The wiki documents exactly which parameters control this shaping.

Key gate-crossing mechanics:
- Requires both plane crossing (checked via projection onto gate normal) AND box containment (amax distance < gate_size/2)
- Default gate_size = 1.5 (diameter), so half-size = 0.75 ±
- If drone passes through plane but misses box, gate is not counted

Used by:
- `37_train_residual_mtrl.py` — via `MorphRotatingVecEnv` wrapper
- `39_visualize_residual_policy.py` — replay + visualization
- Single-morph CMA tuning in `35c_hover_cmaes_minimal.py`
