---
type: source_summary
tags: [source, ariel, residual-policy]
source: examples/spear/library/envs/residual_drone_env.py + examples/spear/library/prior_controller.py
author: Ariel project
date_ingested: 2026-07-10
---

# Source - Residual Drone Environment (ariel)

Documentation of the Stage-2/3 residual learning environment and analytical hover prior for morphology-conditional multi-task drone control in ariel's SPEAR pipeline.

## Entity Pages Created

- [[ResidualDroneEnv]] — The Gymnasium-compatible wrapper that combines analytical prior with learned residual. Extends `TorchDroneGateEnv` to support 5 tasks (hover + 4 trajectory), per-task α scaling, per-task prior gain scaling, and critic-only prior trustworthiness signals.

- [[HoverPrior]] — The analytical PD + yaw-damping controller. Provides `prior_effort()` and `prior_action()` methods for composing with residual actions. Implements closed-form hover control with morphology-conditional gain warm-starts and analytical trim for null static moments.

## Context

These pages document the implementation of [[Residual_Policy_Learning]] as applied to morphology-conditional drone control. The residual env acts as the Stage-2 wrapper (transforms residual actions into total actions via the prior), and the analytical prior is the fixed base controller that enables sample-efficient learning via the residual policy.

Key design features:
- Per-task residual scaling (hover α=0.10, trajectory α=0.40) to prevent prior-residual conflicts
- Per-task prior gain scaling to weaken attitude leveling on trajectory tasks (k_tilt=0.3) so residual can bank freely
- Critic-only prior descriptor (cmaes_params + median_score) to provide trustworthiness signal without letting actor learn to undo prior
- Morphology conditioning via 22-d hand-crafted features so one policy covers 100+ morphologies
- Analytical trim and gain scaling to handle 4.5× mass variation, 2.9× arm-length variation within the library

Used by:
- `37_train_residual_mtrl.py` — Stage-3 multi-task PPO trainer
- `39_visualize_residual_policy.py` — replay + visualization
- `36_build_hover_library.py` → via intermediate CMA tuning in `35c_hover_cmaes_minimal.py`
