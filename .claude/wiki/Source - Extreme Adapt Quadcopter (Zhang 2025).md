---
type: source_summary
tags: [source, algorithm, reinforcement-learning, morphology, drone, adaptation, imitation-learning]
source: https://hiperlab.berkeley.edu/wp-content/uploads/2024/09/2024_Zhang_ExtremeAdapt.pdf
author: Dingqi Zhang, Antonio Loquercio, Jerry Tang, Ting-Hao Wang, Jitendra Malik, Mark W. Mueller
date_ingested: 2026-07-09
---

# Source - Extreme Adapt Quadcopter (Zhang 2025)

**"A Learning-based Quadcopter Controller with Extreme Adaptation"** — Zhang et al. IEEE Transactions on Robotics, 2025. arXiv:2409.12949. Code: https://github.com/muellerlab/xadapt_ctrl

Extends [[single_controller_quadcopter]] (ICRA 2023) with three innovations: (1) dual BC+RL training — behavior cloning from a per-episode adaptive model-based expert with exponentially decaying weight; (2) torque-tracking reward rather than angular velocity tracking, providing denser gradient for 500 Hz low-level control; (3) design-informed domain randomization via a size factor c that correlates mass, inertia, and motor strength (mass ∝ l³, inertia ∝ l⁵, drag ∝ l²). Achieves 100% success rate on in-distribution trajectories vs 77% for best baseline, 84% success at 16× OOD extrapolation, and zero-shot real-robot deployment on platforms with 3.7× mass and 100× motor constant variation.

PDF directly extracted via pdftotext — full equations, tables, ablations, and architecture specs obtained verbatim.

## Entity Pages Created

- [[extreme_adapt_quadcopter]] — full algorithm reference with all equations (BC+RL objective, torque reward, size-factor randomization), complete domain randomization Table II with training and testing ranges, network architectures, all result tables, and ariel-specific adoption notes (BC from CMA prior, design-informed hex randomization, torque tracking, shorter history window).
- [[single_controller_quadcopter]] — updated: added §Successor Paper cross-reference.
- [[morphology_conditioned_control]] — updated: added entry for this paper under RMA-style examples.
