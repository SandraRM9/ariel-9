---
type: source_summary
tags: [source, algorithm, reinforcement-learning, morphology, drone, adaptation]
source: https://arxiv.org/abs/2209.09232
author: Dingqi Zhang, Antonio Loquercio, Xiangyu Wu, Ashish Kumar, Jitendra Malik, Mark W. Mueller
date_ingested: 2026-07-09
---

# Source - Single Controller Quadcopter (Zhang 2023)

**"Learning a Single Near-Hover Position Controller for Vastly Different Quadcopters"** — Zhang et al. arXiv:2209.09232, ICRA 2023.

RMA-adapted two-phase RL framework (teacher-student) training a single policy to hover quadcopters with 4.5× mass and 2.9× arm-length variation — zero-shot, no hardware calibration. Phase 1: PPO with privileged environment parameters (18-d) compressed to 6-d latent. Phase 2: supervised 1D-CNN adaptation module predicts latent from 400-step state-action history. Closest published baseline to ariel's generalist hex controller; key difference is feed-forward morph_features (ariel) vs. online adaptation (this paper).

## Entity Pages Created

- [[single_controller_quadcopter]] — full algorithm reference: two-phase formulation, exact obs/action/env-param spaces with ranges, reward coefficients, network architectures, hyperparameters, result tables, ariel comparison table, and fallback design for online adaptation if morph_features prove insufficient.
- [[morphology_conditioned_control]] — concept reference covering all three families of morphology-conditioned control (feed-forward encoding, online RMA-style adaptation, prior+residual) with ariel's placement within that taxonomy.
