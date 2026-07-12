---
type: source_summary
tags: [source, algorithm, reinforcement-learning, residual-policy, real-robot, sac]
source: https://arxiv.org/abs/1812.03201
author: Tobias Johannink, Shikhar Bahl, Ashvin Nair, Jianlan Luo, Avinash Kumar, Matthias Loskyll, Juan Aparicio Ojea, Eugen Solowjow, Sergey Levine
date_ingested: 2026-07-09
---

# Source - Residual RL for Robot Control (Johannink 2019)

**"Residual Reinforcement Learning for Robot Control"** — Johannink et al. arXiv:1812.03201, ICRA 2019.

Concurrent and independent formulation of residual RL (same concept as Silver et al. 1812.06298), validated on a real Kuka IIWA arm performing precision block insertion. Uses SAC (off-policy) rather than PPO; operates in torque space with explicit safety clipping. Key contribution over Silver et al.: real-hardware validation and the concrete finding that RL alone fails on contact-rich tasks but the base-controller + residual combination succeeds.

## Entity Pages Created

- [[Residual_Policy_Learning]] — updated: appended `## From: Johannink et al. 2019` subsection with SAC hyperparameters, torque-space formulation, empirical findings, and ariel-relevance notes.
