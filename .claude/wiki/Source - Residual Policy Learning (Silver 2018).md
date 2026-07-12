---
type: source_summary
tags: [source, algorithm, reinforcement-learning, residual-policy]
source: https://arxiv.org/abs/1812.06298
author: Tom Silver, Kelsey Allen, Josh Tenenbaum, Leslie Kaelbling
date_ingested: 2026-07-09
---

# Source - Residual Policy Learning (Silver 2018)

**"Residual Policy Learning"** — Silver, Allen, Tenenbaum, Kaelbling. arXiv:1812.06298, 2018.

Introduces RPL: a simple technique for improving any existing (possibly non-differentiable) base controller by learning an additive RL residual on top of it, rather than training from scratch. Demonstrated on six long-horizon, sparse-reward MuJoCo manipulation tasks where pure RL fails. Directly underpins the `examples/spear/library/` architecture in ariel.

## Entity Pages Created

- [[Residual_Policy_Learning]] — algorithm reference covering the full method, equations, failure modes, ariel implementation mapping (prior controller, residual env, MTRL trainer, hex library), and practical notes for the spear generalist-controller project.
