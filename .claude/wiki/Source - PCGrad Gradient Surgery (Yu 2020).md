---
type: source_summary
tags: [source, algorithm, multitask, reinforcement-learning, optimization, gradient]
source: https://arxiv.org/abs/2001.06782
author: Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, Chelsea Finn
date_ingested: 2026-07-09
---

# Source - PCGrad Gradient Surgery (Yu 2020)

**"Gradient Surgery for Multi-Task Learning"** — Yu, Kumar, Gupta, Levine, Hausman, Finn. NeurIPS 2020. arXiv:2001.06782. Code: https://github.com/tianheyu927/PCGrad

Introduces PCGrad: a model-agnostic gradient modification that projects each task's gradient onto the normal plane of any conflicting task gradient before the shared-parameter update. Zero new hyperparameters; inherits the base optimizer's settings. Identifies the "tragic triad" (conflicting gradients, magnitude disparity, high curvature) as the root causes of gradient interference in multi-task learning. Achieves 100% success on MT10 (10-task robotic manipulation) and >30% absolute improvement on MT50 vs. SAC alone. Relevant to ariel's 5-task MTRL residual controller (`37_train_residual_mtrl.py`) where hover's dense continuous reward conflicts with trajectory tasks' sparse gate-spike rewards.

## Entity Pages Created

- [[PCGrad_Gradient_Surgery]] — full algorithm reference: projection equation, Algorithm 1 pseudocode, convergence theorems, all result tables (CIFAR-100, CelebA, NYUv2, MT10, MT50), SB3 integration notes, ariel-specific diagnostic guidance.
- [[multitask_gradient_interference]] — concept reference: tragic triad taxonomy, ariel task gradient characterisation table, diagnostic checklist, remedies ranked by implementation cost.
