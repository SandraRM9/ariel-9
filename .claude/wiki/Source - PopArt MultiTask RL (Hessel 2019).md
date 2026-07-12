---
type: source_summary
tags: [source, algorithm, reinforcement-learning, multitask, normalization, value-function]
source: https://arxiv.org/abs/1809.04474
author: Matteo Hessel, Hubert Soyer, Lasse Espeholt, Wojciech Czarnecki, Simon Schmitt, Hado van Hasselt
date_ingested: 2026-07-09
---

# Source - PopArt MultiTask RL (Hessel 2019)

**"Multi-task Deep Reinforcement Learning with PopArt"** — Hessel et al. AAAI 2019. arXiv:1809.04474.

Introduces PopArt (POP = Preserve Outputs Precisely + ART = Adaptive Rescaling of Targets): per-task value-head normalisation for multi-task actor-critic RL. Maintains per-task running statistics (μ, σ) via EMA; normalises value targets to unit scale; applies an output-preserving weight correction to the value head's last layer after each statistics update so the network's unnormalised outputs don't jump. First agent to exceed human median on all 57 Atari games (110.7% normalised). Directly relevant to ariel's `_PerTaskRewardNormalizer` in `37_train_residual_mtrl.py` — ariel implements ART (target normalisation) but not POP (weight correction); this page documents when and how to add POP.

## Entity Pages Created

- [[PopArt_MultiTask_RL]] — full algorithm reference: POP and ART equations, multi-task extension, per-update procedure, all result tables (Atari-57, DmLab-30, ablation), detailed comparison table of ariel's current `_PerTaskRewardNormalizer` vs. full PopArt, and SB3 implementation path for adding the POP weight correction.
