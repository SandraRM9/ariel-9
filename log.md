## [2026-07-10] Ingest | TorchDroneGateEnv (ariel)
- Files created: TorchDroneEnv.md, Source - TorchDroneGateEnv (ariel).md
- Files updated: (none)
- Model: Claude Haiku 4.5

## [2026-07-10] Ingest | Residual Drone Environment (ariel)
- Files created: ResidualDroneEnv.md, HoverPrior.md, Source - Residual Drone Environment (ariel).md
- Files updated: (none)
- Model: Claude Haiku 4.5

## [2026-07-09] Ingest | Learning to Fly in Seconds (Eschmann 2024)
- Files created: Learning_to_Fly_in_Seconds.md, multirotor_control_taxonomy.md, Source - Learning to Fly in Seconds (Eschmann 2024).md
- Files updated: (none)
- Model: Claude (subscription)

## [2026-07-09] Ingest | Champion-level drone racing using deep reinforcement learning
- Files created: Swift_Drone_Racing.md, empirical_sim_to_real.md, Source - Champion-Level Drone Racing (Kaufmann 2023).md
- Files updated: (none)
- Model: Claude (subscription)

## [2026-04-13] Ingest | MuJoCo Visualization API
- Files created: MjvPerturb.md, MjvGeom.md, MjvFigure.md, mujoco_visualization_functions.md, Source - MuJoCo Visualization API.md
- Files updated: MjvCamera.md, MjvOption.md, MjvScene.md, mujoco_enumerations.md
- Model: Claude (subscription)

## [2026-04-13] Ingest | MuJoCo Programming — Simulation Chapter
- Files created: mujoco_state_control.md, mujoco_jacobians.md, mujoco_contacts.md, mujoco_data_layout.md, mujoco_diagnostics.md, Source - MuJoCo Programming Simulation.md
- Files updated: mujoco_simulation_functions.md (added mj_forwardSkip, skip levels, multi-threading, mj_setConst)
- Model: Claude (subscription)

## [2026-04-13] Ingest | MuJoCo Code Samples
- Files created: mujoco_code_samples.md, Source - MuJoCo Code Samples.md
- Files updated: (none)
- Model: Claude (subscription)

## [2026-04-13] Ingest | MuJoCo MJX Documentation
- Files created: mjx_overview.md, mjx_core_functions.md, mjx_warp.md, mjx_performance.md, Source - MuJoCo MJX Documentation.md
- Files updated: (none)
- Model: Claude (subscription)

## [2026-06-24] Ingest | Nevergrad Optimization API (re-fetch)
- Files created: (none — already covered)
- Files updated: Source - Nevergrad Optimization API.md (date refreshed)
- Notes: Re-verified https://facebookresearch.github.io/nevergrad/optimization.html against existing pages; content unchanged.
- Model: Claude (subscription)

## [2026-06-24] Ingest | Nevergrad Parametrization
- Files created: Source - Nevergrad Parametrization.md
- Files updated: (none — Nevergrad_Parametrization.md already covers the page)
- Notes: Re-fetched https://facebookresearch.github.io/nevergrad/parametrization.html; no new API surface beyond existing wiki coverage. Recorded gaps (set_bounds method options, per-element sigma, custom mutation classes) for future source-code ingest.
- Model: Claude (subscription)

## [2026-06-24] Ingest | pycma CMAEvolutionStrategy API
- Files created: CMAEvolutionStrategy.md, Source - pycma CMAEvolutionStrategy API.md
- Files updated: (none)
- Model: Claude (subscription)

## [2026-06-24] Ingest | Nevergrad optimizerlib.py (ParametrizedCMA)
- Files created: ParametrizedCMA.md, Source - Nevergrad optimizerlib.md
- Files updated: Nevergrad_Optimizers.md
- Model: Claude (subscription)

## [2026-07-09] Ingest | Residual Policy Learning (Silver 2018)
- Files created: Residual_Policy_Learning.md, Source - Residual Policy Learning (Silver 2018).md
- Files updated: (none)
- Notes: arXiv:1812.06298. PDF not machine-readable via WebFetch; content synthesised from abstract + project landing page + ariel codebase cross-reference (prior_controller.py, residual_drone_env.py, 37_train_residual_mtrl.py). Ariel-specific extensions documented: per-task α, TASK_PRIOR_GAIN_SCALE, 11-d cmaes_params, 22-d morph_features, k-NN prior warm-start design.
- Model: Claude Sonnet 4.6

## [2026-07-09] Ingest | Residual Reinforcement Learning for Robot Control (Johannink 2019)
- Files created: Source - Residual RL for Robot Control (Johannink 2019).md
- Files updated: Residual_Policy_Learning.md (appended Johannink et al. section)
- Notes: arXiv:1812.03201 (ICRA 2019). PDF not machine-readable; content from abstract + prior knowledge. Concurrent companion to Silver 1812.06298 — same residual RL concept, real Kuka hardware, SAC instead of PPO, torque domain. SAC hyperparams, clipping design, and ariel relevance documented.
- Model: Claude Sonnet 4.6

## [2026-07-09] Ingest | Learning a Single Near-Hover Position Controller for Vastly Different Quadcopters (Zhang 2023)
- Files created: single_controller_quadcopter.md, morphology_conditioned_control.md, Source - Single Controller Quadcopter (Zhang 2023).md
- Files updated: (none)
- Notes: arXiv:2209.09232 (ICRA 2023). Full technical extraction via ar5iv: obs/action/env-param spaces, reward coefficients, network sizes, domain randomization ranges, two-phase training procedure, result tables. Closest published baseline to ariel's generalist hex controller. Includes ariel comparison table and fallback design (RMA adaptation module as alternative to morph_features).
- Model: Claude Sonnet 4.6

## [2026-07-09] Ingest | A Learning-based Quadcopter Controller with Extreme Adaptation (Zhang 2025)
- Files created: extreme_adapt_quadcopter.md, Source - Extreme Adapt Quadcopter (Zhang 2025).md
- Files updated: single_controller_quadcopter.md, morphology_conditioned_control.md
- Notes: hiperlab.berkeley.edu PDF (arXiv:2409.12949, IEEE T-RO 2025). PDF extracted with pdftotext — full equations, tables, ablations. Three key innovations over 2023 paper documented: BC+RL hybrid (decaying α), torque-tracking reward, design-informed domain randomization via size factor c. Full Table II (training + testing ranges), Table III (baselines), OOD generalisation results, network architectures, all hyperparameters. Ariel adoption notes added.
- Model: Claude Sonnet 4.6

## [2026-07-09] Ingest | Gradient Surgery for Multi-Task Learning / PCGrad (Yu 2020)
- Files created: PCGrad_Gradient_Surgery.md, multitask_gradient_interference.md, Source - PCGrad Gradient Surgery (Yu 2020).md
- Files updated: (none)
- Notes: arXiv:2001.06782 (NeurIPS 2020). Fetched via ar5iv — full equations, Algorithm 1 pseudocode, all three tragic-triad definitions, both theorems, all result tables (CIFAR-100, CelebA, NYUv2, MT10, MT50), ablation. Ariel integration notes: per-task gradient conflict diagnostic, SB3 PPO integration point, interaction with existing _PerTaskRewardNormalizer.
- Model: Claude Sonnet 4.6

## [2026-07-09] Ingest | Multi-task Deep Reinforcement Learning with PopArt (Hessel 2019)
- Files created: PopArt_MultiTask_RL.md, Source - PopArt MultiTask RL (Hessel 2019).md
- Files updated: multitask_gradient_interference.md (wikilink to PopArt)
- Notes: arXiv:1809.04474 (AAAI 2019). Fetched via ar5iv — full POP+ART equations, multi-task extension, IMPALA architecture, all hyperparameters (β=3e-4, σ bounds), Atari-57 and DmLab-30 result tables, ablation. Ariel-specific: comparison table of current _PerTaskRewardNormalizer vs. full PopArt, SB3 implementation path for adding POP weight correction.
- Model: Claude Sonnet 4.6

## [2026-07-09] Ingest | Geometric Adaptive Tracking Control of a Quadrotor UAV on SE(3) (Goodarzi 2014)
- Files created: Lee_Geometric_Control_SE3.md, Source - Lee Geometric Adaptive Control SE3 (Goodarzi 2014).md
- Files updated: (none)
- Notes: arXiv:1411.2986. Full equations from ar5iv: SE(3) dynamics, error functions (Ψ, e_R, e_Ω), thrust/moment control laws, adaptive laws with projection. Cross-referenced against ariel source (lee_controller.py, 14_mujoco_lee_figure8.py) — documented auto_scale_gains derivation (I·ω_n²), NED/ENU sign bug, kinematic playback workaround, and classical-vs-RL comparison table.
- Model: Claude Sonnet 4.6
