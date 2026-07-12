---
type: source_summary
tags: [source, algorithm, drone, control, geometric-control, se3, classical-control]
source: https://arxiv.org/abs/1411.2986
author: Farhad A. Goodarzi, Daewon Lee, Taeyoung Lee
date_ingested: 2026-07-09
---

# Source - Lee Geometric Adaptive Control SE3 (Goodarzi 2014)

**"Geometric Adaptive Tracking Control of a Quadrotor UAV on SE(3) for Agile Maneuvers"** — Goodarzi, Lee, Lee. arXiv:1411.2986, 2014.

Adaptive extension of the foundational Lee geometric controller (Lee, Leok, McClamroch, CDC 2010). Works directly on the SE(3) manifold to avoid gimbal lock and quaternion unwinding. Adds online parameter adaptation for unstructured disturbances via projection-based adaptive laws. Validated on 360° flip and Lissajous trajectory. Implemented in ariel as `LeeGeometricControl` in `src/ariel/simulation/drone/controllers/lee_control/lee_controller.py`; used in `14_mujoco_lee_figure8.py` (NED mode only — ENU branch has a sign bug). Serves as the classical-control upper bound for evaluating ariel's prior+residual generalist controller on trajectory tasks.

## Entity Pages Created

- [[Lee_Geometric_Control_SE3]] — full algorithm reference: SE(3) equations of motion, all error definitions (Ψ, e_R, e_Ω, e_x, e_v), complete thrust and moment control laws, adaptive laws with projection, ariel gain table with auto_scale_gains derivation, convergence propositions, NED/ENU bug note, classical vs. RL comparison table, and usage code from `14_mujoco_lee_figure8.py`.
