---
type: source_summary
tags: [source]
source: https://arxiv.org/abs/2311.13081
author: Eschmann, Albani, Loianno
date_ingested: 2026-07-09
---

# Source - Learning to Fly in Seconds (Eschmann 2024)

IEEE RA-L 2024 paper presenting a TD3 + asymmetric actor-critic + curriculum learning framework for direct RPM quadrotor control, training in 18 seconds on a laptop GPU with only 300k environment steps and no domain randomization. Relevant to ariel as a Sim2Real and low-level control reference; introduces a multirotor control abstraction taxonomy useful for comparing ariel's Level 3.1 approach against Level 5.1.

## Entity Pages Created

- [[Learning_to_Fly_in_Seconds]] — full algorithm reference: TD3 setup, asymmetric actor-critic design, reward function, curriculum learning procedure, observation spaces (actor 18+NH·4 vs critic 28-d), motor delay model, ablation results, trajectory tracking comparison vs. PID/geometric/INDI
- [[multirotor_control_taxonomy]] — concept reference for the 6-level hierarchy (position → motor effort) introduced in the paper; maps ariel and related papers to their respective control levels
