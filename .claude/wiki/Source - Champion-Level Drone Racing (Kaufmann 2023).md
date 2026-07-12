---
type: source_summary
tags: [source]
source: https://www.nature.com/articles/s41586-023-06419-4
author: Kaufmann, Bauersfeld, Loquercio, Müller, Koltun, Scaramuzza
date_ingested: 2026-07-09
---

# Source - Champion-Level Drone Racing (Kaufmann 2023)

Nature 2023 paper presenting Swift, the first autonomous system to beat human world-champion FPV drone pilots in head-to-head races. Combines PPO in simulation with empirical GP+kNN residual models for sim-to-real transfer. Highly relevant to ariel's racing drone task and the generalist hex controller project.

## Entity Pages Created

- [[Swift_Drone_Racing]] — full algorithm reference: PPO setup, quadrotor dynamics model, reward function (including perception-aware `r_perc` term), hardware specs, three-phase training pipeline, and race results
- [[empirical_sim_to_real]] — concept reference for the GP+kNN residual modeling approach; contrasts with domain randomization and links to ariel deployment path
