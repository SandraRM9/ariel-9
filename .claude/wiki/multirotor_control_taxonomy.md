---
type: concept_reference
tags: [concept, drone, taxonomy, control-abstraction, sim-to-real, rpm-control]
source: https://arxiv.org/abs/2311.13081
date_ingested: 2026-07-09
---

# multirotor_control_taxonomy

A 6-level hierarchy classifying multirotor control inputs from highest (position commands) to lowest (motor effort), making explicit which non-linearities and domain parameters enter at each level. Introduced by Eschmann et al. in [[Learning_to_Fly_in_Seconds]].

## Theory

The taxonomy is ordered by the *flat-state* derivative order of the commanded quantity, from the perspective of a position controller. Each sub-bullet is a non-linear transformation that maps between levels:

```
Level 0: Position commands
   ↓ (double integrator)
Level 1: Velocity commands
   ↓ (single integrator)
Level 2: Acceleration commands
   2.1 Attitude + thrust: non-linear orientation → acceleration transform
       [domain param: mass]
   ↓
Level 3: Jerk commands
   3.1 Angular rate + thrust (CTBR): rotational kinematics
       [domain params: mass, rotational kinematics — rigid body]
   ↓
Level 4: Snap commands
   4.1 Body torque + thrust: rotational dynamics
       [domain params: inertia tensor]
   4.2 Individual rotor thrusts (SRT): allocation geometry
       [domain params: vehicle geometry]
   4.3 RPMs: non-linear torque/thrust curves
       [domain params: thrust/torque model parameters]
   ↓
Level 5: Crackle commands
   5.1 Motor RPM setpoints: first-order motor dynamics
       [domain param: low-pass time constant τ]
   5.2 Motor effort: battery/ESC non-linearity
       [domain param: battery level]
```

**Key insight:** Reality gap and domain parameter sensitivity grow *superlinearly* toward lower levels. A Level 3.1 (CTBR) controller only needs to know mass; a Level 5.1 controller must model motor curves, geometry, inertia, and motor delay.

**Each lower level adds an integrator.** High-frequency exploration noise is suppressed through the chain of integrators, making lower-level RL training high-sample-complexity without careful curriculum design.

## Practical Notes

- **Level 3.1 (CTBR)** is the most common RL target in agile flight literature ([[Swift_Drone_Racing]], [[single_controller_quadcopter]], [[extreme_adapt_quadcopter]]). Relatively easy Sim2Real: only mass matters.
- **Level 4.2 (SRT)** is common in morphology-adaptive work (adds geometry). Requires mixer knowledge.
- **Level 5.1 (RPM)** — [[Learning_to_Fly_in_Seconds]] — faces full non-linear stack. Requires motor time constant identification and curriculum to train reliably.
- **Level 5.2** (motor effort / battery-aware) is largely unexplored in deep RL.
- Domain randomization burden is proportional to level: CTBR needs only mass randomization; RPM needs motor curves, geometry, inertia, delay — a much larger randomization space.

## In Ariel

Ariel's hex drone uses **Level 3.1 (CTBR)** internally — the `HoverPrior` and residual PPO policy output collective thrust + body rates, with Betaflight (or equivalent) handling the motor mixing below. This is the same level as [[Swift_Drone_Racing]].

If ariel ever targets Level 5.1 (direct motor commands without a firmware flight controller), [[Learning_to_Fly_in_Seconds]] provides the template: asymmetric actor-critic + curriculum + motor delay modelling.

## See Also

- [[Learning_to_Fly_in_Seconds]] — source paper defining the taxonomy
- [[Swift_Drone_Racing]] — Level 3.1 CTBR; uses Betaflight for sub-Level-3 control
- [[single_controller_quadcopter]] — Level 3.1 (high-level RL), Betaflight below
- [[Residual_Policy_Learning]] — ariel's current approach; operates at Level 3.1
