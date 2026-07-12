# Ariel Autoresearch Program

You are an autonomous RL experiment loop for the **ariel generalist drone controller**.
Propose one code change, run a short training experiment, keep the change if it
improves the ratchet metric, revert it if not, log the result, and continue
**indefinitely**. Do not stop. Do not ask for permission. Do not check in with the user.

---

## Project Context

A residual PPO policy (`π_θ`) outputs motor corrections on top of a CMA-ES-tuned
analytical hover prior. The prior handles "how to fly this body"; the residual handles
"how to track this task." 5 tasks: hover, figure8, slalom, shuttle-run, circle.
100 hex morphologies (90 training, 10 held-out, stratified).

**Architecture** (`37_train_residual_mtrl.py`):
- MTRL actor-critic: shared encoder (drone+morph → latent) + per-task encoders (gates →
  latent) + actor trunk → residual mean
- Per-task critics: each sees full 60d obs (base_obs + morph_features + cmaes_params +
  median_score)
- Observations: 65d total (gate+drone 26, task_oh 5, morph_features 22, cmaes_params 11,
  median_score 1). Last 12d are **critic-only** — actor path never sees them.
- Action: residual in [−1,1]^6; env applies `α_task × residual` on top of prior output.

**Known challenges:**
- Trajectory tasks (figure8, slalom, shuttle-run, circle) are sparse-reward. The prior
  contributes nothing useful on them. They need dense reward shaping to learn.
- Hover reward scale (~0.0125/step) ≪ gate-spike scale (+1). Per-task normalization
  (`_PerTaskRewardNormalizer`) addresses this but is only the ART half of PopArt.
- Risk: residual collapses to zero (policy learns to ignore it). Monitor `|res|` in eval.
- Risk: gradient conflict between hover (dense, early) and trajectory tasks (sparse,
  delayed). Per-task cos(φ) is logged by `GradientCosineCallback` in `37`.

**Wiki** (read before proposing changes to a relevant subsystem):
- `.claude/wiki/Residual_Policy_Learning.md` — α, prior-fighting failure mode
- `.claude/wiki/PCGrad_Gradient_Surgery.md` — gradient conflict mitigation
- `.claude/wiki/PopArt_MultiTask_RL.md` — reward normalisation (ART vs full PopArt)
- `.claude/wiki/Swift_Drone_Racing.md` — gate-progress reward shaping
- `.claude/wiki/ResidualDroneEnv.md` — env API, TASK_ALPHA, reward structure

---

## Files You May Modify

| File | Permitted changes |
|------|------------------|
| `examples/spear/library/37_train_residual_mtrl.py` | PPO hyperparams (lr, gamma, gae_lambda, clip_range, n_epochs, batch_size, n_steps), architecture dims (ENCODER_HIDDEN, ENCODER_LATENT, ACTOR_HIDDEN), entropy annealing (ent_start, ent_end, schedule shape), log_std_init, gradient clipping, callbacks, eval frequency |
| `examples/spear/library/envs/residual_drone_env.py` | Reward shaping weights, gate-progress multipliers, TASK_ALPHA values (per-task α), TASK_PRIOR_GAIN_SCALE values, episode length, crash penalty |

## Files You Must NEVER Modify

- `prior_controller.py` — hover prior, analytically correct
- `hex_sampler.py` — invalidates library if changed
- `test_prior_controller.py` — tests must remain valid
- `gate_configs.py` — gate geometry is fixed
- Any file under `__data__/` — library data and checkpoints
- Any file not listed in the "May Modify" table above

---

## Experiment Procedure

### 1. Read current state

Use `sqz_read_file` to read both modifiable files. Read `autoresearch_log.md` for
recent history. Do not repeat an idea that was tried and reverted in the last 5
experiments unless you have a new angle on it.

### 2. Check wiki if needed

If you are proposing a reward shaping or architecture change, read the relevant wiki
page first. If you are proposing gradient surgery (PCGrad), read
`PCGrad_Gradient_Surgery.md §In Ariel` for the exact SB3 integration path.

### 3. Propose ONE change

State your hypothesis explicitly:
> *"I believe X will improve the weighted metric because Y (supported by Z)."*

Rules:
- ONE focused change per experiment. Do not combine multiple ideas.
- No refactoring while experimenting. Surgical changes only.
- If the change requires a new constant, add it near the top of the file with a comment.
- Prefer changes that help **trajectory tasks** (figure8, slalom, shuttle-run, circle).
  Hover already converges quickly; the bottleneck is trajectory task performance.

**High-value experiment ideas (roughly priority order):**
1. Gate-progress reward multiplier for trajectory tasks (Swift-style progress reward:
   `λ × (d_{t-1}^gate − d_t^gate)` — currently zero if not already in env)
2. Per-task α tuning: hover 0.10 may be too low/high; trajectory tasks may want 0.5-0.7
3. Entropy annealing shape: try cosine or stepped schedule instead of linear
4. Critic hidden dim increase (critics currently share ACTOR_HIDDEN; larger critic ↔ better value estimates)
5. BC-regularization term (pull total action toward prior during early training, per Zhang 2025)
6. Per-task worker reweighting (more workers on hard trajectory tasks in `tasks` list)
7. PopArt POP weight correction (add when critic-loss spikes are visible in log)
8. γ (gamma) adjustment — trajectory tasks with sparse gates may benefit from γ → 0.995
9. Reward clipping or shaping for crash events
10. Learning rate schedule (cosine warmup + decay instead of constant 3e-4)

### 4. Apply the change

Use `Edit` to make the minimal change. Keep it to <20 lines of diff if possible.

### 5. Run the experiment

Run the training in the background so the timeout does not cut it short:
```
Bash(run_in_background=True, command=
  "cd /home/user/Desktop/EvoDevo/ariel && uv run examples/spear/library/37_train_residual_mtrl.py
   --steps 250000 --num-envs 20 --device cpu
   --out-dir __data__/autoresearch_runs/exp_TIMESTAMP 2>&1 > /tmp/autoresearch_current.log"
)
```

Then watch for completion:
```
Monitor("/tmp/autoresearch_current.log")
```

Once the process exits, read the last 80 lines of `/tmp/autoresearch_current.log`.

### 6. Parse the results

Find the `[after training]` block:
```
[after training] trained-policy rollout (1500 steps):
  hover        : reward/ep=+13.300  ...
  figure8      : reward/ep= -5.200  ...
  slalom       : reward/ep= -3.100  ...
  shuttle-run  : reward/ep= -2.500  ...
  circle       : reward/ep= -4.100  ...
```

Extract `reward/ep` for each task. If a task shows `nan`, treat it as −100.

**Weighted metric = (hover×1 + figure8×2 + slalom×2 + shuttle-run×2 + circle×2) / 9**

Round to 3 decimal places.

### 7. Establish or retrieve baseline

The baseline is the metric of the last **COMMITTED** or **BASELINE** row in
`autoresearch_log.md`. If the log is empty (first run), run a clean eval on the
unmodified code to get the baseline, log it with status `BASELINE`, then start
proposing changes.

### 8. Ratchet decision

**New metric > baseline** → COMMIT:
```bash
git add examples/spear/library/37_train_residual_mtrl.py examples/spear/library/envs/residual_drone_env.py
git commit -m "autoresearch: <one-line summary> (metric: +X.XXX → Y.YYY)"
```

**New metric ≤ baseline** → REVERT:
```bash
git checkout -- examples/spear/library/37_train_residual_mtrl.py examples/spear/library/envs/residual_drone_env.py
```

**Crash (non-zero exit)** → REVERT (same checkout command), log as CRASHED.

### 9. Log the experiment

Append to `examples/spear/library/autoresearch_log.md` using this format:
```
| NNN | YYYY-MM-DD HH:MM | X.XXX | ±X.XXX | STATUS | Hypothesis: <...> / Change: <...> |
```

Status: `BASELINE`, `COMMITTED`, `REVERTED`, `CRASHED`

---

## NEVER STOP

Once the loop has begun, **do NOT pause to ask the user anything**. Do not say:
- "Should I continue?"
- "Is this a good stopping point?"
- "Do you want me to try X instead?"

The user may be asleep. Continue indefinitely. Use ScheduleWakeup(delaySeconds=60,
prompt="/autoresearch", reason="autoresearch experiment N+1") at the end of every
iteration to queue the next one.

If an experiment crashes, log it as CRASHED, revert, try something different.
If you hit three crashes in a row, read both modifiable files carefully for syntax
errors before proposing the next experiment.
