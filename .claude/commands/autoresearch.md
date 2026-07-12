You are running one iteration of the **ariel autoresearch loop** — autonomous RL
experimentation for the generalist drone controller. Execute exactly one
experiment, ratchet it, log it, and schedule the next iteration.

## Step 1 — Load state (read all four, in parallel)

1. `examples/spear/library/autoresearch_program.md` — full procedure and rules
2. `examples/spear/library/autoresearch_log.md` — experiment history (create header if missing)
3. `examples/spear/library/37_train_residual_mtrl.py` — current training script
4. `examples/spear/library/envs/residual_drone_env.py` — current env

Use `sqz_read_file` for the two Python files (they are large).

## Step 2 — Execute one experiment

Follow `autoresearch_program.md` exactly. The procedure is:

1. Identify the current baseline metric (last COMMITTED/BASELINE row in log).
   If log is empty, run a clean eval first to establish the baseline.
2. Propose ONE change with an explicit hypothesis.
3. Apply the change with `Edit`.
4. Run training in the background:
   ```
   Bash(run_in_background=True,
        command="cd /home/user/Desktop/EvoDevo/ariel && uv run examples/spear/library/37_train_residual_mtrl.py --steps 250000 --num-envs 20 --device cpu 2>&1 | tee /tmp/autoresearch_current.log")
   ```
   Then monitor: `Monitor("/tmp/autoresearch_current.log")` — wait for process exit.
5. Read the last 80 lines of `/tmp/autoresearch_current.log`.
6. Parse `[after training]` per-task `reward/ep` values.
   Weighted metric = (hover×1 + figure8×2 + slalom×2 + shuttle-run×2 + circle×2) / 9.
   Treat `nan` as −100.
7. Compare to baseline. Commit if improved, else `git checkout --` the modified files.
8. Append the result row to `autoresearch_log.md`.

## Step 3 — Schedule next iteration

After the log is updated, call ScheduleWakeup:
- `delaySeconds: 60`
- `prompt: /autoresearch`
- `reason: autoresearch loop — queuing experiment N+1`

**Do NOT ask the user anything. Do NOT stop. The user may be away.**

$ARGUMENTS
