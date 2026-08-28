# MuJoCo Spot RL

Deep RL locomotion for Boston Dynamics Spot in MuJoCo.

## Pipelines

| Pipeline | Physics | Trainer | Best for |
|----------|---------|---------|----------|
| Classic | MuJoCo CPU | Stable-Baselines3 PPO | Debugging, CPU-only |
| **MJX** | MuJoCo MJX (JAX/GPU) | Brax PPO | Fast walk + run training |

Use the MJX pipeline for serious training (thousands of parallel envs on GPU).

## Setup

```bash
source /home/yunus/mujoco_rl_ws/mjenv/bin/activate
pip install -r requirements.txt
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # important on 4 GB GPUs
```

## MJX training (walk → run)

Benchmark throughput / lock in `timestep` & `frame_skip`:

```bash
python scripts/bench_mjx.py
```

Default locked config on GTX 1650 Ti: `timestep=0.004`, `frame_skip=5`, `num_envs=4096` (~9k env-steps/s).

**Stage 1 — walk** (`vx` in [0, 1.0]):

```bash
python scripts/train_ppo_mjx.py --stage walk --timesteps 20000000
```

**Stage 2 — run** (resume walk checkpoint, `vx` in [-0.5, 2.0]):

```bash
python scripts/train_ppo_mjx.py --stage run --timesteps 60000000 \
  --resume models/ppo_spot_mjx_walk_ckpt
```

Smoke test (plumbing + PPO metrics):

```bash
python scripts/train_ppo_mjx.py --smoke --stage walk
```

Outputs:
- `models/ppo_spot_mjx_{stage}.pkl` — NumPy policy for instant playback
- `models/ppo_spot_mjx_{stage}_ckpt/` — Brax checkpoints (for `--resume`)

## MJX playback (no JAX physics)

```bash
python scripts/play_ppo_mjx.py --stage walk --vx 0.6
python scripts/play_ppo_mjx.py --stage run --vx 1.5
```

Uses classic `mujoco.viewer` + NumPy MLP (no JIT stall).

## Classic CPU pipeline

```bash
python scripts/train_ppo.py
python scripts/play_ppo.py
```

## Notes

- MJX applies Euler + pyramidal cone + foot `condim=3` at load time (shared by train and play) so there is no sim-to-sim gap.
- Domain randomization (friction / mass / damping) is on by default; disable with `--no-domain-rand`.
- Leg joints live at `qpos[9:21]` (laser joints occupy `qpos[7:9]`); the MJX env indexes them by name.
