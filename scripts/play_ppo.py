#!/usr/bin/env python3
"""Eğitilmiş PPO modelini MuJoCo viewer'da izle."""

import time
from pathlib import Path
import sys

import mujoco.viewer
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.spot_walk_env import SpotWalkEnv

MODEL_PATH = ROOT / "models" / "ppo_spot_800000_steps.zip"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model yok: {MODEL_PATH}")

    env = SpotWalkEnv()
    model = PPO.load(str(MODEL_PATH))

    obs, _ = env.reset()
    print("Controls: left-drag rotate, right-drag pan, scroll zoom, Esc quit")
    print(f"Model: {MODEL_PATH}")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            # deterministic=True → en olası action (izlerken daha stabil)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            viewer.sync()
            time.sleep(0.02)

            if terminated or truncated:
                print(
                    f"episode end | reward={reward:.3f} "
                    f"height={info.get('height', float('nan')):.3f} "
                    f"forward={info.get('forward_vel', float('nan')):.3f}"
                )
                obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()