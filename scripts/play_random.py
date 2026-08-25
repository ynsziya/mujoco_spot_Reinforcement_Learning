#!/usr/bin/env python3
"""SpotWalkEnv'i rastgele action ile MuJoCo viewer'da izle."""

import time
from pathlib import Path
import sys

import mujoco
import mujoco.viewer
import numpy as np

# proje kökünü import path'e ekle
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.spot_walk_env import SpotWalkEnv


def main() -> None:
    env = SpotWalkEnv()
    env.reset()

    print("Controls: left-drag rotate, right-drag pan, scroll zoom, Esc quit")
    print("Robot rastgele action alıyor (Adım 2 kontrol testi).")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            action = env.action_space.sample()  # rastgele [-1, 1]^12
            env.step(action)
            viewer.sync()
            # ~50 Hz'e yakın izleme (frame_skip zaten fizikte var)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print("düştü, reset", info)
                env.reset()
            time.sleep(0.02)

    env.close()


if __name__ == "__main__":
    main()