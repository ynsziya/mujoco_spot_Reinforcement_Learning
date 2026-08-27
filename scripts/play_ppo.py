#!/usr/bin/env python3
"""Eğitilmiş PPO modelini MuJoCo viewer'da izle."""

import time
from pathlib import Path
import sys

from gymnasium.wrappers import TimeLimit
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.spot_walk_env import SpotWalkEnv

MODEL_PATH = ROOT / "models" / "ppo_spot_final.zip"
VECNORM_PATH = ROOT / "models" / "ppo_spot_vecnormalize.pkl"


def make_view_env():
    env = SpotWalkEnv(randomize_command=False)
    env = TimeLimit(env, max_episode_steps=1000)
    return Monitor(env)


def unwrap_spot_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model yok: {MODEL_PATH}")

    vec_env = DummyVecEnv([make_view_env])
    if VECNORM_PATH.exists():
        vec_env = VecNormalize.load(str(VECNORM_PATH), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print(f"Uyarı: VecNormalize dosyası yok, ham gözlem kullanılacak: {VECNORM_PATH}")

    base_vec_env = vec_env.venv if hasattr(vec_env, "venv") else vec_env
    env = unwrap_spot_env(base_vec_env.envs[0])
    model = PPO.load(str(MODEL_PATH), env=vec_env)

    obs = vec_env.reset()
    print("Controls: left-drag rotate, right-drag pan, scroll zoom, Esc quit")
    print(f"Model: {MODEL_PATH}")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            # deterministic=True → en olası action (izlerken daha stabil)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = vec_env.step(action)
            info = infos[0]

            viewer.sync()
            time.sleep(0.02)

            if bool(done[0]):
                print(
                    f"episode end | reward={float(reward[0]):.3f} "
                    f"height={info.get('height', float('nan')):.3f} "
                    f"fwd={info.get('forward_vel', float('nan')):.3f} "
                    f"target={info.get('target_forward_vel', float('nan')):.3f} "
                    f"lat={info.get('lateral_pos', float('nan')):.3f} "
                    f"yaw={info.get('yaw', float('nan')):.3f}"
                )
                obs = vec_env.reset()

    vec_env.close()


if __name__ == "__main__":
    main()