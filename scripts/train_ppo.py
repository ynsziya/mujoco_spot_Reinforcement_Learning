#!/usr/bin/env python3
"""PPO ile Spot yürüyüş eğitimi."""

from pathlib import Path
import sys

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.spot_walk_env import SpotWalkEnv

MODELS_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
MODELS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


def make_env():
    env = SpotWalkEnv()
    # ~20 saniye @ 50 Hz (frame_skip=10, timestep=0.002)
    env = TimeLimit(env, max_episode_steps=1000)
    return env


def main() -> None:
    # İlk deneme: 1 env. Hızlanınca n_envs=4 veya 8 yap.
    n_envs = 4
    total_timesteps = 5_000_000  # kısa test; sonra 5_000_000+ yükselt

    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=DummyVecEnv)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(LOG_DIR / "tb"),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=str(MODELS_DIR),
        name_prefix="ppo_spot",
    )

    print(f"Eğitim başlıyor: {total_timesteps} step, n_envs={n_envs}")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_cb)

    out = MODELS_DIR / "ppo_spot_final.zip"
    model.save(str(out))
    print(f"Kaydedildi: {out}")
    env.close()


if __name__ == "__main__":
    main()