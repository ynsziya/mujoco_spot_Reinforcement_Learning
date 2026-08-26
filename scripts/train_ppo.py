#!/usr/bin/env python3
"""PPO ile Spot yürüyüş eğitimi."""

import argparse
import os
from pathlib import Path
import sys

# SubprocVecEnv ile oversubscribe olmasın (MuJoCo/BLAS iç thread'leri)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.spot_walk_env import SpotWalkEnv

MODELS_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
DEFAULT_MODEL = MODELS_DIR / "ppo_spot_final.zip"
MODELS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


def make_env():
    env = SpotWalkEnv()
    # ~20 saniye @ 50 Hz (frame_skip=10, timestep=0.002)
    env = TimeLimit(env, max_episode_steps=1000)
    return env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spot PPO eğitimi")
    p.add_argument(
        "--resume",
        nargs="?",
        const=str(DEFAULT_MODEL),
        default=None,
        metavar="PATH",
        help=(
            "Kaldığı yerden devam et. PATH verilmezse "
            f"{DEFAULT_MODEL.name} kullanılır."
        ),
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=2_000_000,
        help="Bu çalıştırmada eklenecek adım sayısı (varsayılan: 2e6)",
    )
    p.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Paralel ortam sayısı (varsayılan: 8; CPU çekirdeğine göre 8–12 dene)",
    )
    p.add_argument(
        "--vec-env",
        choices=("subproc", "dummy"),
        default="subproc",
        help="subproc=çok süreç (hızlı), dummy=tek süreç (debug)",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=250_000,
        help="Kaç env-step'te bir checkpoint (varsayılan: 250000)",
    )
    return p.parse_args()


def resolve_model_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Model bulunamadı: {path}")
    return path


def main() -> None:
    args = parse_args()
    n_envs = max(1, args.n_envs)
    total_timesteps = args.timesteps
    vec_env_cls = SubprocVecEnv if args.vec_env == "subproc" else DummyVecEnv

    # n_envs büyüdükçe rollout buffer büyür; batch'i biraz aç
    batch_size = 64 if n_envs <= 4 else min(256, 64 * (n_envs // 4))

    print(
        f"VecEnv={vec_env_cls.__name__}, n_envs={n_envs}, "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}"
    )
    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=vec_env_cls)

    if args.resume is not None:
        resume_path = resolve_model_path(args.resume)
        print(f"Devam: {resume_path}")
        model = PPO.load(
            str(resume_path),
            env=env,
            tensorboard_log=str(LOG_DIR / "tb"),
        )
        model.ent_coef = 0.001
        reset_num_timesteps = False
    else:
        print("Yeni eğitim başlıyor")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.001,
            verbose=1,
            tensorboard_log=str(LOG_DIR / "tb"),
        )
        reset_num_timesteps = True

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_every // n_envs, 1),
        save_path=str(MODELS_DIR),
        name_prefix="ppo_spot",
    )

    already = int(model.num_timesteps)
    print(
        f"Eğitim: +{total_timesteps} step "
        f"(şimdi {already} → hedef ~{already + total_timesteps}), "
        f"n_envs={n_envs}, batch_size={getattr(model, 'batch_size', batch_size)}"
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        reset_num_timesteps=reset_num_timesteps,
    )

    out = DEFAULT_MODEL
    model.save(str(out))
    print(f"Kaydedildi: {out} (toplam step={model.num_timesteps})")
    env.close()


if __name__ == "__main__":
    # SubprocVecEnv Windows/Linux için spawn güvenliği
    main()
