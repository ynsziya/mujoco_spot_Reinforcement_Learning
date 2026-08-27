#!/usr/bin/env python3
"""PPO ile doğal ve kararlı dört bacaklı Spot yürüyüş eğitimi."""

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
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.spot_walk_env import SpotWalkEnv

MODELS_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
DEFAULT_MODEL = MODELS_DIR / "ppo_spot_final.zip"
DEFAULT_VECNORM = MODELS_DIR / "ppo_spot_vecnormalize.pkl"
MODELS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


def linear_schedule(initial_value: float):
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule


class VecNormalizeSaveCallback(BaseCallback):
    """Periodically persist observation/reward normalization statistics."""

    def __init__(self, save_freq: int, save_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = max(1, save_freq)
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self.training_env.save(str(self.save_path))
            if self.verbose > 0:
                print(f"VecNormalize kaydedildi: {self.save_path}")
        return True


def make_env(randomize_command: bool = True):
    env = SpotWalkEnv(randomize_command=randomize_command)
    # ~20 saniye @ 50 Hz (frame_skip=10, timestep=0.002)
    env = TimeLimit(env, max_episode_steps=1000)
    return Monitor(env)


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
        default=5_000_000,
        help="Bu çalıştırmada eklenecek adım sayısı (varsayılan: 5e6)",
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
    p.add_argument(
        "--eval-every",
        type=int,
        default=100_000,
        help="Kaç env-step'te bir deterministik değerlendirme (varsayılan: 100000)",
    )
    p.add_argument(
        "--vecnormalize",
        default=str(DEFAULT_VECNORM),
        metavar="PATH",
        help="VecNormalize istatistik dosyası (varsayılan: models/ppo_spot_vecnormalize.pkl)",
    )
    return p.parse_args()


def resolve_model_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Model bulunamadı: {path}")
    return path


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    return path


def main() -> None:
    args = parse_args()
    n_envs = max(1, args.n_envs)
    total_timesteps = args.timesteps
    vec_env_cls = SubprocVecEnv if args.vec_env == "subproc" else DummyVecEnv
    vecnorm_path = resolve_path(args.vecnormalize)

    n_steps = 1024
    batch_size = min(2048, max(256, 256 * n_envs))

    print(
        f"VecEnv={vec_env_cls.__name__}, n_envs={n_envs}, "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}"
    )
    env = make_vec_env(
        make_env,
        n_envs=n_envs,
        vec_env_cls=vec_env_cls,
        env_kwargs={"randomize_command": True},
    )
    if args.resume is not None and vecnorm_path.exists():
        print(f"VecNormalize yükleniyor: {vecnorm_path}")
        env = VecNormalize.load(str(vecnorm_path), env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
        )

    eval_env = make_vec_env(
        make_env,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        env_kwargs={"randomize_command": False},
    )
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    if args.resume is not None:
        resume_path = resolve_model_path(args.resume)
        print(f"Devam: {resume_path}")
        model = PPO.load(
            str(resume_path),
            env=env,
            tensorboard_log=str(LOG_DIR / "tb"),
        )
        reset_num_timesteps = False
    else:
        print("Yeni eğitim başlıyor")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=linear_schedule(3e-4),
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.03,
            policy_kwargs={
                "activation_fn": th.nn.ELU,
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            },
            verbose=1,
            tensorboard_log=str(LOG_DIR / "tb"),
        )
        reset_num_timesteps = True

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_every // n_envs, 1),
        save_path=str(MODELS_DIR),
        name_prefix="ppo_spot",
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR / "best"),
        log_path=str(LOG_DIR / "eval"),
        eval_freq=max(args.eval_every // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    vecnorm_cb = VecNormalizeSaveCallback(
        save_freq=max(args.checkpoint_every, 1),
        save_path=vecnorm_path,
        verbose=1,
    )
    callbacks = CallbackList(
        [
            checkpoint_cb,
            eval_cb,
            vecnorm_cb,
        ]
    )

    already = int(model.num_timesteps)
    print(
        f"Eğitim: +{total_timesteps} step "
        f"(şimdi {already} → hedef ~{already + total_timesteps}), "
        f"n_envs={n_envs}, batch_size={getattr(model, 'batch_size', batch_size)}"
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps,
    )

    out = DEFAULT_MODEL
    model.save(str(out))
    env.save(str(vecnorm_path))
    print(f"Kaydedildi: {out} (toplam step={model.num_timesteps})")
    print(f"VecNormalize kaydedildi: {vecnorm_path}")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    # SubprocVecEnv Windows/Linux için spawn güvenliği
    main()
