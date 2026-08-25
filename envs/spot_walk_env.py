from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scene.xml"

N_LEG = 12


class SpotWalkEnv(gym.Env):
    """Spot env: ctrl + obs + walk reward / fall termination."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)

        self.home_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        if self.home_key_id < 0:
            raise RuntimeError("Keyframe 'home' bulunamadı.")

        self.frame_skip = 10
        self.default_pose = self.model.key_ctrl[self.home_key_id, :N_LEG].copy()
        self.action_scale = 0.3
        self.fall_height = 0.25
        self._last_action = np.zeros(N_LEG, dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_LEG,), dtype=np.float32
        )
        obs_dim = self._get_obs().shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)
        self._last_action = np.zeros(N_LEG, dtype=np.float32)

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        targets = self.default_pose + action * self.action_scale
        self.data.ctrl[:N_LEG] = targets
        self.data.ctrl[N_LEG:] = 0.0

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._last_action = action.copy()

        obs = self._get_obs()
        reward, terminated, info = self._compute_reward(action)
        truncated = False
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action: np.ndarray):
        forward_vel = float(self.data.qvel[0])
        height = float(self.data.qpos[2])
        gravity_z = float(self._gravity_in_body_frame(self.data.qpos[3:7])[2])
        upright = float(np.clip(-gravity_z, 0.0, 1.0))

        action_penalty = float(np.sum(np.square(action)))
        ang_vel = float(np.sum(np.square(self.data.qvel[3:6])))

        reward = (
            1.0 * forward_vel
            + 0.5 * upright
            - 0.01 * action_penalty
            - 0.05 * ang_vel
        )

        terminated = bool(height < self.fall_height)

        info = {
            "forward_vel": forward_vel,
            "height": height,
            "upright": upright,
            "reward": reward,
        }
        return reward, terminated, info

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel

        base_lin_vel = qvel[0:3].astype(np.float32)
        base_ang_vel = qvel[3:6].astype(np.float32)
        gravity_body = self._gravity_in_body_frame(qpos[3:7])
        joint_pos = (qpos[7 : 7 + N_LEG] - self.default_pose).astype(np.float32)
        joint_vel = qvel[6 : 6 + N_LEG].astype(np.float32)

        return np.concatenate(
            [
                base_lin_vel,
                base_ang_vel,
                gravity_body,
                joint_pos,
                joint_vel,
                self._last_action,
            ]
        ).astype(np.float32)

    def _gravity_in_body_frame(self, quat_wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = [float(v) for v in quat_wxyz]
        gx, gy, gz = 0.0, 0.0, -1.0
        return np.array(
            [
                gx * (1 - 2 * (y * y + z * z))
                + gy * (2 * (x * y - w * z))
                + gz * (2 * (x * z + w * y)),
                gx * (2 * (x * y + w * z))
                + gy * (1 - 2 * (x * x + z * z))
                + gz * (2 * (y * z - w * x)),
                gx * (2 * (x * z - w * y))
                + gy * (2 * (y * z + w * x))
                + gz * (1 - 2 * (x * x + y * y)),
            ],
            dtype=np.float32,
        )

    def close(self):
        pass