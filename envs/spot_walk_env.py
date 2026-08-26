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

        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        if self.base_body_id < 0:
            raise RuntimeError("Body 'base_link' bulunamadı.")

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

        obs = self._get_obs()
        reward, terminated, info = self._compute_reward(action)
        self._last_action = action.copy()
        truncated = False
        return obs, reward, terminated, truncated, info

    def _base_lin_vel_body(self) -> np.ndarray:
        """World lin vel → base body frame (x=ileri, y=yana)."""
        R = self.data.xmat[self.base_body_id].reshape(3, 3)
        v_world = self.data.qvel[0:3]
        return (R.T @ v_world).astype(np.float64)

    def _compute_reward(self, action: np.ndarray):
        v_body = self._base_lin_vel_body()
        forward_vel = float(v_body[0])
        lateral_vel = float(v_body[1])
        world_vx = float(self.data.qvel[0])
        lateral_pos = float(self.data.qpos[1])
        height = float(self.data.qpos[2])
        yaw_rate = float(self.data.qvel[5])
        roll_pitch_rate = float(np.sum(np.square(self.data.qvel[3:5])))

        gravity_z = float(self._gravity_in_body_frame(self.data.qpos[3:7])[2])
        upright = float(np.clip(-gravity_z, 0.0, 1.0))
        yaw = self._yaw_from_quat(self.data.qpos[3:7])
        heading = float(np.cos(yaw))  # 1 = +x bakış
        # y sapmasını sınırla; karesel birikim epizodu öldürmesin
        lat_track = float(np.tanh(lateral_pos))

        action_penalty = float(np.sum(np.square(action)))

        # Body ileri hız + +x bakış; yana/yaw sert ceza
        reward = (
            2.0 * forward_vel
            + 0.5 * world_vx
            + 0.5 * upright
            + 1.5 * heading
            - 1.5 * yaw ** 2
            - 2.0 * lateral_vel ** 2
            - 0.8 * lat_track ** 2
            - 0.5 * yaw_rate ** 2
            - 0.05 * roll_pitch_rate
            - 0.01 * action_penalty
        )

        terminated = bool(height < self.fall_height)

        info = {
            "forward_vel": forward_vel,
            "world_vx": world_vx,
            "lateral_vel": lateral_vel,
            "lateral_pos": lateral_pos,
            "yaw": yaw,
            "height": height,
            "upright": upright,
            "reward": reward,
        }
        return reward, terminated, info

    @staticmethod
    def _yaw_from_quat(quat_wxyz: np.ndarray) -> float:
        w, x, y, z = [float(v) for v in quat_wxyz]
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel

        base_lin_vel_body = self._base_lin_vel_body().astype(np.float32)
        base_ang_vel = qvel[3:6].astype(np.float32)
        gravity_body = self._gravity_in_body_frame(qpos[3:7])
        yaw = self._yaw_from_quat(qpos[3:7])
        heading_feat = np.array(
            [np.sin(yaw), np.cos(yaw), np.tanh(qpos[1])], dtype=np.float32
        )
        joint_pos = (qpos[7 : 7 + N_LEG] - self.default_pose).astype(np.float32)
        joint_vel = qvel[6 : 6 + N_LEG].astype(np.float32)

        return np.concatenate(
            [
                base_lin_vel_body,
                base_ang_vel,
                gravity_body,
                heading_feat,
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
