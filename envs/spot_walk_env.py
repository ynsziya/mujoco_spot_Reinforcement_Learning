from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scene.xml"

N_LEG = 12
LEG_NAMES = ("fl", "fr", "rl", "rr")
FOOT_GEOMS = tuple(f"{leg}_foot" for leg in LEG_NAMES)
FOOT_SITES = tuple(f"{leg}_foot_site" for leg in LEG_NAMES)


class SpotWalkEnv(gym.Env):
    """MuJoCo Spot locomotion task shaped for stable trot-like walking."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None, command=None, randomize_command=True):
        super().__init__()
        self.render_mode = render_mode
        self.randomize_command = randomize_command
        self.fixed_command = command

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

        self.floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self.foot_geom_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in FOOT_GEOMS
            ],
            dtype=np.int32,
        )
        self.foot_site_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
                for name in FOOT_SITES
            ],
            dtype=np.int32,
        )
        if np.any(self.foot_geom_ids < 0) or np.any(self.foot_site_ids < 0):
            raise RuntimeError("Ayak geom/site isimleri modelde bulunamadı.")

        self.frame_skip = 10
        self.dt = float(self.model.opt.timestep * self.frame_skip)
        self.default_pose = self.model.key_ctrl[self.home_key_id, :N_LEG].copy()
        self.action_scale = np.tile([0.25, 0.45, 0.45], 4).astype(np.float32)
        self.action_filter = 0.55
        self.fall_height = 0.31
        self.max_tilt = 0.65
        self.target_height = float(self.model.key_qpos[self.home_key_id, 2])

        self.gait_period = 0.55
        self.duty_factor = 0.58
        self.gait_offsets = np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32)

        self.command = np.zeros(3, dtype=np.float32)
        self._phase = 0.0
        self._elapsed_steps = 0
        self._last_action = np.zeros(N_LEG, dtype=np.float32)
        self._applied_action = np.zeros(N_LEG, dtype=np.float32)
        self._last_foot_pos = np.zeros((4, 3), dtype=np.float32)
        self._last_contacts = np.zeros(4, dtype=np.float32)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)
        self._last_foot_pos = self._foot_positions_body().astype(np.float32)

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
        self._sample_command(options)
        self._phase = float(self.np_random.random())
        self._elapsed_steps = 0
        self._last_action.fill(0.0)
        self._applied_action.fill(0.0)
        self._last_contacts.fill(0.0)

        # Hafif başlangıç çeşitliliği ezberlenmiş tek duruşa aşırı uyumu azaltır.
        self.data.qpos[7 : 7 + N_LEG] += self.np_random.normal(0.0, 0.015, N_LEG)
        self.data.qvel[: 6 + N_LEG] += self.np_random.normal(0.0, 0.02, 6 + N_LEG)
        mujoco.mj_forward(self.model, self.data)
        self._last_foot_pos = self._foot_positions_body().astype(np.float32)

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        prev_action = self._applied_action.copy()
        prev_foot_pos = self._last_foot_pos.copy()
        self._applied_action = (
            self.action_filter * self._applied_action
            + (1.0 - self.action_filter) * action
        ).astype(np.float32)
        targets = self.default_pose + self._applied_action * self.action_scale
        self.data.ctrl[:N_LEG] = targets
        self.data.ctrl[N_LEG:] = 0.0

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._elapsed_steps += 1
        self._phase = (self._phase + self.dt / self.gait_period) % 1.0
        self._last_foot_pos = self._foot_positions_body().astype(np.float32)

        obs = self._get_obs()
        reward, terminated, info = self._compute_reward(
            action=action,
            applied_action=self._applied_action,
            prev_action=prev_action,
            prev_foot_pos=prev_foot_pos,
        )
        self._last_action = self._applied_action.copy()
        self._last_contacts = info["foot_contacts"].copy()
        truncated = False
        return obs, reward, terminated, truncated, info

    def _sample_command(self, options):
        if options is not None and "command" in options:
            command = options["command"]
        elif self.fixed_command is not None:
            command = self.fixed_command
        elif self.randomize_command:
            command = [
                self.np_random.uniform(0.35, 0.75),
                self.np_random.uniform(-0.04, 0.04),
                self.np_random.uniform(-0.10, 0.10),
            ]
        else:
            command = [0.55, 0.0, 0.0]
        self.command = np.asarray(command, dtype=np.float32)

    def _base_lin_vel_body(self) -> np.ndarray:
        """World linear velocity projected into the base frame."""
        R = self.data.xmat[self.base_body_id].reshape(3, 3)
        v_world = self.data.qvel[0:3]
        return (R.T @ v_world).astype(np.float64)

    def _foot_positions_body(self) -> np.ndarray:
        R = self.data.xmat[self.base_body_id].reshape(3, 3)
        base_pos = self.data.xpos[self.base_body_id]
        foot_world = self.data.site_xpos[self.foot_site_ids]
        return (foot_world - base_pos) @ R

    def _foot_contacts(self) -> np.ndarray:
        contacts = np.zeros(4, dtype=np.float32)
        for idx in range(self.data.ncon):
            contact = self.data.contact[idx]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 == self.floor_geom_id:
                foot_matches = np.where(self.foot_geom_ids == g2)[0]
            elif g2 == self.floor_geom_id:
                foot_matches = np.where(self.foot_geom_ids == g1)[0]
            else:
                continue
            if foot_matches.size:
                contacts[int(foot_matches[0])] = 1.0
        return contacts

    def _gait_clock(self) -> tuple[np.ndarray, np.ndarray]:
        phases = (self._phase + self.gait_offsets) % 1.0
        desired_contacts = (phases < self.duty_factor).astype(np.float32)
        clock = np.stack(
            [np.sin(2.0 * np.pi * phases), np.cos(2.0 * np.pi * phases)], axis=1
        )
        return clock.astype(np.float32), desired_contacts

    def _compute_reward(
        self,
        action: np.ndarray,
        applied_action: np.ndarray,
        prev_action: np.ndarray,
        prev_foot_pos: np.ndarray,
    ):
        v_body = self._base_lin_vel_body()
        forward_vel = float(v_body[0])
        lateral_vel = float(v_body[1])
        vertical_vel = float(self.data.qvel[2])
        yaw_rate = float(self.data.qvel[5])
        roll_pitch_rate = float(np.sum(np.square(self.data.qvel[3:5])))
        height = float(self.data.qpos[2])
        lateral_pos = float(self.data.qpos[1])

        gravity_body = self._gravity_in_body_frame(self.data.qpos[3:7])
        tilt = float(np.linalg.norm(gravity_body[:2]))
        upright = float(np.clip(-gravity_body[2], 0.0, 1.0))
        yaw = self._yaw_from_quat(self.data.qpos[3:7])

        foot_contacts = self._foot_contacts()
        _, desired_contacts = self._gait_clock()
        foot_delta = (self._last_foot_pos - prev_foot_pos) / max(self.dt, 1e-6)
        foot_speed_xy = np.linalg.norm(foot_delta[:, :2], axis=1)
        foot_height = self.data.site_xpos[self.foot_site_ids, 2]

        vel_error = np.array(
            [
                forward_vel - float(self.command[0]),
                lateral_vel - float(self.command[1]),
                yaw_rate - float(self.command[2]),
            ],
            dtype=np.float32,
        )
        tracking_reward = float(np.exp(-3.0 * np.sum(np.square(vel_error))))
        upright_reward = float(np.exp(-8.0 * tilt * tilt))
        height_reward = float(np.exp(-80.0 * (height - self.target_height) ** 2))
        heading_reward = float(np.exp(-2.0 * yaw * yaw))
        phase_reward = float(1.0 - np.mean(np.abs(foot_contacts - desired_contacts)))

        slip_penalty = float(np.mean(foot_contacts * foot_speed_xy))
        swing_clearance = np.maximum(0.0, 0.055 - foot_height)
        clearance_penalty = float(np.mean((1.0 - desired_contacts) * swing_clearance))
        stumble_penalty = float(
            np.mean((1.0 - desired_contacts) * foot_contacts * foot_speed_xy)
        )
        action_rate_penalty = float(np.mean(np.square(applied_action - prev_action)))
        action_penalty = float(np.mean(np.square(action)))
        joint_vel_penalty = float(np.mean(np.square(self.data.qvel[6 : 6 + N_LEG])))

        reward = (
            3.0 * tracking_reward
            + 1.2 * upright_reward
            + 0.8 * height_reward
            + 0.6 * heading_reward
            + 0.8 * phase_reward
            - 1.4 * lateral_vel**2
            - 0.6 * lateral_pos**2
            - 0.4 * vertical_vel**2
            - 0.12 * roll_pitch_rate
            - 0.35 * yaw_rate**2
            - 0.30 * slip_penalty
            - 0.80 * stumble_penalty
            - 0.40 * clearance_penalty
            - 0.04 * action_penalty
            - 0.10 * action_rate_penalty
            - 0.001 * joint_vel_penalty
        )

        fallen = height < self.fall_height or tilt > self.max_tilt
        terminated = bool(fallen)
        if terminated:
            reward -= 5.0

        info = {
            "forward_vel": forward_vel,
            "target_forward_vel": float(self.command[0]),
            "lateral_vel": lateral_vel,
            "target_lateral_vel": float(self.command[1]),
            "yaw_rate": yaw_rate,
            "target_yaw_rate": float(self.command[2]),
            "lateral_pos": lateral_pos,
            "yaw": yaw,
            "height": height,
            "tilt": tilt,
            "upright": upright,
            "phase_reward": phase_reward,
            "slip_penalty": slip_penalty,
            "foot_contacts": foot_contacts,
            "desired_contacts": desired_contacts,
            "reward": float(reward),
        }
        return float(reward), terminated, info

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
        clock, desired_contacts = self._gait_clock()
        foot_pos = self._foot_positions_body().astype(np.float32).reshape(-1)

        return np.concatenate(
            [
                self.command,
                base_lin_vel_body,
                base_ang_vel,
                gravity_body,
                heading_feat,
                joint_pos,
                joint_vel,
                self._last_action,
                clock.reshape(-1),
                desired_contacts,
                self._last_contacts,
                foot_pos,
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
