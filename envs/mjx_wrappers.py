"""Training wrappers for MJX Spot env (vmap, episode, true autoreset, DR)."""

from __future__ import annotations

from typing import Callable, Optional

from brax.envs.base import Env, State, Wrapper
import jax
from jax import numpy as jp
from mujoco import mjx


class VmapWrapper(Wrapper):
    """Vectorize a single-env Brax environment."""

    def __init__(self, env: Env, batch_size: Optional[int] = None):
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, rng: jax.Array) -> State:
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        return jax.vmap(self.env.reset)(rng)

    def step(self, state: State, action: jax.Array) -> State:
        return jax.vmap(self.env.step)(state, action)


class EpisodeWrapper(Wrapper):
    """Track episode length and mark truncations."""

    def __init__(self, env: Env, episode_length: int, action_repeat: int = 1):
        super().__init__(env)
        self.episode_length = int(episode_length)
        self.action_repeat = int(action_repeat)

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        zeros = jp.zeros(rng.shape[:-1])
        state.info["steps"] = zeros
        state.info["truncation"] = zeros
        state.info["time_out"] = zeros
        state.info["episode_done"] = zeros
        episode_metrics = {
            "sum_reward": jp.zeros(rng.shape[:-1]),
            "length": jp.zeros(rng.shape[:-1]),
        }
        for name in state.metrics:
            episode_metrics[name] = jp.zeros(rng.shape[:-1])
        state.info["episode_metrics"] = episode_metrics
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if self.action_repeat == 1:
            state = self.env.step(state, action)
        else:

            def f(carry, _):
                nstate = self.env.step(carry, action)
                return nstate, nstate.reward

            state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
            state = state.replace(reward=jp.sum(rewards, axis=0))

        steps = state.info["steps"] + self.action_repeat
        one = jp.ones_like(state.done)
        zero = jp.zeros_like(state.done)
        done = jp.where(steps >= self.episode_length, one, state.done)
        state.info["truncation"] = jp.where(
            steps >= self.episode_length, 1 - state.done, zero
        )
        state.info["time_out"] = state.info["truncation"]
        state.info["steps"] = steps

        prev_done = state.info["episode_done"]
        state.info["episode_metrics"]["sum_reward"] = (
            state.info["episode_metrics"]["sum_reward"] * (1 - prev_done)
            + state.reward
        )
        state.info["episode_metrics"]["length"] = (
            state.info["episode_metrics"]["length"] * (1 - prev_done)
            + self.action_repeat
        )
        for name in state.metrics:
            state.info["episode_metrics"][name] = (
                state.info["episode_metrics"][name] * (1 - prev_done)
                + state.metrics[name]
            )
        state.info["episode_done"] = done
        return state.replace(done=done)


def _where_done(done: jax.Array, x, y):
    """Select y where done, else x (broadcast done over x's trailing dims)."""
    if not hasattr(x, "shape"):
        return y
    if not x.shape:
        return jp.where(done, y, x)
    d = done
    if d.ndim == 0:
        return jp.where(d, y, x)
    d = jp.reshape(d, (d.shape[0],) + (1,) * (x.ndim - 1))
    return jp.where(d, y, x)


class TrueAutoResetWrapper(Wrapper):
    """On done, call a fresh env.reset() so the next episode gets a new command.

    Keeps the terminating transition's reward / done / truncation / metrics so
    PPO's GAE still sees the terminal signal.
    """

    def reset(self, rng: jax.Array) -> State:
        return self.env.reset(rng)

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info:
            steps = jp.where(
                state.done, jp.zeros_like(state.info["steps"]), state.info["steps"]
            )
            state.info.update(steps=steps)

        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        done = state.done
        reward = state.reward
        metrics = state.metrics
        truncation = state.info.get("truncation", jp.zeros_like(done))
        time_out = state.info.get("time_out", truncation)
        episode_done = state.info.get("episode_done", done)
        episode_metrics = state.info.get("episode_metrics", {})
        kept_steps = state.info.get("steps")

        rng = state.info["rng"]
        batched = rng.ndim > 1
        if batched:
            split_keys = jax.vmap(lambda k: jax.random.split(k))(rng)  # (N, 2, 2)
            reset_keys = split_keys[:, 0]
            next_rng = split_keys[:, 1]
            reset_state = self.env.reset(reset_keys)
        else:
            reset_key, next_rng = jax.random.split(rng)
            reset_state = self.env.reset(reset_key)

        pipeline = jax.tree.map(
            lambda a, b: _where_done(done, a, b),
            state.pipeline_state,
            reset_state.pipeline_state,
        )
        obs = jax.tree.map(
            lambda a, b: _where_done(done, a, b),
            state.obs,
            reset_state.obs,
        )
        new_info = dict(state.info)
        for k, v in reset_state.info.items():
            if k in ("steps", "truncation", "time_out", "episode_done", "episode_metrics"):
                continue
            if k in new_info:
                new_info[k] = jax.tree.map(
                    lambda a, b: _where_done(done, a, b), new_info[k], v
                )
            else:
                new_info[k] = v
        new_info["rng"] = next_rng
        new_info["truncation"] = truncation
        new_info["time_out"] = time_out
        new_info["episode_done"] = episode_done
        new_info["episode_metrics"] = episode_metrics
        if kept_steps is not None:
            new_info["steps"] = kept_steps
        return state.replace(
            pipeline_state=pipeline,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=new_info,
        )


class MjxDomainRandomizationVmapWrapper(Wrapper):
    """Vmap over a batch of randomized mjx.Model instances.

    ``randomization_fn`` matches Brax after ``partial(..., rng=)``:
    ``(model) -> (model_batch, in_axes)``.
    """

    def __init__(self, env: Env, randomization_fn: Callable):
        super().__init__(env)
        base_model = env.unwrapped.mjx_model
        self._model_v, self._in_axes = randomization_fn(base_model)

    def reset(self, rng: jax.Array) -> State:
        def reset(model, key):
            env = self.env.unwrapped
            old = env.mjx_model
            env.mjx_model = model
            try:
                return env.reset(key)
            finally:
                env.mjx_model = old

        return jax.vmap(reset, in_axes=[self._in_axes, 0])(self._model_v, rng)

    def step(self, state: State, action: jax.Array) -> State:
        def step(model, s, a):
            env = self.env.unwrapped
            old = env.mjx_model
            env.mjx_model = model
            try:
                return env.step(s, a)
            finally:
                env.mjx_model = old

        return jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
            self._model_v, state, action
        )


def make_domain_randomize_fn(
    *,
    friction_range=(0.5, 1.4),
    mass_scale=(0.85, 1.15),
    damping_scale=(0.8, 1.2),
):
    """Return ``(model, rng) -> (model_batch, in_axes)`` for Brax PPO.

    ``rng`` is expected to have shape ``(num_envs, 2)`` (already split by Brax).
    """

    def randomization_fn(model: mjx.Model, rng: jax.Array):
        def rand_one(key):
            k1, k2, k3 = jax.random.split(key, 3)
            friction = jax.random.uniform(
                k1,
                (model.geom_friction.shape[0],),
                minval=friction_range[0],
                maxval=friction_range[1],
            )
            geom_friction = model.geom_friction.at[:, 0].set(friction)
            mass_s = jax.random.uniform(
                k2, model.body_mass.shape, minval=mass_scale[0], maxval=mass_scale[1]
            )
            body_mass = model.body_mass * mass_s
            damp_s = jax.random.uniform(
                k3, (), minval=damping_scale[0], maxval=damping_scale[1]
            )
            dof_damping = model.dof_damping * damp_s
            return model.replace(
                geom_friction=geom_friction,
                body_mass=body_mass,
                dof_damping=dof_damping,
            )

        model_v = jax.vmap(rand_one)(rng)
        in_axes = jax.tree.map(lambda _: 0, model_v)
        return model_v, in_axes

    return randomization_fn


def wrap_for_training(
    env: Env,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn=None,
) -> Wrapper:
    """Brax-compatible wrap_env_fn: Vmap/DR → Episode → TrueAutoReset."""
    if randomization_fn is not None:
        env = MjxDomainRandomizationVmapWrapper(env, randomization_fn)
    else:
        env = VmapWrapper(env)
    env = EpisodeWrapper(env, episode_length, action_repeat)
    env = TrueAutoResetWrapper(env)
    return env
