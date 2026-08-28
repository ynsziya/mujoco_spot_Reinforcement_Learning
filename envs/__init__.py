from .spot_walk_env import SpotWalkEnv

__all__ = ["SpotWalkEnv"]

try:
    from .spot_locomotion_mjx import SpotLocomotionEnv, config_for_stage

    __all__ += ["SpotLocomotionEnv", "config_for_stage"]
except ImportError:
    pass
