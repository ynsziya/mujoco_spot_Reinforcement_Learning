# bosdyn_spot MuJoCo package
# Source: robot_dog_laser_designator (URDF + STL meshes)

## Layout

```
mujoco_spot/
  spot.xml              # robot MJCF (12 leg DoF + 2 laser DoF + freejoint)
  scene.xml             # robot + groundplane / skybox
  assets/*.STL          # visual meshes (mm, scaled 0.001 in MJCF)
  urdf/spot_zero_mujoco.urdf   # MuJoCo-ready URDF used for conversion
  scripts/urdf_to_mjcf.py      # regenerate MJCF from the ROS URDF
  scripts/view_spot.py         # passive viewer
```

## Joints / actuators

| Actuator | Joint | Axis | Notes |
|---|---|---|---|
| 12 leg | `*_hip_roll_joint`, `*_hip_yaw_joint`, `*_knee_joint` | X / Y / Y | position actuators |
| 2 laser | `laser_pointer_body_joint`, `laser_pointer_head_joint` | Z / Y | pan ±π, tilt ±30° |

`home` keyframe = Gazebo stand pose (hip yaw 0.764, knee -1.646).

Feet contact via sphere geoms on each `*_knee` (mesh collisions are visual-only).

## Usage

```bash
# from mujoco_rl_ws
source mjenv/bin/activate
cd mujoco_spot

# view
python scripts/view_spot.py
# or
python -m mujoco.viewer --mjcf scene.xml

# regenerate after URDF / mesh changes
python scripts/urdf_to_mjcf.py
```

Requires the `mjenv` MuJoCo install (tested with 3.12) and the source package at
`/home/yunus/ros2_ws/src/robot_dog_laser_designator` for regeneration.

## Conversion notes

Follows the MuJoCo Menagerie URDF→MJCF pattern:

1. Strip Gazebo / ros2_control, rewrite mesh paths, drop `base_footprint`
2. Load URDF with `fusestatic="false"` (keeps laser / sensor frames)
3. Add freejoint, visual/collision classes, foot spheres, contact excludes,
   position actuators, and a stand keyframe

Masses / inertias are taken from the source URDF as-is (not retuned).
