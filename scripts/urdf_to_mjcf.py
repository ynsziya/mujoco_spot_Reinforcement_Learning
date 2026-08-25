#!/usr/bin/env python3
"""Convert robot_dog_laser_designator spot_zero.urdf into a MuJoCo MJCF package.

Pipeline (menagerie-style):
  1. Strip Gazebo / ros2_control, rewrite mesh paths, drop base_footprint.
  2. Load with MuJoCo (fusestatic=false to keep laser / sensor frames).
  3. Save raw MJCF, then post-process: freejoint, defaults, actuators,
     foot contact spheres, contact excludes, home keyframe, scene.xml.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
SRC_URDF = Path(
    "/home/yunus/ros2_ws/src/robot_dog_laser_designator/urdf/spot_zero.urdf"
)
URDF_OUT = ROOT / "urdf" / "spot_zero_mujoco.urdf"
RAW_MJCF = ROOT / "spot_raw.xml"
SPOT_MJCF = ROOT / "spot.xml"
SCENE_MJCF = ROOT / "scene.xml"
ASSETS = ROOT / "assets"

# Stand pose from ros2_control initial_value in the source URDF.
STAND = {
    "fl_hip_roll_joint": 0.0,
    "fl_hip_yaw_joint": 0.764,
    "fl_knee_joint": -1.646,
    "fr_hip_roll_joint": 0.0,
    "fr_hip_yaw_joint": 0.764,
    "fr_knee_joint": -1.646,
    "rl_hip_roll_joint": 0.0,
    "rl_hip_yaw_joint": 0.764,
    "rl_knee_joint": -1.646,
    "rr_hip_roll_joint": 0.0,
    "rr_hip_yaw_joint": 0.764,
    "rr_knee_joint": -1.646,
    "laser_pointer_body_joint": 0.0,
    "laser_pointer_head_joint": 0.0,
}

ACTUATED = list(STAND.keys())

FOOT_SPHERES = {
    "fl_knee": (0.000730, -0.022200, -0.370380, 0.0415),
    "fr_knee": (0.000770, 0.021910, -0.370470, 0.0415),
    "rl_knee": (0.000710, -0.022170, -0.370600, 0.0415),
    "rr_knee": (0.000780, 0.021790, -0.370380, 0.0415),
}


def prepare_urdf() -> Path:
    text = SRC_URDF.read_text()

    # Drop Gazebo / ros2_control blocks (non-URDF extensions).
    text = re.sub(r"<gazebo\b[\s\S]*?</gazebo>", "", text)
    text = re.sub(r"<ros2_control\b[\s\S]*?</ros2_control>", "", text)

    # package://.../meshes/foo.STL -> assets/foo.STL (meshdir relative).
    text = re.sub(
        r'filename="package://robot_dog_laser_designator/meshes/([^"]+)"',
        r'filename="\1"',
        text,
    )

    # Inject MuJoCo compiler hints inside <robot>.
    mujoco_hint = (
        '  <mujoco>\n'
        '    <compiler meshdir="../assets" balanceinertia="true" '
        'discardvisual="false" fusestatic="false"/>\n'
        "  </mujoco>\n"
    )
    text = text.replace("<robot name=\"bosdyn_spot\">", 
                        "<robot name=\"bosdyn_spot\">\n" + mujoco_hint, 1)

    # Remove empty base_footprint chain — freejoint will be on base_link.
    text = re.sub(
        r'<!-- Ground / TF root:.*?-->\s*'
        r'<link name="base_footprint"/>\s*'
        r'<joint name="base_footprint_joint"[\s\S]*?</joint>',
        "<!-- base_footprint removed for MuJoCo freejoint on base_link -->",
        text,
        count=1,
    )

    # Drop massless optical frame (no inertial / geom).
    text = re.sub(
        r'<link name="camera_optical_frame"/>\s*'
        r'<joint name="camera_optical_joint"[\s\S]*?</joint>',
        "",
        text,
        count=1,
    )

    URDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    URDF_OUT.write_text(text)
    return URDF_OUT


def convert_urdf(urdf_path: Path) -> None:
    # Load from the urdf/ directory so meshdir="../assets" resolves.
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    mujoco.mj_saveLastXML(str(RAW_MJCF), model)
    print(f"Saved raw MJCF: {RAW_MJCF}  (nq={model.nq}, nv={model.nv}, nu={model.nu})")


def _indent(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def postprocess() -> None:
    tree = ET.parse(RAW_MJCF)
    root = tree.getroot()
    root.set("model", "bosdyn_spot")

    # --- compiler ---
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("angle", "radian")
    compiler.set("meshdir", "assets")
    compiler.set("autolimits", "true")
    # Drop absolute paths; keep only local mesh filenames.
    for mesh in root.findall("./asset/mesh"):
        f = mesh.get("file", "")
        mesh.set("file", Path(f).name)

    # --- option / visual ---
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.set("timestep", "0.002")
    option.set("integrator", "implicitfast")
    option.set("cone", "elliptic")
    option.set("impratio", "100")

    # --- defaults ---
    # Remove any existing default and replace with a clean class tree.
    for d in list(root.findall("default")):
        root.remove(d)
    default = ET.Element("default")
    spot = ET.SubElement(default, "default", {"class": "spot"})
    ET.SubElement(spot, "geom", {"solref": "0.004 1", "friction": "0.8 0.02 0.01"})
    ET.SubElement(
        spot,
        "joint",
        {"damping": "1.0", "armature": "0.01", "frictionloss": "0.1"},
    )
    ET.SubElement(
        spot,
        "position",
        {"kp": "200", "kv": "10", "inheritrange": "1"},
    )
    visual = ET.SubElement(spot, "default", {"class": "visual"})
    ET.SubElement(
        visual,
        "geom",
        {
            "group": "2",
            "type": "mesh",
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
        },
    )
    collision = ET.SubElement(spot, "default", {"class": "collision"})
    ET.SubElement(collision, "geom", {"group": "3", "contype": "1", "conaffinity": "1"})
    foot = ET.SubElement(collision, "default", {"class": "foot"})
    ET.SubElement(
        foot,
        "geom",
        {
            "type": "sphere",
            "priority": "1",
            "condim": "6",
            "solimp": "0.015 1 0.031",
            "friction": "1.0 0.02 0.01",
        },
    )
    # Insert default after compiler/option.
    insert_at = 0
    for i, child in enumerate(list(root)):
        if child.tag in ("compiler", "option", "size", "visual", "statistic"):
            insert_at = i + 1
    root.insert(insert_at, default)

    # --- materials (ensure Spot palette exists) ---
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    existing_mats = {m.get("name") for m in asset.findall("material")}
    for name, rgba in (
        ("spot_yellow", "1.0 0.82 0.0 1"),
        ("midnight_black", "0.1 0.1 0.1 1"),
        ("silver", "0.75 0.75 0.75 1"),
    ):
        if name not in existing_mats:
            ET.SubElement(asset, "material", {"name": name, "rgba": rgba})

    # --- worldbody: find base_link, add freejoint + light ---
    worldbody = root.find("worldbody")
    assert worldbody is not None

    # URDF conversion usually puts base_link as a top-level body.
    base = None
    for body in worldbody.findall("body"):
        if body.get("name") == "base_link":
            base = body
            break
    if base is None:
        # Sometimes named differently or nested — take first body.
        base = worldbody.find("body")
    assert base is not None, "Could not find base body"

    base.set("childclass", "spot")
    # Freejoint owns world pose; body pos is unused but kept at origin.
    base.set("pos", "0 0 0")

    # Remove any existing freejoint; add ours at the start of base.
    for fj in list(base.findall("freejoint")):
        base.remove(fj)
    fj = ET.Element("freejoint", {"name": "root"})
    # Insert after any inertial if present, else at front.
    base.insert(0, fj)

    # Tracking light.
    if not any(l.get("name") == "tracking" for l in base.findall("light")):
        light = ET.Element(
            "light",
            {"name": "tracking", "mode": "trackcom", "pos": "0 0 2"},
        )
        base.insert(1, light)

    # Spotlight targeting base.
    if not any(l.get("name") == "spotlight" for l in worldbody.findall("light")):
        worldbody.insert(
            0,
            ET.Element(
                "light",
                {
                    "name": "spotlight",
                    "mode": "targetbodycom",
                    "target": "base_link",
                    "pos": "3 0 4",
                    "cutoff": "30",
                },
            ),
        )

    def walk_bodies(body: ET.Element):
        yield body
        for child in body.findall("body"):
            yield from walk_bodies(child)

    # Continuous laser pan joint has no URDF range — give ±π for actuators.
    for body in walk_bodies(base):
        for joint in body.findall("joint"):
            jname = joint.get("name", "")
            if jname == "laser_pointer_body_joint":
                joint.set("axis", "0 0 1")
                joint.set("range", "-3.14159 3.14159")
            # Ensure axis is set where missing.
            if "axis" not in joint.attrib and jname.endswith("_joint"):
                pass  # leave as-is; URDF conversion usually sets axis

    for body in walk_bodies(base):
        bname = body.get("name", "")
        geoms = list(body.findall("geom"))

        # URDF→MJCF emits a visual (density=0) + collision mesh pair per link.
        # Keep one visual mesh; drop the colliding mesh duplicate.
        mesh_geoms = [g for g in geoms if g.get("mesh")]
        if len(mesh_geoms) >= 2:
            # Prefer the density=0 / group=1 visual copy; delete the rest.
            keep = next(
                (g for g in mesh_geoms if g.get("density") == "0" or g.get("group") == "1"),
                mesh_geoms[0],
            )
            for g in mesh_geoms:
                if g is not keep:
                    body.remove(g)

        for g in list(body.findall("geom")):
            gtype = g.get("type", "")
            gname = g.get("name", "")
            if gtype == "sphere" or "foot_contact" in gname:
                g.set("class", "foot")
                continue
            if g.get("mesh"):
                g.set("class", "visual")
                # Prefer palette materials over raw rgba when mesh name matches.
                rgba = g.get("rgba", "")
                if rgba.startswith("1 ") and "0.82" in rgba:
                    g.set("material", "spot_yellow")
                elif rgba.startswith("0.1 "):
                    g.set("material", "midnight_black")
                elif rgba.startswith("0.75 "):
                    g.set("material", "silver")
                g.attrib.pop("rgba", None)
                g.attrib.pop("contype", None)
                g.attrib.pop("conaffinity", None)
                g.attrib.pop("density", None)
                g.attrib.pop("group", None)

        if bname in FOOT_SPHERES:
            x, y, z, r = FOOT_SPHERES[bname]
            # Remove any leftover foot spheres then add one clean geom.
            for g in list(body.findall("geom")):
                if g.get("type") == "sphere" or "foot" in (g.get("name") or ""):
                    body.remove(g)
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"{bname[:2]}_foot",
                    "class": "foot",
                    "size": str(r),
                    "pos": f"{x} {y} {z}",
                },
            )
            if not any(
                s.get("name") == f"{bname[:2]}_foot_site" for s in body.findall("site")
            ):
                ET.SubElement(
                    body,
                    "site",
                    {
                        "name": f"{bname[:2]}_foot_site",
                        "pos": f"{x} {y} {z}",
                        "size": "0.01",
                        "group": "5",
                        "rgba": "1 0 0 1",
                    },
                )

    # IMU site on base if imu body exists.
    imu_body = None
    for body in walk_bodies(base):
        if body.get("name") == "imu_sensor":
            imu_body = body
            break
    if imu_body is not None and not any(s.get("name") == "imu" for s in imu_body.findall("site")):
        ET.SubElement(imu_body, "site", {"name": "imu", "size": "0.01", "group": "5"})

    # --- contact excludes: base vs hip_yaw (upper legs) ---
    for c in list(root.findall("contact")):
        root.remove(c)
    contact = ET.SubElement(root, "contact")
    for leg in ("fl", "fr", "rl", "rr"):
        ET.SubElement(
            contact,
            "exclude",
            {"body1": "base_link", "body2": f"{leg}_hip_yaw"},
        )
        ET.SubElement(
            contact,
            "exclude",
            {"body1": "base_link", "body2": f"{leg}_hip_roll"},
        )

    # --- actuators ---
    for a in list(root.findall("actuator")):
        root.remove(a)
    actuator = ET.SubElement(root, "actuator")
    for jname in ACTUATED:
        ET.SubElement(
            actuator,
            "position",
            {"class": "spot", "name": jname, "joint": jname},
        )

    # --- keyframe (freejoint xyz+quat + joint qpos) ---
    for k in list(root.findall("keyframe")):
        root.remove(k)

    # Joint order in MuJoCo follows depth-first body tree. We rebuild model
    # after write to measure exact qpos layout for the keyframe.
    keyframe = ET.SubElement(root, "keyframe")
    # Placeholder; filled after reload.
    ET.SubElement(keyframe, "key", {"name": "home", "qpos": "", "ctrl": ""})

    _indent(root)
    tree.write(SPOT_MJCF, encoding="unicode", xml_declaration=True)
    print(f"Wrote {SPOT_MJCF}")


def fill_keyframe() -> None:
    """Reload spot.xml and write home keyframe with correct qpos ordering."""
    model = mujoco.MjModel.from_xml_path(str(SPOT_MJCF))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    joint_addrs = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name:
            joint_addrs[name] = model.jnt_qposadr[j]

    qpos = data.qpos.copy()
    qpos[:] = 0.0
    qpos[3:7] = (1.0, 0.0, 0.0, 0.0)  # identity quat

    for jname, val in STAND.items():
        if jname not in joint_addrs:
            raise KeyError(
                f"Joint {jname} missing from model. Have: {sorted(joint_addrs)}"
            )
        qpos[joint_addrs[jname]] = val

    # Measure foot height at stand pose with base at origin, then lift so
    # foot sphere centers sit at z = radius (resting on the floor).
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    foot_z = []
    foot_r = 0.0415
    for site in ("fl_foot_site", "fr_foot_site", "rl_foot_site", "rr_foot_site"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        foot_z.append(float(data.site_xpos[sid][2]))
    lift = foot_r - min(foot_z)
    qpos[2] = lift

    ctrl = [STAND[j] for j in ACTUATED]
    qpos_str = " ".join(f"{v:.6g}" for v in qpos)
    ctrl_str = " ".join(f"{v:.6g}" for v in ctrl)

    tree = ET.parse(SPOT_MJCF)
    key = tree.getroot().find("./keyframe/key[@name='home']")
    assert key is not None
    key.set("qpos", qpos_str)
    key.set("ctrl", ctrl_str)
    _indent(tree.getroot())
    tree.write(SPOT_MJCF, encoding="unicode", xml_declaration=True)

    model = mujoco.MjModel.from_xml_path(str(SPOT_MJCF))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    print(f"home keyframe OK: nq={model.nq} nu={model.nu} freejoint_z={lift:.4f}")
    for site in ("fl_foot_site", "fr_foot_site", "rl_foot_site", "rr_foot_site"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if sid >= 0:
            print(f"  {site} z={data.site_xpos[sid][2]:.4f}")


def write_scene() -> None:
    SCENE_MJCF.write_text(
        """\
<mujoco model="bosdyn_spot scene">
  <include file="spot.xml"/>

  <statistic center="0 0 0.3" extent="1.2" meansize="0.05"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="220" elevation="-10"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
      width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
      width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
      texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane"/>
  </worldbody>
</mujoco>
"""
    )
    print(f"Wrote {SCENE_MJCF}")


def main() -> None:
    assert ASSETS.is_dir() and any(ASSETS.glob("*.STL")), "assets/ missing STL meshes"
    urdf = prepare_urdf()
    print(f"Prepared {urdf}")
    convert_urdf(urdf)
    postprocess()
    fill_keyframe()
    write_scene()
    if RAW_MJCF.exists():
        RAW_MJCF.unlink()
    # Final load of scene.
    model = mujoco.MjModel.from_xml_path(str(SCENE_MJCF))
    print(f"Scene loads: nbody={model.nbody} ngeom={model.ngeom} njnt={model.njnt}")


if __name__ == "__main__":
    main()
