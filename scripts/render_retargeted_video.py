"""Render a retargeted trajectory (.h5) to an mp4 video using Drake's CPU renderer.

Sim-free / Isaac-free: only needs pydrake + h5py + imageio. Works on macOS arm64
(RenderEngineVtk is CPU-based). Renders the actual URDF meshes (hand + object + table).

Example:
    export PYTHONPATH=$PWD/source/regrind:$PYTHONPATH
    export REGRIND_DATA_DIR=$PWD/data
    python scripts/render_retargeted_video.py \
        --robot wujihand --object scissors \
        --traj data/retargeted_traj/wujihand/scissors/retargeted_120fps.h5 \
        --out /tmp/wuji_scissors.mp4 --fps 120
"""

import argparse
import os

import h5py
import numpy as np
from tqdm import tqdm

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    RigidTransform,
    RotationMatrix,
)
from pydrake.systems.sensors import CameraInfo
from pydrake.geometry import (
    ClippingRange,
    ColorRenderCamera,
    MakeRenderEngineVtk,
    RenderCameraCore,
    RenderEngineVtkParams,
)

from regrind.retargeting.drake_utils import create_plant


def _load_robot_object_cfg(robot_name: str, object_name: str):
    if robot_name == "leaphand":
        from regrind.retargeting import leaphand_constants as rc
    elif robot_name == "wujihand":
        from regrind.retargeting import wujihand_constants as rc
    else:
        raise ValueError(robot_name)
    cfg = rc.get_object_config(object_name)
    # Prefer a render-only copy of the object URDF whose visual meshes carry vertex
    # normals (VTK requires them) and whose collision tags are stripped. Fall back to
    # the original if the render copy does not exist.
    from regrind.assets import REGRIND_ASSETS_DIR
    render_urdf = REGRIND_ASSETS_DIR / f"{object_name}_render" / f"{object_name}.urdf"
    if render_urdf.exists():
        cfg["object_urdf_file"] = str(render_urdf)
    return rc, cfg


def _robot_render_urdf(rc, robot_name: str):
    """Prefer a *_render.urdf variant whose visual STL meshes were converted to
    normal-carrying OBJ (RenderEngineVtk ignores STL). Fall back to the original."""
    orig = rc.ROBOT_URDF_FILE
    cand = orig.replace(".urdf", "_render.urdf")
    return cand if os.path.exists(cand) else orig


def _look_at(eye, target, width=960, height=720, fov_y_deg=45.0):
    """Drake color-camera frame convention: +z forward (into scene), +x right, +y down."""
    eye = np.asarray(eye, float)
    z_c = np.asarray(target, float) - eye
    z_c /= np.linalg.norm(z_c)
    up = np.array([0.0, 0.0, 1.0])
    x_c = np.cross(z_c, up)
    x_c /= np.linalg.norm(x_c)
    y_c = np.cross(z_c, x_c)
    R_WC = np.column_stack([x_c, y_c, z_c])
    X_WC = RigidTransform(RotationMatrix(R_WC), eye)
    intrinsics = CameraInfo(width=width, height=height, fov_y=np.radians(fov_y_deg))
    core = RenderCameraCore("vtk", intrinsics, ClippingRange(0.01, 20.0), RigidTransform())
    return X_WC, ColorRenderCamera(core, show_window=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", default="wujihand", choices=("leaphand", "wujihand"))
    p.add_argument("--object", default="scissors", choices=("scissors", "screwdriver"))
    p.add_argument("--traj", required=True, help="retargeted .h5 (robot_pos/quat/joints, object_pos/quat/joint)")
    p.add_argument("--out", default="/tmp/retarget_render.mp4")
    p.add_argument("--fps", type=int, default=120, help="original trajectory is 120fps")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--eye", type=float, nargs=3, default=[0.55, 0.55, 1.35])
    p.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 0.98])
    args = p.parse_args()

    rc, obj_cfg = _load_robot_object_cfg(args.robot, args.object)

    # --- Build plant + scene graph with a registered VTK renderer ---
    plant, scene_graph, builder = create_plant(
        robot_model_path=_robot_render_urdf(rc, args.robot),
        object_model_path=obj_cfg["object_urdf_file"],
        collision_exclusion_setter=None,
        table_height=obj_cfg["table_height"],
    )
    scene_graph.AddRenderer("vtk", MakeRenderEngineVtk(RenderEngineVtkParams()))

    X_WC, color_camera = _look_at(args.eye, args.target, args.width, args.height)
    world_frame_id = plant.GetBodyFrameIdOrThrow(plant.world_body().index())

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, context)
    sg_context = diagram.GetMutableSubsystemContext(scene_graph, context)
    query_port = scene_graph.get_query_output_port()

    # --- Load trajectory and assemble q ---
    with h5py.File(args.traj, "r") as f:
        robot_quat = f["robot_quat"][:]   # (T,4) wxyz
        robot_pos = f["robot_pos"][:]     # (T,3)
        robot_joints = f["robot_joints"][:]  # (T,DOF)
        object_quat = f["object_quat"][:]    # (T,4) wxyz
        object_pos = f["object_pos"][:]      # (T,3)
        object_joint = f["object_joint"][:] if "object_joint" in f else None

    T = robot_pos.shape[0]
    if args.max_frames:
        T = min(T, args.max_frames)
    nq = plant.num_positions()
    print(f"plant nq={nq}, frames={T}, fps={args.fps}")

    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8)

    for t in tqdm(range(T)):
        q = np.zeros(nq)
        q[0:4] = robot_quat[t] / np.linalg.norm(robot_quat[t])
        q[4:7] = robot_pos[t]
        dof = robot_joints.shape[1]
        q[7 : 7 + dof] = robot_joints[t]
        obj_start = 7 + dof
        oq = object_quat[t]
        q[obj_start : obj_start + 4] = oq / np.linalg.norm(oq)
        q[obj_start + 4 : obj_start + 7] = object_pos[t]
        if object_joint is not None and nq > obj_start + 7:
            q[obj_start + 7] = np.asarray(object_joint[t]).reshape(-1)[0]

        plant.SetPositions(plant_context, q)
        query_object = query_port.Eval(sg_context)
        color_image = query_object.RenderColorImage(color_camera, world_frame_id, X_WC)
        rgb = np.asarray(color_image.data)[:, :, :3]
        writer.append_data(rgb)

    writer.close()
    print(f"Saved video to {args.out}")


if __name__ == "__main__":
    main()
