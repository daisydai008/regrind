"""Filtered left Wuji fingertip–table contact sensors for an explicitly named hand."""

from __future__ import annotations

from isaaclab.sensors import ContactSensorCfg

WUJI_LEFT_FINGERTIP_TABLE_CONTACT_SENSOR_NAMES: tuple[str, ...] = tuple(
    f"contact_left_fingertip{i}_table" for i in range(1, 6)
)


def install_wuji_left_fingertip_table_contact_sensors(scene) -> None:
    """Install five sensors on ``scene.robot_left`` against ``scene.table``.

    URDF-imported bodies live directly below ``RobotLeft``. Sensor names and robot
    binding are separate from the existing right-hand installer.
    """
    robot = getattr(scene, "robot_left", None)
    if robot is None:
        return
    if not getattr(robot.spawn, "activate_contact_sensors", False):
        scene.robot_left = robot.replace(spawn=robot.spawn.replace(activate_contact_sensors=True))
    for i, attr in enumerate(WUJI_LEFT_FINGERTIP_TABLE_CONTACT_SENSOR_NAMES, start=1):
        if getattr(scene, attr, None) is not None:
            continue
        setattr(
            scene,
            attr,
            ContactSensorCfg(
                prim_path=rf"{{ENV_REGEX_NS}}/RobotLeft/left_finger{i}_link4",
                filter_prim_paths_expr=[r"{ENV_REGEX_NS}/Table"],
                history_length=3,
                track_air_time=False,
                debug_vis=False,
                force_threshold=0.1,
            ),
        )
