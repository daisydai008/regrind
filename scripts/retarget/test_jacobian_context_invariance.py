"""检查 object-frame Jacobians 是否独立于 fresh-instance 首次调用的 context 状态。

从 regrind repo 运行（需要 pydrake，不需要 Mosek license 或轨迹数据）：
    PYTHONPATH=source/regrind python -B scripts/retarget/test_jacobian_context_invariance.py

Visualization tools: Drake Meshcat (production).
Visualization tools: scene-prep PNG audit (tmp PNGs from bash/matplotlib).
Visualization tools: Drake compare videos (data/tactidex/videos).
"""

import numpy as np

from regrind.assets import REGRIND_ASSETS_DIR
from regrind.retargeting.retargeter import HandInteractionMeshOneStageRetargeter
from regrind.retargeting.wujihand_constants import (
    MANO_JOINTS,
    MANO_TO_WUJI_MAPPING,
    ROBOT_URDF_FILE,
)


def main():
    retargeter = HandInteractionMeshOneStageRetargeter(
        robot_model_path=ROBOT_URDF_FILE,
        robot_name="wujihand",
        object_model_path=str(REGRIND_ASSETS_DIR / "tactidex/grindhandle/grindhandle.urdf"),
        object_name="grindhandle",
        table_height=0.012,
        demo_joints=MANO_JOINTS,
        laplacian_match_links=MANO_TO_WUJI_MAPPING,
    )
    q = retargeter.plant.GetPositions(retargeter.plant_context).copy()
    q[retargeter.q_a_indices[4:7]] = [0.20, -0.10, 0.15]
    q[retargeter.q_a_indices[7:]] = 0.25
    # A non-identity object rotation exposes stale object-frame Jacobians.
    q[-7:] = [np.cos(0.6), 0.0, 0.0, np.sin(0.6), 0.23, -0.11, 0.19]

    first_J, first_p, _ = retargeter._calc_manipulator_jacobians(
        q, MANO_TO_WUJI_MAPPING, obj_frame=True
    )
    second_J, second_p, _ = retargeter._calc_manipulator_jacobians(
        q, MANO_TO_WUJI_MAPPING, obj_frame=True
    )
    max_J_diff = max(
        float(np.max(np.abs(first_J[name] - second_J[name])))
        for name in MANO_TO_WUJI_MAPPING
    )
    max_p_diff = max(
        float(np.max(np.abs(first_p[name] - second_p[name])))
        for name in MANO_TO_WUJI_MAPPING
    )
    print(f"first/second max abs Jacobian diff: {max_J_diff:.12e}", flush=True)
    print(f"first/second max abs landmark diff (m): {max_p_diff:.12e}", flush=True)
    assert max_J_diff < 1e-9, f"Object-frame Jacobians depend on call history: {max_J_diff}"
    assert max_p_diff < 1e-9, f"Object-frame landmarks depend on call history: {max_p_diff}"
    print("PASS: fresh-instance first/second calls agree (< 1e-9).")


if __name__ == "__main__":
    main()
