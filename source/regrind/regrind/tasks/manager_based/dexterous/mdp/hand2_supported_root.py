"""Batched root-only weight support; vendor fingers and native SE3 residuals.

This is an uncalibrated controller hypothesis, not dynamics approval. Pure tensor
functions are deliberately separable by AST for CPU tests without SimulationApp.
No disk hashes, NumPy conversions, live mass reads, or per-body loops per step.
"""
import math
import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul
from .actions import SE3ImpedanceActionCfg, SE3ImpedanceActionTerm

GAINS = (300.0, 30.0, 0.01825096675712572, 0.0036501933514251437)
# Nonofficial virtual-wrist learning hypotheses, not hardware calibration.
PROFILE_GAINS = {
    "root_supported_v1": GAINS,
    "com_supported_v2": (300.0, 30.0, 3.0, 0.02),
    "com_bounded_v3": (300.0, 30.0, 3.0, 0.02),
}


def finite_tensor(value, shape, label):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"Nonfinite/malformed {label}; expected {shape}")


def unit_quaternions(value, label):
    if not bool((torch.abs(torch.linalg.vector_norm(value, dim=-1) - 1) <= 1e-5).all()):
        raise ValueError(f"Nonunit {label}")


def support_wrench(masses, body_com_pos_w, root_link_pos_w, gravity):
    """World wrench about each env's root LINK, using only Robot body COMs."""
    if masses.ndim != 2 or min(masses.shape) < 1:
        raise ValueError("Expected nonempty batched masses")
    n, bodies = masses.shape
    for value, shape, label in ((masses, (n, bodies), "masses"),
                                (body_com_pos_w, (n, bodies, 3), "body COM"),
                                (root_link_pos_w, (n, 3), "root LINK"), (gravity, (3,), "gravity")):
        finite_tensor(value, shape, label)
        if value.device != masses.device or value.dtype != masses.dtype:
            raise ValueError("Support dtype/device mismatch")
    if not bool((masses > 0).all()):
        raise ValueError("Positive body masses required")
    weight = masses.unsqueeze(-1) * gravity
    force = -weight.sum(dim=1)
    torque = -torch.cross(body_com_pos_w - root_link_pos_w.unsqueeze(1), weight, dim=-1).sum(dim=1)
    finite_tensor(force, (n, 3), "support force")
    finite_tensor(torque, (n, 3), "support torque")
    return force, torque


def stable_vector_norm(value):
    """Float64 max-scaled norm avoids squaring large finite components."""
    wide = value.to(dtype=torch.float64)
    scale = wide.abs().amax(dim=-1, keepdim=True)
    scaled = wide / torch.where(scale > 0, scale, torch.ones_like(scale))
    return scale.squeeze(-1) * torch.linalg.vector_norm(scaled, dim=-1)


def bounded_pd_torque(requested):
    """SIM-only radial PD saturation; not a total root-wrench limit."""
    finite_tensor(requested, (requested.shape[0], 3), "requested PD torque")
    requested_norm = stable_vector_norm(requested)
    saturated = requested_norm > 1.0
    wide = requested.to(dtype=torch.float64)
    scale = wide.abs().amax(dim=-1, keepdim=True)
    scaled = wide / torch.where(scale > 0, scale, torch.ones_like(scale))
    length = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    unit = scaled / torch.where(length > 0, length, torch.ones_like(length))
    # Round saturated vectors inward so casting cannot exceed the 1 Nm bound.
    bounded = (unit * (1.0 - torch.finfo(requested.dtype).eps)).to(dtype=requested.dtype)
    applied = torch.where(saturated.unsqueeze(-1), bounded, requested)
    return applied, {"requested": requested, "applied": applied,
                     "requested_norm": requested_norm, "applied_norm": stable_vector_norm(applied),
                     "saturated": saturated}


def guarded_native_pd(state, origins, target_pos, target_quat, gains=GAINS, profile="root_supported_v1", diagnostics=None):
    """Native root_state_w LINK pose + root-body COM velocity damping, no FF."""
    n = state.shape[0]
    for value, shape, label in ((state, (n, 13), "root state"), (origins, (n, 3), "env origins"),
                                (target_pos, (n, 3), "root target position"), (target_quat, (n, 4), "root target quaternion")):
        finite_tensor(value, shape, label)
        if value.device != state.device or value.dtype != state.dtype:
            raise ValueError("PD dtype/device mismatch")
    if profile not in PROFILE_GAINS or tuple(gains) != PROFILE_GAINS[profile] or not all(math.isfinite(v) for v in gains):
        raise ValueError("Unknown root profile or gains differ from fixed profile hypothesis")
    unit_quaternions(state[:, 3:7], "state quaternion")
    unit_quaternions(target_quat, "target quaternion")
    kp_pos, kd_pos, kp_rot, kd_rot = gains
    force = kp_pos * (target_pos - (state[:, :3] - origins)) - kd_pos * state[:, 7:10]
    error = quat_mul(target_quat, quat_conjugate(state[:, 3:7]))
    error = torch.where(error[..., 0:1] < 0, -error, error)
    torque = kp_rot * axis_angle_from_quat(error) - kd_rot * state[:, 10:13]
    finite_tensor(force, (n, 3), "PD force")
    finite_tensor(torque, (n, 3), "PD torque")
    if profile == "com_bounded_v3":
        if bool((torch.linalg.vector_norm(force, dim=-1) > 50.0).any()):
            raise ValueError("PD exceeds 50 N in at least one environment; abort, no force clipping/reset")
        torque, values = bounded_pd_torque(torque)
        if diagnostics is not None:
            diagnostics.update(values)
        return force, torque
    bad = (torch.linalg.vector_norm(force, dim=-1) > 50.0) | (torch.linalg.vector_norm(torque, dim=-1) > 1.0)
    if bool(bad.any()):
        raise ValueError("PD exceeds 50 N / 1 Nm in at least one environment; abort, no clipping/reset")
    return force, torque


def guard_joints(q, velocity, velocity_target, effort_target, limits):
    n = q.shape[0]
    for value, shape, label in ((q, (n, 20), "joint position"), (velocity, (n, 20), "joint velocity"),
                                (velocity_target, (n, 20), "joint velocity target"),
                                (effort_target, (n, 20), "joint effort target"), (limits, (n, 20, 2), "joint limits")):
        finite_tensor(value, shape, label)
    if bool((velocity_target != 0).any()) or bool((effort_target != 0).any()):
        raise ValueError("Joint velocity/effort targets must be exactly zero; no joint FF")
    # PhysX position limits can have numerical overshoot. Preserve native target
    # clamping and normal terminations, not a new live-position abort threshold.


def audit_startup(robot, env):
    """Read actual PhysX values once; fixed parent task has no mass/gain DR."""
    from regrind.robots.free_hand2_right_hand import expected_mass_contract, contract
    expected = expected_mass_contract()
    names = list(robot.body_names)
    masses_by_name = {**expected["physical_mass_kg"], **expected["query_frame_mass_kg"]}
    if len(names) != 33 or len(set(names)) != 33 or set(names) != set(masses_by_name):
        raise ValueError("Expected all 33 unique Robot body names")
    if len(robot.joint_names) != 20 or set(robot.joint_names) != set(contract.JOINT_NAMES):
        raise ValueError("Expected exactly 20 vendor joints")
    n = robot.num_instances
    device, dtype = robot.data.joint_pos.device, robot.data.joint_pos.dtype
    masses = robot.root_physx_view.get_masses().to(device=device, dtype=dtype).clone()
    finite_tensor(masses, (n, 33), "live masses")
    wanted = torch.tensor([masses_by_name[name] for name in names], device=device, dtype=dtype)
    tol = torch.tensor([5e-12 if name in expected["query_frame_mass_kg"] else 1e-8 + 1e-5 * masses_by_name[name]
                        for name in names], device=device, dtype=dtype)
    if not bool((torch.abs(masses - wanted) <= tol).all()):
        raise ValueError("Live Robot masses differ from vendor contract")
    gravity = torch.tensor(tuple(env.sim.physics_sim_view.get_gravity()), device=device, dtype=dtype)
    finite_tensor(gravity, (3,), "live scene gravity")
    if not torch.allclose(gravity, masses.new_tensor([0.0, 0.0, -9.81]), rtol=0, atol=1e-6):
        raise ValueError("Unexpected live scene gravity")
    if tuple(env.cfg.sim.gravity) != (0.0, 0.0, -9.81):
        raise ValueError("Unexpected configured gravity")
    flags = robot.root_physx_view.get_disable_gravities()
    finite_tensor(flags, (n, 33), "live gravity flags")
    if bool((flags != 0).any()):
        raise ValueError("Hand gravity must be enabled in every environment/body")
    parameters = [contract.PARAMETERS[name] for name in robot.joint_names]
    getters = {"kp": "get_dof_stiffnesses", "kv": "get_dof_dampings", "effort": "get_dof_max_forces",
               "velocity": "get_dof_max_velocities", "armature": "get_dof_armatures"}
    for key, getter in getters.items():
        actual = getattr(robot.root_physx_view, getter)().to(device=device, dtype=dtype)
        wanted = torch.tensor([p[key] for p in parameters], device=device, dtype=dtype).expand(n, -1)
        finite_tensor(actual, (n, 20), f"live {key}")
        if not torch.allclose(actual, wanted, rtol=2e-6, atol=1e-8):
            raise ValueError(f"Live vendor drive mismatch: {key}")
    limits = robot.root_physx_view.get_dof_limits().to(device=device, dtype=dtype).clone()
    wanted = torch.tensor([[p["lower"], p["upper"]] for p in parameters], device=device, dtype=dtype).expand(n, -1, -1)
    finite_tensor(limits, (n, 20, 2), "live limits")
    if not torch.allclose(limits, wanted, rtol=2e-6, atol=1e-8):
        raise ValueError("Live vendor limits mismatch")
    if set(robot.actuators) != {"fingers"} or robot.actuators["fingers"].is_implicit_model is not True:
        raise ValueError("Only official implicit fingers actuator is permitted")
    actuator = robot.actuators["fingers"]
    indices = actuator.joint_indices
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().tolist()  # startup only; GPU indices supported
    ids = list(range(20))[indices] if isinstance(indices, slice) else [int(i) for i in indices]
    if len(ids) != 20 or set(ids) != set(range(20)) or list(actuator.joint_names) != [robot.joint_names[i] for i in ids]:
        raise ValueError("Non-bijective fingers actuator")
    for field, key in (("stiffness", "kp"), ("damping", "kv"), ("effort_limit", "effort"), ("effort_limit_sim", "effort")):
        actual = getattr(actuator, field)
        wanted = torch.tensor([parameters[i][key] for i in ids], device=device, dtype=dtype).expand(n, -1)
        finite_tensor(actual, (n, 20), f"actuator {field}")
        if not torch.allclose(actual, wanted, rtol=2e-6, atol=1e-8):
            raise ValueError(f"Actuator vendor parameter mismatch: {field}")
    return masses, limits


class Hand2SupportedRootAction(SE3ImpedanceActionTerm):
    """Native six-dimensional residual action plus one combined Robot-root write."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.body_name != "r_mount" or self._asset.body_names[self._wrench_body_ids[0]] != "r_mount":
            raise ValueError("Support must act on named r_mount only")
        self._root_controller_profile = cfg.root_controller_profile
        if (self._root_controller_profile not in PROFILE_GAINS
                or tuple(getattr(self, key) for key in ("kp_pos", "kd_pos", "kp_rot", "kd_rot"))
                != PROFILE_GAINS[self._root_controller_profile]):
            raise ValueError("Unknown root profile or gains differ from fixed profile hypothesis")
        self._body_mass, self._limits = audit_startup(self._asset, env)
        self._gravity = self._body_mass.new_tensor([0.0, 0.0, -9.81])
        # Targets are None until native process_actions; never call PD in ctor.
        guard_joints(self._asset.data.joint_pos, self._asset.data.joint_vel,
                     self._asset.data.joint_vel_target, self._asset.data.joint_effort_target, self._limits)

    def process_actions(self, actions):
        finite_tensor(actions, (self.num_envs, 6), "root residual input")
        super().process_actions(actions)
        finite_tensor(self.target_pos, (self.num_envs, 3), "processed target")
        finite_tensor(self.target_quat, (self.num_envs, 4), "processed quaternion")
        if self._root_controller_profile == "com_bounded_v3":
            self._wrist_v3_stats = {key: self.target_pos.new_zeros((), dtype=torch.float64)
                                    for key in ("requested_max", "applied_max", "saturated_count", "combined_max")}
            self._wrist_v3_samples = 0

    def _publish_wrist_v3_log(self):
        stats = self._wrist_v3_stats
        self._env.extras.setdefault("log", {}).update({
            "WristV3/pd_requested_torque_max_nm": stats["requested_max"],
            "WristV3/pd_applied_torque_max_nm": stats["applied_max"],
            "WristV3/pd_torque_saturation_fraction": stats["saturated_count"] / self._wrist_v3_samples,
            "WristV3/combined_root_torque_max_nm": stats["combined_max"],
        })

    def reset(self, env_ids=None):
        super().reset(env_ids)
        # IsaacLab replaces extras['log'] before action_manager.reset at episode
        # end. Republish the completed control step, without resetting its stats.
        if self._root_controller_profile == "com_bounded_v3" and getattr(self, "_wrist_v3_samples", 0):
            self._publish_wrist_v3_log()

    def apply_actions(self):
        robot, n = self._asset, self.num_envs
        finite_tensor(self._env.action_manager.action, (n, 26), "full 6+20 action input")
        joint_term = self._env.action_manager.get_term("joint_pos")
        finite_tensor(joint_term.processed_actions, (n, 20), "finger residuals")
        finite_tensor(joint_term.get_base_joint_pos(), (n, 20), "finger motion targets")
        guard_joints(robot.data.joint_pos, robot.data.joint_vel,
                     robot.data.joint_vel_target, robot.data.joint_effort_target, self._limits)
        diagnostics = {} if self._root_controller_profile == "com_bounded_v3" else None
        pd = guarded_native_pd(robot.data.root_state_w, self._env.scene.env_origins,
                               self.target_pos, self.target_quat,
                               tuple(getattr(self, key) for key in ("kp_pos", "kd_pos", "kp_rot", "kd_rot")),
                               profile=self._root_controller_profile, diagnostics=diagnostics)
        point = robot.data.root_link_pos_w
        support = support_wrench(self._body_mass, robot.data.body_com_pos_w, point, self._gravity)
        force, torque = pd[0] + support[0], pd[1] + support[1]
        if self._root_controller_profile in ("com_supported_v2", "com_bounded_v3"):
            hand_com = (self._body_mass.unsqueeze(-1) * robot.data.body_com_pos_w).sum(dim=1) / self._body_mass.sum(dim=1, keepdim=True)
            finite_tensor(hand_com, (n, 3), "whole-hand COM")
            # Shift only the guarded PD force to the whole-hand COM; gravity
            # support stays a root-LINK wrench. The plus sign preserves moments.
            torque = torque + torch.cross(hand_com - point, pd[0], dim=-1)
        finite_tensor(force, (n, 3), "combined force")
        finite_tensor(torque, (n, 3), "combined torque")
        # One call only: a second external-wrench call would overwrite support.
        robot.set_external_force_and_torque(forces=force.unsqueeze(1), torques=torque.unsqueeze(1),
                                            positions=point.unsqueeze(1), body_ids=self._wrench_body_ids,
                                            is_global=True)
        if diagnostics is not None:
            self._wrist_v3_last_pd = diagnostics
            stats = self._wrist_v3_stats
            stats["requested_max"] = torch.maximum(stats["requested_max"], diagnostics["requested_norm"].max())
            stats["applied_max"] = torch.maximum(stats["applied_max"], diagnostics["applied_norm"].max())
            stats["saturated_count"] = stats["saturated_count"] + diagnostics["saturated"].sum()
            stats["combined_max"] = torch.maximum(stats["combined_max"], stable_vector_norm(torque).max())
            self._wrist_v3_samples += n  # N envs x actual physics calls (normally decimation=4).
            self._publish_wrist_v3_log()


@configclass
class Hand2SupportedRootActionCfg(SE3ImpedanceActionCfg):
    class_type: type = Hand2SupportedRootAction
    root_controller_profile: str = "root_supported_v1"
    kp_pos: float = GAINS[0]
    kd_pos: float = GAINS[1]
    kp_rot: float = GAINS[2]
    kd_rot: float = GAINS[3]
    body_name: str = "r_mount"
