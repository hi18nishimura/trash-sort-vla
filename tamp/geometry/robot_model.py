"""Robot kinematics wrapper around pykin (FK + Jacobian) with a robust
damped-least-squares (DLS) IK implemented here.

Why our own IK: pykin's bundled IK relies on ``get_quaternion_from_matrix``,
which returns NaN on Python 3.12 / NumPy 1.26, so it performs zero iterations.
pykin's ``forward_kin`` and ``jacobian.calc_jacobian`` are correct, so we drive
a standard DLS loop on top of them. This is the same feasibility primitive
PyTAMP's geometric layer provides (IK existence + joint-limit check).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import tamp.geometry._compat  # noqa: F401  (must precede pykin import)

import numpy as np
from pykin.kinematics import jacobian as _jac
from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm

# Default = pykin's bundled Panda (matches the reference repo's Franka arm).
PANDA_URDF = "urdf/panda/panda.urdf"
PANDA_BASE = "panda_link_0"
PANDA_EEF = "right_gripper"


@dataclass
class IKResult:
    success: bool
    q: np.ndarray
    iters: int
    error: float


class RobotModel:
    def __init__(self, urdf: str = PANDA_URDF, base: str = PANDA_BASE,
                 eef: str = PANDA_EEF, base_pose: Optional[np.ndarray] = None):
        offset = Transform()
        self.robot = SingleArm(urdf, offset=offset)
        self.robot.setup_link_name(base, eef)
        self.eef = self.robot.eef_name
        self.active_joints = list(self.robot.active_joint_names)
        self.dof = len(self.active_joints)
        jl = self.robot.joint_limits
        self.lower = np.array([jl[j][0] for j in self.active_joints])
        self.upper = np.array([jl[j][1] for j in self.active_joints])
        self.home = np.clip(np.zeros(self.dof), self.lower, self.upper)
        # base_pose: 4x4 placing the robot base in world (humanoid / table offset)
        self.base_pose = np.eye(4) if base_pose is None else np.asarray(base_pose)
        self._base_inv = np.linalg.inv(self.base_pose)

    # --- kinematics ---
    def fk(self, q) -> np.ndarray:
        """End-effector 4x4 transform in world frame for joint vector q."""
        fk = self.robot.forward_kin(np.asarray(q, float))
        return self.base_pose @ fk[self.eef].h_mat

    def link_positions(self, q) -> dict:
        """World-frame origin of every link for joint vector q (coarse per-link
        collision checks in the refiner — see collision.arm_links_free)."""
        fk = self.robot.forward_kin(np.asarray(q, float))
        return {name: (self.base_pose @ frame.h_mat)[:3, 3]
                for name, frame in fk.items()}

    def jacobian(self, q) -> np.ndarray:
        fk = self.robot.forward_kin(np.asarray(q, float))
        return np.asarray(_jac.calc_jacobian(self.robot.desired_frames, fk, len(q)))

    # --- IK (damped least squares, world-frame target) ---
    def solve_ik(self, target_world: np.ndarray, q0=None, iters: int = 200,
                 lam: float = 0.1, pos_tol: float = 1e-3, rot_tol: float = 5e-2,
                 position_only: bool = False, restarts: int = 12, seed: int = 0) -> IKResult:
        """DLS IK with joint-limit clamping and random restarts.

        A solution must both converge to the target *and* lie within joint
        limits (joint-limit violation is the most common pykin/DLS failure).
        """
        target = self._base_inv @ np.asarray(target_world)  # to base frame
        rng = np.random.default_rng(seed)
        seeds = [self.home.copy() if q0 is None else np.array(q0, float)]
        seeds += [rng.uniform(self.lower, self.upper) for _ in range(restarts)]
        best = IKResult(False, self.home.copy(), 0, np.inf)
        for q in seeds:
            res = self._dls(target, np.clip(q, self.lower, self.upper),
                            iters, lam, pos_tol, rot_tol, position_only)
            if res.success:
                return res
            if res.error < best.error:
                best = res
        return best

    def _dls(self, target, q, iters, lam, pos_tol, rot_tol, position_only) -> IKResult:
        err = np.inf
        for k in range(iters):
            fk = self.robot.forward_kin(q)
            T = fk[self.eef].h_mat
            pe = target[:3, 3] - T[:3, 3]
            if position_only:
                e = pe
                J = self.jacobian(q)[:3]
            else:
                oe = _rotation_error(target[:3, :3], T[:3, :3])
                e = np.concatenate([pe, oe])
                J = self.jacobian(q)
            err = float(np.linalg.norm(e))
            converged = (np.linalg.norm(pe) < pos_tol and
                         (position_only or np.linalg.norm(e[3:]) < rot_tol))
            if converged:
                return IKResult(self._within_limits(q), q, k, err)
            JT = J.T
            dq = JT @ np.linalg.solve(J @ JT + (lam ** 2) * np.eye(J.shape[0]), e)
            q = np.clip(q + dq, self.lower, self.upper)
        return IKResult(False, q, iters, err)

    def _wrap(self, q):
        return np.clip(q, self.lower, self.upper)

    def _within_limits(self, q) -> bool:
        return bool(np.all(q >= self.lower - 1e-6) and np.all(q <= self.upper + 1e-6))


def _rotation_error(Rd: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Orientation error as a rotation vector (axis*angle) from R to Rd."""
    Re = Rd @ R.T
    ang = np.arccos(np.clip((np.trace(Re) - 1) / 2, -1.0, 1.0))
    if abs(ang) < 1e-9:
        return np.zeros(3)
    return (ang / (2 * np.sin(ang))) * np.array(
        [Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
