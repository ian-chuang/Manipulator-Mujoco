from manipulator_mujoco.controllers import JointEffortController

import numpy as np

from manipulator_mujoco.utils.controller_utils import (
    task_space_inertia_matrix,
    pose_error,
)

from manipulator_mujoco.utils.mujoco_utils import (
    get_site_jac, 
    get_fullM
)

from manipulator_mujoco.utils.transform_utils import (
    mat2quat,
)

class OperationalSpaceController(JointEffortController):
    def __init__(
        self,
        physics,
        joints,
        eef_site,
        min_effort: np.ndarray,
        max_effort: np.ndarray,
        kp: float,
        ko: float,
        kv: float,
        vmax_xyz: float,
        vmax_abg: float,
    ) -> None:
        
        super().__init__(physics, joints, min_effort, max_effort)

        self._eef_site = eef_site
        self._kp = kp
        self._ko = ko
        self._kv = kv
        self._vmax_xyz = vmax_xyz
        self._vmax_abg = vmax_abg

        self._eef_id = self._physics.bind(eef_site).element_id
        self._jnt_dof_ids = self._physics.bind(joints).dofadr
        self._dof = len(self._jnt_dof_ids)

        self._task_space_gains = np.array([self._kp] * 3 + [self._ko] * 3)
        self._lamb = self._task_space_gains / self._kv
        self._sat_gain_xyz = vmax_xyz / self._kp * self._kv
        self._sat_gain_abg = vmax_abg / self._ko * self._kv
        self._scale_xyz = vmax_xyz / self._kp * self._kv
        self._scale_abg = vmax_abg / self._ko * self._kv

    def run(self, target):
        # target pose is a 7D vector [x, y, z, qx, qy, qz, qw]
        target_pose = target

        # Get the Jacobian matrix for the end-effector.
        J = get_site_jac(
            self._physics.model.ptr, 
            self._physics.data.ptr, 
            self._eef_id,
        )
        J = J[:, self._jnt_dof_ids]

        # Get the mass matrix and its inverse for the controlled degrees of freedom (DOF) of the robot.
        M_full = get_fullM(
            self._physics.model.ptr, 
            self._physics.data.ptr,
        )
        M = M_full[self._jnt_dof_ids, :][:, self._jnt_dof_ids]
        Mx, M_inv = task_space_inertia_matrix(M, J)

        # Get the joint velocities for the controlled DOF.
        dq = self._physics.bind(self._joints).qvel

        # Get the end-effector position, orientation matrix, and twist (spatial velocity).
        ee_pos = self._physics.bind(self._eef_site).xpos
        ee_quat = mat2quat(self._physics.bind(self._eef_site).xmat.reshape(3, 3))
        ee_pose = np.concatenate([ee_pos, ee_quat])

        # Calculate the pose error (difference between the target and current pose).
        pose_err = pose_error(target_pose, ee_pose)

        # Initialize the task space control signal (desired end-effector motion).
        u_task = np.zeros(6)

        # Calculate the task space control signal.
        u_task += self._scale_signal_vel_limited(pose_err)

        # joint space control signal
        u = np.zeros(self._dof)
        
        # Add the task space control signal to the joint space control signal
        u += np.dot(J.T, np.dot(Mx, u_task))

        # Add damping to joint space control signal
        u += -self._kv * np.dot(M, dq)

        # Add gravity compensation to the target effort
        u += self._physics.bind(self._joints).qfrc_bias

        # send the target effort to the joint effort controller
        super().run(u)

    def _scale_signal_vel_limited(self, u_task: np.ndarray) -> np.ndarray:
        """
        Scale the control signal such that the arm isn't driven to move faster in position or orientation than the specified vmax values.

        Parameters:
            u_task (numpy.ndarray): The task space control signal.

        Returns:
            numpy.ndarray: The scaled task space control signal.
        """
        norm_xyz = np.linalg.norm(u_task[:3])
        norm_abg = np.linalg.norm(u_task[3:])
        scale = np.ones(6)
        if norm_xyz > self._sat_gain_xyz:
            scale[:3] *= self._scale_xyz / norm_xyz
        if norm_abg > self._sat_gain_abg:
            scale[3:] *= self._scale_abg / norm_abg

        return self._kv * scale * self._lamb * u_task

