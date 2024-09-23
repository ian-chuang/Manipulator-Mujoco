import numpy as np
from gymnasium import spaces

from manipulator_mujoco.action_modes.action_mode import ActionMode
from manipulator_mujoco.utils.transform_utils import euler2quat, quat_multiply


class DeltaEndEffectorPose(ActionMode):
    def compute_target_pose_from_action(self, environment, action: np.ndarray):
        """
        Computes the target end-effector pose based on the given action.

        Parameters:
        environment (object): The environment object containing the arm and physics.
        action (np.ndarray): A numpy array containing the delta position and delta angles for the end-effector.

        Returns:
        np.ndarray: The target pose of the end-effector as a concatenated array of target position and target quaternion.
        """
        eef_pose = environment._arm.get_eef_pose(environment._physics)
        eef_position = eef_pose[:3]
        eef_quat = eef_pose[3:]

        delta_position = action[:3]
        delta_angles = action[3:6]

        target_position = eef_position + delta_position
        target_quat = quat_multiply(eef_quat, euler2quat(delta_angles))

        target_pose = np.concatenate([target_position, target_quat])

        return target_pose

    def get_action_space(self):
        return spaces.Box(
            low=-0.1, high=0.1, shape=(6,), dtype=np.float64
        )


