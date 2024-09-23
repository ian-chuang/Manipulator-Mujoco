import numpy as np
from gymnasium import spaces

from manipulator_mujoco.action_modes.action_mode import ActionMode
from manipulator_mujoco.utils.transform_utils import euler2quat, quat_multiply


class AbsoluteEndEffectorPose(ActionMode):
    def compute_target_pose_from_action(self, environment, action: np.ndarray):
        return action

    def get_action_space(self):
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
        )


