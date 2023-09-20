# Import necessary modules and classes
import numpy as np

class JointEffortController:
    def __init__(
        self,
        physics,
        joints,
        min_effort: np.ndarray,
        max_effort: np.ndarray,
    ) -> None:

        self._physics = physics
        self._joints = joints
        self._min_effort = min_effort
        self._max_effort = max_effort

    def run(self, target) -> None:
        """
        Run the robot controller.

        Parameters:
            target (numpy.ndarray): The desired target joint positions or states for the robot.
                                   The size of `target` should be (n_joints,) where n_joints is the number of robot joints.
            ctrl (numpy.ndarray): Control signals for the robot actuators from `mujoco._structs.MjData.ctrl` of size (nu,).
        """

        # Clip the target efforts to ensure they are within the allowable effort range
        target_effort = np.clip(target, self._min_effort, self._max_effort)

        # Set the control signals for the actuators to the desired target joint positions or states
        self._physics.bind(self._joints).qfrc_applied = target_effort

    def reset(self) -> None:
        pass