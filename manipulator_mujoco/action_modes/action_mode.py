from abc import ABC, abstractmethod
import numpy as np


class ActionMode(ABC):

    @abstractmethod
    def compute_target_pose_from_action(self, environment, action: np.ndarray):
        pass

    @abstractmethod
    def get_action_space(self):
        pass