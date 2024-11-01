from abc import ABC, abstractmethod
from typing import List

import numpy as np
from dm_control import mjcf

from manipulator_mujoco.arenas.standard import StandardArena
from manipulator_mujoco.robots.arm import Arm


class Task(ABC):

    # TODO: I don't like the fact that arm and physics is passed here. It is needed to compute end effector position.
    def __init__(self, arena: StandardArena, arm: Arm):
        self.arena = arena
        self.arm = arm

    @abstractmethod
    def get_waypoints(self) -> List[np.array]:
        raise NotImplementedError()

    @abstractmethod
    def init_task(self):
        """Initialize task.

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError()

    @abstractmethod
    def success(self, physics: mjcf.Physics) -> bool:
        """If the task is currently successful."""
        raise NotImplementedError()
