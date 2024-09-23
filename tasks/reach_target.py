from typing import List

import numpy as np
from manipulator_mujoco.arenas.standard import StandardArena
from manipulator_mujoco.robots.arm import Arm
from tasks.task import Task
from dm_control import mjcf


class ReachTarget(Task):
    def __init__(self, arena: StandardArena, arm: Arm):
        super().__init__(arena, arm)

        # Define task scene
        self.target = self.arena.mjcf_model.worldbody.add("geom", type="sphere", size=[0.01], pos=[0.5, 0.2, 0.2])

    def get_waypoints(self) -> List[np.array]:
        return [np.array([0.5, 0.2, 0.2, 0, 0, 0, 1])]

    def init_task(self):
        pass

    def success(self, physics: mjcf.Physics) -> bool:
        target_position = self.target.pos
        eef_position = self.arm.get_eef_pose(physics)[:3]

        distance = np.sqrt(((target_position - eef_position) ** 2).sum())

        print(distance)

        return distance < 0.01