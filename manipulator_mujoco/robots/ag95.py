import os
from manipulator_mujoco.robots.gripper import Gripper

_AG95_XML = os.path.join(
    os.path.dirname(__file__),
    '../assets/robots/ag95/ag95.xml',
)

_JOINT = 'left_outer_knuckle_joint'

_ACTUATOR = 'fingers_actuator'

class AG95(Gripper):
    def __init__(self, name: str = None):
        super().__init__(_AG95_XML, _JOINT, _ACTUATOR, name)