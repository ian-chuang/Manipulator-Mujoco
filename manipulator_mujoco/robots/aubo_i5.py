import os
from manipulator_mujoco.robots.arm import Arm

_AUBOI5_XML = os.path.join(
    os.path.dirname(__file__),
    '../assets/robots/aubo_i5/aubo_i5.xml',
)

_JOINTS = (
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
)

_EEF_SITE = 'eef_site'

_ATTACHMENT_SITE = 'attachment_site'

class AuboI5(Arm):
    def __init__(self, name: str = None):
        super().__init__(_AUBOI5_XML, _EEF_SITE, _ATTACHMENT_SITE, _JOINTS, name)