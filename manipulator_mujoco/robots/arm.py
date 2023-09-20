from dm_control import mjcf
from manipulator_mujoco.utils.transform_utils import (
    mat2quat
)
import numpy as np

class Arm():
    def __init__(self, xml_path, eef_site_name, attachment_site_name, joint_names = None, name: str = None):
        self._mjcf_root = mjcf.from_path(xml_path)
        if name:
            self._mjcf_root.model = name

        # Find MJCF elements that will be exposed as attributes.
        if joint_names is None:
            self._joints = self.mjcf_model.find_all('joint')
        else:
            self._joints = [self._mjcf_root.find('joint', name) for name in joint_names]
        
        self._eef_site = self._mjcf_root.find('site', eef_site_name)
        self._attachment_site = self._mjcf_root.find('site', attachment_site_name)

    @property
    def joints(self):
        """List of joint elements belonging to the arm."""
        return self._joints

    @property
    def eef_site(self):
        """Wrist site of the arm (attachment point for the hand)."""
        return self._eef_site

    @property
    def mjcf_model(self):
        """Returns the `mjcf.RootElement` object corresponding to this robot."""
        return self._mjcf_root
    
    def attach_tool(self, child, pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]) -> mjcf.Element:
        frame = self._attachment_site.attach(child)
        frame.pos = pos
        frame.quat = quat
        return frame
    
    def get_eef_pose(self, physics):
        ee_pos = physics.bind(self._eef_site).xpos
        ee_quat = mat2quat(physics.bind(self._eef_site).xmat.reshape(3, 3))
        ee_pose = np.concatenate((ee_pos, ee_quat))
        return ee_pose