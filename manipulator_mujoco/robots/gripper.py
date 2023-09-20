from dm_control import mjcf

class Gripper():
    def __init__(self, xml_path, joint_name, actuator_name, name: str = None):
        self._mjcf_root = mjcf.from_path(xml_path)
        if name:
            self._mjcf_root.model = name
        # Find MJCF elements that will be exposed as attributes.
        self._joint = self._mjcf_root.find('joint', joint_name)
        self._bodies = self.mjcf_model.find_all('body')
        self._actuator = self._mjcf_root.find('actuator', actuator_name)

    @property
    def joint(self):
        """List of joint elements belonging to the arm."""
        return self._joint

    @property
    def actuator(self):
        """List of actuator elements belonging to the arm."""
        return self._actuator

    @property
    def mjcf_model(self):
        """Returns the `mjcf.RootElement` object corresponding to this robot."""
        return self._mjcf_root