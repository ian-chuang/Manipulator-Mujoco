import time
import os
import numpy as np
from dm_control import mjcf
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from manipulator_mujoco.arenas import StandardArena
from manipulator_mujoco.robots import Arm
from manipulator_mujoco.mocaps import Target
from manipulator_mujoco.controllers import OperationalSpaceController
from manipulator_mujoco.utils.transform_utils import euler2quat, quat_multiply

class UR5eActionEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,
    }  # TODO add functionality to render_fps

    def __init__(self, render_mode=None):
        # TODO come up with an observation space that makes sense
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64
        )

        # TODO come up with an action space that makes sense
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(6,), dtype=np.float64
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._render_mode = render_mode

        ############################
        # create MJCF model
        ############################
        
        # checkerboard floor
        self._arena = StandardArena()

        # mocap target that OSC will try to follow
        self._target = Target(self._arena.mjcf_model)

        # ur5e arm
        self._arm = Arm(
            xml_path= os.path.join(
                os.path.dirname(__file__),
                '../assets/robots/ur5e/ur5e.xml',
            ),
            eef_site_name='eef_site',
            attachment_site_name='attachment_site'
        )

        # attach arm to arena
        self._arena.attach(
            self._arm.mjcf_model, pos=[0,0,0], quat=[0.7071068, 0, 0, -0.7071068]
        )
       
        # Add front camera
        # self._arena.mjcf_model.worldbody.add(
        #     "camera",
        #     name="front_camera",
        #     pos=[0, -2, 1],  # Position in front of the robot
        #     quat=[1, 0, 0, 0],  # Default orientation
        #     mode="fixed"
        # )

        # # Add wrist camera (attached to the end-effector)
        # self._arm.mjcf_model.worldbody.add(
        #     "camera",
        #     name="wrist_camera",
        #     pos=[0, 0, 0],  # Position at the end-effector
        #     mode="fixed"
        # )

        # generate model
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        # set up OSC controller
        self._controller = OperationalSpaceController(
            physics=self._physics,
            joints=self._arm.joints,
            eef_site=self._arm.eef_site,
            min_effort=-150.0,
            max_effort=150.0,
            kp=200,
            ko=200,
            kv=50,
            vmax_xyz=1.0,
            vmax_abg=2.0,
        )

        # for GUI and time keeping
        self._timestep = self._physics.model.opt.timestep
        self._viewer = None
        self._step_start = None

    def _get_obs(self) -> np.ndarray:
        """
        Returns a dictionary containing images from the front camera and wrist camera.
        """
        # # Image size for the cameras
        # width, height = 128, 128

        # # Render the image from the front camera
        # front_camera_image = self._physics.render(
        #     width=width, height=height, camera_id=self._physics.model.name2id('front_camera', 'camera')
        # )

        # # # Render the image from the wrist camera
        # # wrist_camera_image = self._physics.render(
        # #     width=width, height=height, camera_id=self._physics.model.name2id('wrist_camera', 'camera')
        # # )

        # # Return images as a dictionary
        # return {
        #     "front_camera": front_camera_image,
        #     # "wrist_camera": wrist_camera_image
        # }
        pass

    def _get_info(self) -> dict:
        # TODO come up with an info dict that makes sense for your RL task
        return {}

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)

        # reset physics
        with self._physics.reset_context():
            # put arm in a reasonable starting position
            self._physics.bind(self._arm.joints).qpos = [
                0.0,
                -1.5707,
                1.5707,
                -1.5707,
                -1.5707,
                0.0,
            ]
            # put target in a reasonable starting position
            self._target.set_mocap_pose(self._physics, position=[0.5, 0, 0.3], quaternion=[0, 0, 0, 1])

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> tuple:
        eef_pose = self._arm.get_eef_pose(self._physics)
        eef_position = eef_pose[:3]
        eef_quat = eef_pose[3:]

        delta_position = action[:3]
        delta_angles = action[3:6]

        target_position = eef_position + delta_position
        target_quat = quat_multiply(eef_quat, euler2quat(delta_angles))

        # get mocap target pose
        target_pose = np.concatenate([target_position, target_quat])

        # run OSC controller to move to target pose
        self._controller.run(target_pose)

        # step physics
        self._physics.step()

        print(self._arm.get_eef_pose(self._physics))

        # render frame
        if self._render_mode == "human":
            self._render_frame()
        
        # TODO come up with a reward, termination function that makes sense for your RL task
        observation = self._get_obs()
        reward = 0
        terminated = False
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self) -> np.ndarray:
        """
        Renders the current frame and returns it as an RGB array if the render mode is set to "rgb_array".

        Returns:
            np.ndarray: RGB array of the current frame.
        """
        if self._render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> None:
        """
        Renders the current frame and updates the viewer if the render mode is set to "human".
        """
        if self._viewer is None and self._render_mode == "human":
            # launch viewer
            self._viewer = mujoco.viewer.launch_passive(
                self._physics.model.ptr,
                self._physics.data.ptr,
            )
        if self._step_start is None and self._render_mode == "human":
            # initialize step timer
            self._step_start = time.time()

        if self._render_mode == "human":
            # render viewer
            self._viewer.sync()

            # TODO come up with a better frame rate keeping strategy
            time_until_next_step = self._timestep - (time.time() - self._step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            self._step_start = time.time()

        else:  # rgb_array
            return self._physics.render()

    def close(self) -> None:
        """
        Closes the viewer if it's open.
        """
        if self._viewer is not None:
            self._viewer.close()