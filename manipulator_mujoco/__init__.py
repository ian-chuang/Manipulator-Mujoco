# Import the registration function from Gymnasium
from gymnasium.envs.registration import register

from manipulator_mujoco.envs.ur5e_env_with_task import UR5eEnvWithTask
from tasks.reach_target import ReachTarget

register(
    id="manipulator_mujoco/AuboI5Env-v0",
    entry_point="manipulator_mujoco.envs:AuboI5Env",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)

register(
    id="manipulator_mujoco/UR5eEnv-v0",
    entry_point="manipulator_mujoco.envs:UR5eEnv",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)

register(
    id="manipulator_mujoco/UR5eEnv-v1",
    entry_point="manipulator_mujoco.envs:UR5eActionEnv",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)

register(
    id="manipulator_mujoco/UR5eEnv-task",
    entry_point="manipulator_mujoco.envs:UR5eEnvWithTask",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)