import gymnasium
import manipulator_mujoco

# Create the environment with rendering in human mode
env = gymnasium.make('manipulator_mujoco/AuboI5Env-v0', render_mode='human')

# Reset the environment with a specific seed for reproducibility
observation, info = env.reset(seed=42)

# Run simulation for a fixed number of steps
# for _ in range(1000):
while True:
    # Choose a random action from the available action space
    action = env.action_space.sample()

    # Take a step in the environment using the chosen action
    observation, reward, terminated, truncated, info = env.step(action)

    # Check if the episode is over (terminated) or max steps reached (truncated)
    if terminated or truncated:
        # If the episode ends or is truncated, reset the environment
        observation, info = env.reset()

# Close the environment when the simulation is done
env.close()
