import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from suika_gym import SuikaEnv
from agents.base_agent import Agent  # Import the base Agent class


class RandomAgent(Agent):
    """
    An agent that selects actions randomly from the action space.
    """

    def __init__(self, action_space):
        """
        Initializes the RandomAgent.

        Args:
            action_space: The action space of the environment.
        """
        super().__init__()  # Call to base class constructor
        self.action_space = action_space

    def select_action(self, obs):
        """
        Selects a random action.

        Args:
            obs: The current observation (ignored by this agent).

        Returns:
            A randomly sampled action.
        """
        return self.action_space.sample()


# The following block demonstrates how to run the RandomAgent.
if __name__ == "__main__":
    print("Running RandomAgent example...")
    # Use render_mode="human" to visualize, or None for no rendering
    env = SuikaEnv(render_mode="human")

    obs, info = env.reset()
    agent = RandomAgent(env.action_space)

    done = False
    total_reward = 0
    episode_length = 0

    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_length += 1
        done = terminated or truncated

    print(f"RandomAgent episode finished.")
    print(f"Total Reward: {total_reward}")
    print(f"Episode Length: {episode_length}")

    env.close()
