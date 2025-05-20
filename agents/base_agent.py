from abc import ABC, abstractmethod


class Agent(ABC):
    """
    Abstract base class for all agents.
    """

    @abstractmethod
    def select_action(self, obs):
        """
        Selects an action given the current observation.

        Args:
            obs: The current observation from the environment.

        Returns:
            The action to take.
        """
        pass
