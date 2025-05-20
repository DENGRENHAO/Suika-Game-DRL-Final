import sys
import os
import importlib.util
import inspect
import numpy as np
import time
import traceback
import argparse  # Added for command-line argument parsing

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from suika_gym import SuikaEnv
from agents.base_agent import Agent  # Import the base Agent class

# Files to exclude from agent discovery (add script's own name dynamically later)
EXCLUDE_FILES_DEFAULT = ["__init__.py", "base_agent.py"]


def discover_agents(agents_dir, exclude_files):
    """
    Discovers agent classes in the specified directory.
    An agent class must inherit from agents.base_agent.Agent.
    """
    discovered_agents = {}  # Store as {class_name: class_object}
    print(f"Searching for agents in: {agents_dir}")
    for filename in os.listdir(agents_dir):
        if filename.endswith(".py") and filename not in exclude_files:
            module_name = filename[:-3]  # remove .py
            file_path = os.path.join(agents_dir, filename)

            try:
                # Dynamically import the module
                spec = importlib.util.spec_from_file_location(
                    f"agents.{module_name}", file_path
                )
                if spec is None or spec.loader is None:
                    print(
                        f"Warning: Could not create spec for module {module_name} at {file_path}"
                    )
                    continue

                module = importlib.util.module_from_spec(spec)
                # Add to sys.modules before exec_module to handle potential relative imports within agent files
                sys.modules[f"agents.{module_name}"] = module
                spec.loader.exec_module(module)

                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, Agent)
                        and obj is not Agent
                    ):  # Ensure it's a subclass and not Agent itself
                        if name in discovered_agents:
                            print(
                                f"Warning: Agent class name '{name}' from {filename} conflicts with an already discovered agent from {discovered_agents[name].__module__}. Skipping {filename}."
                            )
                        else:
                            discovered_agents[name] = obj
                            print(f"  Discovered agent class: {name} from {filename}")
            except ImportError as e:
                print(
                    f"Warning: Could not import module {module_name} from {file_path}. Error: {e}"
                )
            except Exception as e:
                print(f"Warning: Error processing file {file_path}. Error: {e}")
                # traceback.print_exc() # Uncomment for more detailed debugging
    return discovered_agents


def run_evaluation_for_agent(
    agent_class,
    agent_name,
    num_episodes=20,
    env_render_mode=None,
    max_steps_per_episode=1000,
):
    """
    Evaluates a single agent class.
    """
    print(f"\n--- Evaluating: {agent_name} ---")

    try:
        env = SuikaEnv(render_mode=env_render_mode)
    except Exception as e:
        print(f"  Error initializing environment for {agent_name}: {e}")
        return None, None

    agent_instance = None
    try:
        # Inspect agent's __init__ to provide necessary env spaces if needed
        init_signature = inspect.signature(agent_class.__init__)
        init_params = {}
        if "action_space" in init_signature.parameters:
            init_params["action_space"] = env.action_space
        if "observation_space" in init_signature.parameters:
            init_params["observation_space"] = env.observation_space

        agent_instance = agent_class(**init_params)
        print(
            f"  Successfully instantiated {agent_name} with auto-provided params: {list(init_params.keys()) if init_params else 'None'}"
        )

    except TypeError as e:
        print(f"  Error instantiating {agent_name}: {e}")
        print(
            f"    Ensure '{agent_name}' constructor can be called with (action_space, observation_space) if needed, or with no arguments."
        )
        env.close()
        return None, None
    except Exception as e:
        print(f"  An unexpected error occurred during {agent_name} instantiation: {e}")
        traceback.print_exc()
        env.close()
        return None, None

    episode_rewards = []
    episode_lengths = []
    start_time = time.time()

    for episode in range(num_episodes):
        try:
            reset_result = env.reset()
            if (
                isinstance(reset_result, tuple)
                and len(reset_result) == 2
                and isinstance(reset_result[1], dict)
            ):
                obs, info = reset_result
            else:
                obs = reset_result

            done = False
            total_reward = 0
            current_episode_length = 0

            while (
                not done and current_episode_length < max_steps_per_episode
            ):  # Added max_steps_per_episode condition
                action = agent_instance.select_action(obs)
                obs_next, reward, terminated, truncated, info = env.step(action)

                total_reward += reward
                current_episode_length += 1
                done = terminated or truncated

                if hasattr(agent_instance, "learn") and callable(
                    getattr(agent_instance, "learn")
                ):
                    agent_instance.learn(obs, action, reward, obs_next, done)

                obs = obs_next

            if hasattr(agent_instance, "episode_end") and callable(
                getattr(agent_instance, "episode_end")
            ):
                agent_instance.episode_end()

            episode_rewards.append(total_reward)
            episode_lengths.append(current_episode_length)

            if (episode + 1) % max(
                1, num_episodes // 5
            ) == 0 or episode == num_episodes - 1:
                print(
                    f"    Episode {episode + 1}/{num_episodes} | Score: {total_reward:.2f} | Length: {current_episode_length}"
                )
            if current_episode_length >= max_steps_per_episode:
                print(
                    f"    Episode {episode + 1} for {agent_name} reached max steps ({max_steps_per_episode})."
                )

        except Exception as e:
            print(f"  Error during episode {episode+1} for {agent_name}: {e}")
            traceback.print_exc()
            episode_rewards.append(float("nan"))  # Mark as failed for stats
            episode_lengths.append(0)
            # break # Option: stop evaluating this agent on first error in an episode

    end_time = time.time()
    total_time = end_time - start_time

    print(
        f"  Evaluation for {agent_name} completed in {total_time:.2f}s ({total_time/num_episodes:.3f}s/ep)."
    )
    env.close()
    return episode_rewards, episode_lengths


def main():
    parser = argparse.ArgumentParser(description="Evaluate Suika Game agents.")
    parser.add_argument(
        "--agents",
        nargs="+",
        help="Specify one or more agent class names to evaluate (e.g., RandomAgent MyCustomAgent). If not set, all discoverable agents are evaluated.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes to run for each agent.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1500,  # Increased default a bit
        help="Maximum number of steps per episode.",
    )
    args = parser.parse_args()

    agents_dir = os.path.join(PROJECT_ROOT, "agents")
    exclude_files = EXCLUDE_FILES_DEFAULT
    discovered_agent_classes = discover_agents(agents_dir, exclude_files)

    if not discovered_agent_classes:
        print("No agent classes found to evaluate.")
        return

    agents_to_evaluate = {}
    if args.agents:
        for agent_name_to_find in args.agents:
            if agent_name_to_find in discovered_agent_classes:
                agents_to_evaluate[agent_name_to_find] = discovered_agent_classes[
                    agent_name_to_find
                ]
            else:
                print(
                    f"Warning: Specified agent '{agent_name_to_find}' not found among discovered agents. Skipping."
                )
        if not agents_to_evaluate:
            print("None of the specified agents were found. Exiting.")
            return
    else:
        agents_to_evaluate = discovered_agent_classes

    all_agent_stats = {}
    for agent_name, agent_class in agents_to_evaluate.items():
        rewards, lengths = run_evaluation_for_agent(
            agent_class,
            agent_name,
            num_episodes=args.episodes,
            env_render_mode=None,  # Use None for faster evaluation; "human" for visualization
            max_steps_per_episode=args.max_steps,
        )
        if rewards is not None:  # Check if evaluation ran
            valid_rewards = [r for r in rewards if not np.isnan(r)]
            valid_lengths = [
                l for i, l in enumerate(lengths) if not np.isnan(rewards[i])
            ]
            all_agent_stats[agent_name] = {
                "rewards": valid_rewards,
                "lengths": valid_lengths,
                "num_episodes_run": len(rewards),  # Total attempted episodes
            }

    print("\n\n" + "=" * 15 + " Evaluation Summary " + "=" * 15)
    if not all_agent_stats:
        print("No agents were successfully evaluated.")
        return

    for agent_name, stats in all_agent_stats.items():
        rewards = stats["rewards"]
        lengths = stats["lengths"]
        print(f"\nAgent: {agent_name}")
        if rewards:
            print(
                f"  Episodes Evaluated (Successful): {len(rewards)} / {stats['num_episodes_run']}"
            )
            print(f"  Average Score: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
            print(f"  Min Score: {np.min(rewards):.2f}")
            print(f"  Max Score: {np.max(rewards):.2f}")
            print(f"  Average Episode Length: {np.mean(lengths):.2f} steps")
        else:
            print(
                f"  No successful episodes recorded out of {stats['num_episodes_run']} attempted."
            )
    print("=" * 50)


if __name__ == "__main__":
    main()
