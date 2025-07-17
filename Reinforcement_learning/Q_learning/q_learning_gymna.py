from typing import Dict, List, Tuple
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from matplotlib import pyplot as plt

class FrozenLakeEnv:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95,
    ):
        """
        Initialize a Q-learning agent for the Frozen Lake environment.
        Args:
            env: The Frozen Lake environment.
            learning_rate: The learning rate for the Q-learning algorithm.
            initial_epsilon: The initial exploration rate.
            epsilon_decay: The decay rate for the exploration rate.
            final_epsilon: The final exploration rate.
            discount_factor: The discount factor for the Q-learning algorithm.
        """
        self.env = env
        
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: int) -> int:
        """
        Choose an action based on the current state.
        Args:
            obs: The current state of the environment.
        Returns:
            The action to take.
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def update(
            self,
            obs: Tuple[int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: Tuple[int, int, bool],
    ):
        
        """
        Update the Q-values based on the observed transition.
        Args:
            obs: The current state of the environment.
            action: The action taken.
            reward: The reward received.
            terminated: Whether the episode has terminated.
            next_obs: The next state of the environment.
        """
        future_q_value = (not terminated) * (self.discount_factor * np.max(self.q_values[next_obs]))

<<<<<<< HEAD:Q_learning/q_learning_gymna.py
        target = reward + self.discount_factor * future_q_value
        
=======
        # target = reward + self.discount_factor * future_q_value
        target = reward + future_q_value
>>>>>>> a7b3edab59fcae207bb68958ee7a3b8565080dff:Reinforcement_learning/Q_learning/q_learning_gymna.py
        temporal_difference = target - self.q_values[obs][action]

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """ Reduce the exploration rate based on the decay rate. """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    

    def test_agent(
            self, 
            env,
            map_desc,
            num_episodes: int,
            visual: bool = False,
            delay: float = 0.05
    ):
        """
        Test the agent's performance over a number of episodes.
        Args:
            env: The environment to test the agent in.
            num_episodes: The number of episodes to test the agent over.
        """
        import time
        if visual:
            test_env = gym.make("FrozenLake-v1",
                                render_mode="human",
                                desc=map_desc,
                                is_slippery=True)
        else:
            test_env = gym.make("FrozenLake-v1",
                                render_mode="rgb_array",
                                desc=map_desc,
                                is_slippery=True)
        total_rewards = []
        total_steps = []
        old_epsilon = self.epsilon
        self.epsilon = 0.0

        for episode in range(num_episodes):
            if visual:
                print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
            
            obs, info = test_env.reset()
            episode_reward = 0
            step = 0
            done = False

            if visual:
                test_env.render()
                time.sleep(delay)
            while not done and step < 100:
                action = self.get_action(obs)
                if visual:
                    action_names = ["Left", "Down", "Right", "Up"]
                    print(f"Step {step + 1}: Action = {action_names[action]}")

                obs, reward, terminated, truncated, info = test_env.step(action)
                episode_reward += reward
                done = terminated or truncated
                step += 1

                if visual:
                    test_env.render()
                    time.sleep(delay)
                    if reward > 0:
                        print("Reached the goal!")
                    elif terminated and reward == 0:
                        print("Fell into a hole!")

            total_rewards.append(episode_reward)
            total_steps.append(step)
            if visual:
                print(f"Episode reward: {episode_reward}, Steps: {step}")

        if visual:
            test_env.close()
        self.epsilon = old_epsilon

        win_rate = np.mean(np.array(total_rewards) > 0)
        average_reward = np.mean(total_rewards)
                                 
        print(f"Test Results over {num_episodes} episodes:")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Average Reward: {average_reward:.3f}")
        print(f"Standard Deviation: {np.std(total_rewards):.3f}")          

        return win_rate, average_reward
if __name__ == "__main__":
    learning_rate = 0.1
    initial_epsilon = 1.0
    epsilon_decay = 0.0001
    final_epsilon = 0.01
    n_episodes = 100_00
    discount_factor = 0.99   
    map_desc = generate_random_map(size=5)

    env = gym.make("FrozenLake-v1", render_mode="rgb_array", desc=map_desc, is_slippery=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
    agent = FrozenLakeEnv(
        env,
        learning_rate,
        initial_epsilon,
        epsilon_decay,
        final_epsilon,
        discount_factor
    )

    for episode in tqdm(range(n_episodes)):

        obs, info = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)

            agent.update(obs, action, reward, terminated, next_obs)

            done = terminated or truncated
            obs = next_obs


        agent.decay_epsilon()
    def get_moving_avgs(arr, window, convolution_mode):
        """Compute moving average to smooth noisy data."""
        return np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode
        ) / window

    # Smooth over a 500-episode window
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning)
    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()

    test_agent = agent.test_agent(env,map_desc,visual=True, num_episodes=100)
    