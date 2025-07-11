from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple

def create_env(render_mode=None):
    """
    Create the environment.
    """
    return gym.make("FrozenLake-v1", render_mode=render_mode,desc=generate_random_map(size = 8))

def explore_environment(env):
    """
    Explore the environment and print the state and action.
    """

    observation, info = env.reset(seed=42)
    print(f"Initial observation: {observation}")
    print(f"Initial info: {info}")
    
    for i in range(4):
        print(f"\n---Action {i}---")
        obs, reward, terminated, truncated, info = env.step(i)
        print(f"Observation: {obs}, Reward: {reward}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")

        if terminated or truncated:
            observation, info = env.reset()
            print(f"Environment reset")
            break
def init_q_table(
        state_space_size: int,
        action_space_size: int,
)-> np.ndarray:
    """
    Initialize the Q-table with zeros.
    """
    return np.zeros((state_space_size, action_space_size))



def epsilon_greedy_policy(q_table: np.ndarray,
                          state: int, 
                          epsilon: float,
                          env) -> int:
    """
    Generate a policy based on epsilon-greedy strategy.

    Args:
        q_table (Dict[Tuple[int, int], Dict[str, float]]): Q-table.
        epsilon (float): Probability of choosing a random action.
        action_space (List[str]): List of actions available in the environment.

    Returns:
        int: Action index (0-3 for FrozenLake)
    """
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

def adjust_epsilon(
        initial_epsilon: float,
        min_epsilon: float,
        decay_rate: float,
        episode: int
)-> float:
    """
    Adjust epsilon value for epsilon-greedy policy.
    """
    return max(min_epsilon, initial_epsilon * np.exp(-decay_rate * episode))

def update_q_value(q_table:    np.ndarray,
                   state:      int,
                   action:     int,
                   reward:     float,
                   n_state:    int,
                   alpha:      float,
                   gamma:      float,
                   terminated: bool)-> None:
    """
    Update the Q-value for a given state-action pair.

    Args:
        q_table (np.ndarray): Q-table.
        state (int): Current state.
        action (int): Action taken.
        reward (float): Reward received.
        n_state (int): Next state.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        terminated (bool): Whether the episode is terminated.
    """
    if terminated:
        td_target = reward
    else:
        td_target = reward + gamma * np.max(q_table[n_state])
    td_error = td_target - q_table[state, action]
    q_table[state, action] += alpha * td_error


def run_episode(
        env,
        q_table: np.ndarray,
        episodes: int,
        alpha: float,
        gamma: float,
        initial_epsilon: float,
        min_epsilon: float,
        decay_rate: float,
        max_steps: int,
)-> Tuple[List[float], List[int]]:
    """
    Run an episode of the Q-learning algorithm.
    """
    reward_per_episode = []
    steps_per_episode = []
    print("=== TRAINING PROGRESS ===")
    print(f"Episodes: {episodes}, Learning Rate: {alpha}, Discount: {gamma}")
    print(f"Epsilon: {initial_epsilon} → {min_epsilon} (decay: {decay_rate})")
    print("-" * 60)
    env = create_env(render_mode=None)
    for episode in range(episodes):
        observation, info = env.reset(seed=42)
        state = observation

        total_reward = 0
        steps = 0
        epsilon = adjust_epsilon(initial_epsilon, min_epsilon, decay_rate, episode)
        for steps in range(max_steps):
            action = epsilon_greedy_policy(q_table, state, epsilon, env)

            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = next_observation

            update_q_value(q_table, state, action, reward, next_state, alpha, gamma, terminated)
            total_reward += reward
            state = next_state
            steps += 1

            if terminated or truncated:
                break
        reward_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        if (episode + 1) % 50 == 0:  # Mỗi 50 episodes
            avg_reward = np.mean(reward_per_episode[-50:])
            avg_steps = np.mean(steps_per_episode[-50:])
            success_rate = sum(1 for r in reward_per_episode[-50:] if r > 0) / 50
            print(f"Episode {episode+1:4d}/{episodes} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Avg Steps: {avg_steps:.1f} | "
                  f"Success Rate: {success_rate:.2%} | "
                  f"Epsilon: {epsilon:.3f}")
    return reward_per_episode, steps_per_episode

def train_agent():
    """
    Train the agent.
    """
    episodes = 5000
    alpha = 0.1
    gamma = 0.99
    initial_epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.001
    max_steps = 100

    env = create_env(render_mode=None)

    state_space_size = env.observation_space.n
    action_space_size = env.action_space.n
    q_table = init_q_table(state_space_size, action_space_size)

    print("Training agent...")
    rewards, steps = run_episode(
        env, q_table, episodes, alpha, gamma, initial_epsilon,
        min_epsilon, decay_rate, max_steps
    )
    print("Training completed!\n")

    success_rate , avg_reward = test_agent(env, q_table, max_steps)
    env.close()
    print(f"Success Rate: {success_rate:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Q-table shape: {q_table.shape}")
    return q_table, rewards, steps

def plot_training_results(rewards: List[float],
                          steps: List[int]):
    """
    Plot the training progess.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,5))
    
    #Rewards per episode
    plt.subplot(1, 3, 1)
    plt.plot(rewards)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)

    #Steps per episode
    plt.subplot(1, 3, 2)
    plt.plot(steps)
    plt.title("Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True)

    #Moving avarage
    plt.subplot(1, 3, 3)
    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode="valid")
        plt.plot(moving_avg)
        plt.title(f"Moving Average Reward (window = {window})")
        plt.xlabel("Episode")
        plt.ylabel("Avg Reward")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def test_agent(env, q_table: np.ndarray, num_episodes: int)-> float:
    total_reward = []
    success_count = 0

    for episode in range(num_episodes):
        observation, info = env.reset()
        state = observation
        episode_reward = 0
        steps = 0

        while steps < 100:
            action = np.argmax(q_table[state])
            observation, reward, terminated, truncated, info = env.step(action)
            state = observation
            episode_reward += reward
            steps += 1

            if terminated or truncated:
                if reward > 0:
                    success_count += 1
                break
        total_reward.append(episode_reward)

    success_rate = success_count / num_episodes
    avg_reward = np.mean(total_reward)
    print(f"Test Results over {num_episodes} episodes:")
    print(f"Success Rate: {success_rate:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")
    return success_rate, avg_reward
def demonstrate_agent(q_table: np.ndarray, num_demos: int = 5):
    """
    Xem agent hoạt động với render visual
    """
    print(f"\n🎮 DEMONSTRATING TRAINED AGENT - {num_demos} episodes")
    print("Actions: 0=Left, 1=Down, 2=Right, 3=Up")
    print("S=Start, F=Frozen, H=Hole, G=Goal")
    print("-" * 50)
    
    # Tạo environment với render
    env = create_env(render_mode="human")
    
    success_count = 0
    
    for demo in range(num_demos):
        print(f"\n🎯 Demo Episode {demo + 1}/{num_demos}")
        observation, info = env.reset()
        state = observation
        steps = 0
        total_reward = 0
        
        print(f"Starting at state {state}")
        
        while steps < 100:  # Max steps
            # GREEDY POLICY - no exploration
            action = np.argmax(q_table[state])
            action_names = ['Left ←', 'Down ↓', 'Right →', 'Up ↑']
            
            print(f"Step {steps + 1}: State {state} → Action {action} ({action_names[action]})")
            
            observation, reward, terminated, truncated, info = env.step(action)
            state = observation
            total_reward += reward
            steps += 1
            
            # Pause để xem rõ
            import time
            time.sleep(0.5)  # Dừng 0.5 giây mỗi step
            
            if terminated or truncated:
                if reward > 0:
                    print(f"🎉 SUCCESS! Reached goal in {steps} steps!")
                    success_count += 1
                else:
                    print(f"💀 Failed! Fell into hole")
                break
        
        print(f"Total reward: {total_reward}")
        
        # Pause between episodes
        input("Press Enter for next episode (or Ctrl+C to stop)...")
    
    print(f"\n📊 Demo Results: {success_count}/{num_demos} successes ({success_count/num_demos:.1%})")
    env.close()
if __name__ == "__main__":
    q_table, rewards, steps = train_agent()
    
    plot_training_results(rewards, steps)   # Training progress

        # THÊM DEMO
    choice = input("\n🎮 Want to see the agent in action? (y/n): ").lower()
    if choice == 'y':
        try:
            demonstrate_agent(q_table, 5)
        except KeyboardInterrupt:
            print("\n👋 Demo stopped by user")