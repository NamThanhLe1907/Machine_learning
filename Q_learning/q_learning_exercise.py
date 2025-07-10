import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional

def Environment(
        rows: int,
        cols: int,
        terminal_kill: List[Tuple[int, int]],
        rewards: Dict[Tuple[int, int], int],
) -> Tuple[np.array, List[Tuple[int,int]], List[str]]:
    """
    Create a grid environment for Q-learning.

    Args:
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        terminal_kill (List[Tuple[int, int]]): List of terminal states that end the episode.
        rewards (Dict[Tuple[int, int], float]): Dictionary mapping states to rewards.

    Returns:
        Tuple[np.array, List[Tuple[int,int]], List[str]]: The grid environment, terminal states, and actions.
    """
    grid = np.zeros((rows, cols))

    for (i, j), r in rewards.items():
        grid[i, j] = r

    state_space = [
        (row,col)
        for row in range(rows)
        for col in range(cols)
    ]

    action_space = ['up', 'down', 'left', 'right']
    return grid, state_space, action_space

def state_transition(
        state: Tuple[int, int],
        action: str,
        rows: int,
        cols: int,
)-> Tuple[int, int]:
    """
    Determine the next state based on the current state and action.

    Args:
        state (Tuple[int, int]): Current state (row, column).
        action (str): Action to take ('up', 'down', 'left', 'right').
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.

    Returns:
        Tuple[int, int]: Next state after taking the action.
    """
    row, col = state
    if action == 'up' and row > 0:
        row -= 1
    elif action == 'down' and row < rows - 1:
        row += 1
    elif action == 'left' and col > 0:
        col -= 1
    elif action == 'right' and col < cols - 1:
        col += 1

    return (row, col)

def get_rewards(
        state: Tuple[int, int],
        prev_state: Tuple[int, int],
        rewards: Dict[Tuple[int, int], float],
) -> float:
    """
    Get the reward for a given state.

    Args:
        state (Tuple[int, int]): Current state (row, column).
        rewards (Dict[Tuple[int, int], float]): Dictionary mapping states to rewards.

    Returns:
        float: Reward for the current state.
    """
    base_reward = rewards.get(state, 0)
    if state == prev_state:
        base_reward -= 1.5
    return base_reward

def init_q_table(
        state_space: List[Tuple[int, int]],
        action_space: List[str],
)-> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Initialize the Q-table with zeros.

    Args:
        state_space (List[Tuple[int, int]]): List of states in the environment.
        action_space (List[str]): List of actions available in the environment.

    Returns:
        Dict[Tuple[int, int], Dict[str, float]]: Initialized Q-table.
    """
    q_table: Dict[Tuple[int, int], Dict[str, float]] = {}
    for state in state_space:
        q_table[state] = {action: 0.0 for action in action_space}
    return q_table

def update_q_value(
        q_table: Dict[Tuple[int, int], Dict[str, float]],
        state: Tuple[int, int],
        action: str,
        reward: float,
        next_state: Tuple[int, int],
        alpha: float,
        gamma: float,
        terminal_states: List[Tuple[int, int]]
) -> None:
    """
    Update the Q-value for a given state-action pair.

    Args:
        q_table (Dict[Tuple[int, int], Dict[str, float]]): Q-table.
        state (Tuple[int, int]): Current state (row, column).
        action (str): Action taken.
        reward (float): Reward received after taking the action.
        next_state (Tuple[int, int]): Next state after taking the action.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        """
    if next_state in terminal_states:
        td_target = reward
    else:
        best_next_action = max(q_table[next_state], key=q_table[next_state].get) if next_state in q_table else 0.0
        td_target = reward + gamma * q_table[next_state][best_next_action]
    td_error = td_target - q_table[state][action]
    q_table[state][action] += alpha * td_error

    
def epsilon_greedy_policy(
        q_table: Dict[Tuple[int, int], Dict[str, float]],
        epsilon: float,
        action_space: List[str],
        state: Tuple[int,int]
) -> str:
    """
    Generate a policy based on epsilon-greedy strategy.

    Args:
        q_table (Dict[Tuple[int, int], Dict[str, float]]): Q-table.
        epsilon (float): Probability of choosing a random action.
        action_space (List[str]): List of actions available in the environment.
    """
    if np.random.rand() < epsilon:
        return np.random.choice(action_space)
    else:
        return max(q_table[state], key=q_table[state].get)
    
def adjust_epsilon(
        initial_epsilon: float,
        min_epsilon: float,
        decay_rate: float,
        episode: int
)-> float:
    """
    Adjust epsilon value for epsilon-greedy policy.

    Args:
        initial_epsilon (float): Initial value of epsilon.
        min_epsilon (float): Minimum value of epsilon.
        decay_rate (float): Rate at which epsilon decays.
        episode (int): Current episode number.

    Returns:
        float: Adjusted epsilon value.
    """
    return max(min_epsilon, initial_epsilon * np.exp(-decay_rate * episode))

def run_episode(
        q_table: Dict[Tuple[int, int], Dict[str, float]],
        state_space: List[Tuple[int, int]],
        action_space: List[str],
        rewards: Dict[Tuple[int, int], float],
        terminal_states: List[Tuple[int, int]],  
        rows: int,
        cols: int,
        initial_epsilon: float,
        min_epsilon: float,
        decay_rate: float,
        episodes:int,
        alpha: float,
        gamma: float,
        max_steps: int
)-> Tuple[List[int], List[int]]:  
    reward_for_episode: List[int] = []
    episode_length: List[int] = []
    for episode in range(episodes):
        state: Tuple[int,int] = (0,0) 
        total_reward: int = 0
        steps: int = 0
        epsilon: float = adjust_epsilon(initial_epsilon, min_epsilon, decay_rate, episode)

        for _ in range(max_steps):
            action: str = epsilon_greedy_policy(q_table, epsilon, action_space, state)
            next_state: Tuple[int, int] = state_transition(state, action, rows, cols)
            reward: float = get_rewards(next_state,state, rewards)
            update_q_value(
                q_table, state, action, reward, next_state, alpha, gamma, terminal_states
            )
            total_reward += reward
            state = next_state
            steps += 1 

            if state in terminal_states:
                break


        reward_for_episode.append(total_reward)
        episode_length.append(steps)
    return reward_for_episode, episode_length

def plot_q_values(q_table: Dict[Tuple[int,int], Dict[str,float]], rows: int, cols: int, action_space: List[str]) -> None:
    fig, axes = plt.subplots(1, len(action_space), figsize=(15, 5))
    for i, action in enumerate(action_space):
        # Initialize a grid to store Q-values for the current action
        q_values = np.zeros((rows, cols))
        for (row, col), actions in q_table.items():
            q_values[row, col] = actions[action]  # Extract Q-value for the current action

        # Plot the heatmap for the current action
        ax = axes[i]
        cax = ax.matshow(q_values, cmap='viridis')
        fig.colorbar(cax, ax=ax)
        ax.set_title(f"Q-values for action: {action}")
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")

    # Adjust layout and display the heatmaps
    plt.tight_layout()
    plt.show()
    
def plot_policy(q_table: Dict[Tuple[int,int], Dict[str,float]], rows: int, cols: int, terminal_states: List[Tuple[int,int]])-> None:
    policy_grid = np.empty((rows, cols), dtype=str)
    action_symbols = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
    terminal_symbol = '■'
    
    # Determine the best action for each state based on Q-values
    for (row, col), actions in q_table.items():
        if (row, col) in terminal_states:
            policy_grid[row, col] = terminal_symbol
        else:
            best_action = max(actions, key=actions.get)
            policy_grid[row, col] = action_symbols[best_action]

    # Plot the policy grid
    fig, ax = plt.subplots(figsize=(16, 3))
    for i in range(rows):
        for j in range(cols):
            color = 'red' if (i, j) in terminal_states else 'black'
            
            ax.text(j, rows-1-i, policy_grid[i, j], ha='center', va='center', fontsize=14, color=color)
    
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.matshow(np.zeros((rows, cols)), cmap='Greys', alpha=0.1)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("Learned Policy")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    alpha: float = 0.1
    gamma: float = 0.9
    initial_epsilon: float = 1.0
    min_epsilon: float = 0.05
    decay_rate: float = 0.005
    episodes: int = 2000
    max_steps: int = 20
    rows, cols = (4,4)
    terminal_states = [(1,1),(1,2), (3,3)]
    rewards = {
        (1,1): -10.0,
        (1,2): -10.0,
        (1,3): 1.0,
        (2,2): 1.0,
        (3,3): 10.0
    }
    
    grids, state_space, action_space = Environment(rows, cols, terminal_states, rewards)
    q_table = init_q_table(state_space, action_space)
    reward_per_episode, episode_length = run_episode(
        q_table, state_space, action_space, rewards, terminal_states,  # THÊM terminal_states
        rows, cols, initial_epsilon, min_epsilon, decay_rate, episodes, alpha, gamma, max_steps
    )
     # Plot cumulative rewards over episodes
    plt.figure(figsize=(20, 3))

    # Plot total rewards per episode
    plt.subplot(1, 2, 1)
    plt.plot(reward_per_episode)
    plt.xlabel('Episode')  
    plt.ylabel('Total Reward')  
    plt.title('Cumulative Rewards Over Episodes')  

    # Plot episode lengths per episode
    plt.subplot(1, 2, 2)
    plt.plot(episode_length)
    plt.xlabel('Episode')  
    plt.ylabel('Episode Length')  
    plt.title('Episode Lengths Over Episodes')  

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()
    
    # Plot Q-value heatmaps for each action
    plot_q_values(q_table, rows, cols, action_space)

    # Plot policy, 
    plot_policy(q_table, rows, cols, terminal_states)

    # Display Q-table as a table
    q_policy_data = []
    for state, actions in q_table.items():
        q_policy_data.append({
            'State': state,
            'up': actions['up'],
            'down': actions['down'],
            'left': actions['left'],
            'right': actions['right'],
            'Optimal Action': max(actions, key=actions.get)
        })
    header = ['State', 'up', 'down', 'left', 'right', 'Optimal Action']
    print(f"{header[0]:<10} {header[1]:<10} {header[2]:<10} {header[3]:<10} {header[4]:<10} {header[5]:<15}")
    print("-" * 65)
    for row in q_policy_data:
        print(f"{row['State']!s:<10} {row['up']:<10.2f} {row['down']:<10.2f} {row['left']:<10.2f} {row['right']:<10.2f} {row['Optimal Action']:<15}")