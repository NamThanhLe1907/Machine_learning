import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional



def Environment(
    rows: int,
    cols: int,
    terminal_kill: List[Tuple[int,int]],
    rewards: Dict[Tuple[int,int],int],
) -> Tuple[np.array, List[Tuple[int,int]], List[str]]:
    
    grids = np.zeros((rows,cols))
    
    for (i,j), r in rewards.items():
        grids[i,j] = r
        
    state_space = [
        (row,col)
        for row in range(rows)
        for col in range(cols)
    ]
    
    actions_space = ['up', 'down', 'left', 'right']
    return grids, state_space, actions_space

def state_transition(state: Tuple[int,int], action: str, rows: int, cols: int) -> Tuple[int,int]:
    row, col = state
    if action == 'up' and row > 0:
        row -= 1
    elif action =='down' and row < rows - 1 :
        row += 1
    elif action == 'left' and col > 0:
        col -= 1
    elif action == 'right' and col < cols - 1:
        col += 1
        
    return (row, col)

def get_rewards(state: Tuple[int,int], rewards: Dict[Tuple[int,int],int]) -> int:
    if state in rewards:
        return rewards[state]
    else:
        return 0
    #else we can write return rewards.get(state,0) for easy the 4 lines code above
    
def init_q_table(state_space: List[Tuple[int,int]], action_space: List[str]) -> Dict[Tuple[int,int], Dict[str, float]]: 
    #init q_table with simple q-function
    #such as Q(s,a)
    
    q_table: Dict[Tuple[int,int], Dict[str,float]] = {}
    for state in state_space:
        q_table[state] = {action: 0.0 for action in action_space}
    return q_table


def choose_action(state: Tuple[int,int], action_space: List[str], epsilon: float, q_table: Dict[Tuple[int, int], Dict[str, float]] ) -> str:
    
    #With probability epsilon, choose a random action (exploration)
    if np.random.rand() < epsilon:
        return np.random.choice(action_space)
    else:
        return max(q_table[state], key=q_table[state].get)
    
def update_q_value(
    q_table: Dict[Tuple[int,int], Dict[str,float]],
    state: Tuple[int, int],
    action: str,
    reward: int,
    next_state: Tuple[int, int],
    alpha: float,
    gamma: float,
    action_space: List[str]
) -> None:
    
    max_next_q: float = max(q_table[next_state].values()) if next_state in q_table else 0.0
    
    q_table[state][action] += alpha * (reward + gamma* max_next_q - q_table[state][action])
    
def run_episode(
    q_table: Dict[Tuple[int, int], Dict[str, float]],
    state_space: List[Tuple[int, int]],
    action_space: List[str],
    rewards: Dict[Tuple[int, int], int],
    rows: int,
    cols: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    max_steps: int
) -> int:
    
    state: Tuple[int, int] = state_space[np.random.choice(len(state_space))]
    total_reward: int = 0
    
    for _ in range(max_steps):
        action: str = choose_action(state, action_space, epsilon, q_table)
        
        next_state: Tuple[int, int] = state_transition(state, action, rows, cols)
        
        reward: int = get_rewards(next_state, rewards)
        
        update_q_value(q_table, state, action, reward, next_state, alpha, gamma, action_space)
        
        total_reward += reward
        
        state = next_state
        
        if state in terminal_states:
            break
        
    return total_reward

def epsilon_greedy_policy(q_table: Dict[Tuple[int,int], Dict[str,float]],
                          state: Tuple[int,int],
                          action_space: List[str],
                          epsilon: float)-> str:
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
    
    return max(min_epsilon, initial_epsilon* np.exp(-decay_rate*episode)) 


def run_episode_v2(
    q_table: Dict[Tuple[int, int], Dict[str, float]],
    state_space: List[Tuple[int, int]],
    action_space: List[str],
    rewards: Dict[Tuple[int, int], int],
    rows: int,
    cols: int,
    initial_epsilon: float,
    min_epsilon: float,
    decay_rate: float,
    episodes: int,
    alpha: float,
    gamma: float,
    max_steps: int
) -> Tuple[List[int], List[int]]:
    reward_for_episode: List[int] = []
    episode_length: List[int] = []
    for episode in range(episodes):
       state: Tuple[int,int] = state_space[np.random.choice(len(state_space))]
       total_reward: int = 0
       steps: int = 0
       
       epsilon: float = adjust_epsilon(initial_epsilon, min_epsilon, decay_rate, episode)
       
       for _ in range(max_steps):
           action: str = epsilon_greedy_policy(q_table, state, action_space, epsilon)
           
           next_state: Tuple[int,int] = state_transition(state, action, rows, cols)
           reward: int = get_rewards(next_state, rewards)
           update_q_value(q_table, state, action, reward, next_state, alpha, gamma, action_space)
           total_reward += reward
           state = next_state
           steps += 1
        
           if state in terminal_states:
               break
           
       reward_for_episode.append(total_reward)
       episode_length.append(steps)
    return reward_for_episode, episode_length 

def evaluate(hyper_params: Dict[str, float],
            env_params: Dict,
            seed: int = 0) -> Dict[str, float]:
    """
    Evaluate the performance of the Q-learning algorithm with different hyperparameters.
    
    Args:
        hyper_params (Dict[str, float]): A dictionary containing the hyperparameters to evaluate.
    """
    np.random.seed(seed)
    rows, cols = env_params['rows'], env_params['cols']
    terminal_states = env_params['terminal_states']
    rewards = env_params['rewards']

    _ , state_space, action_space = Environment(rows, cols, terminal_states, rewards)
    q_table = init_q_table(state_space, action_space)

    alpha = hyper_params['alpha']
    gamma = hyper_params['gamma']
    initial_epsilon = hyper_params['initial_epsilon']
    min_epsilon = hyper_params['min_epsilon']
    decay_rate = hyper_params['decay_rate']
    episodes = hyper_params['episodes']
    max_steps = hyper_params['max_steps']

    rewards_per_episode, _ = run_episode_v2(
        q_table,state_space,action_space,rewards,rows,cols,
        initial_epsilon,min_epsilon, decay_rate, episodes, alpha,
        gamma, max_steps
    )
    
    mean_last100 = np.mean(rewards_per_episode[-100:])
    best_mean    = np.max[(np.mean(rewards_per_episode[i:i+100]
                                   for i in range(len(rewards_per_episode)-100)))]
    return {
        'mean_last100': mean_last100,
        'best_mean': best_mean
        **hyper_params,
        'seed': seed
    }

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
    
def plot_policy(q_table: Dict[Tuple[int,int], Dict[str,float]], rows: int, cols: int)-> None:
    policy_grid = np.empty((rows, cols), dtype=str)
    action_symbols = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}  # Symbols for actions

    # Determine the best action for each state based on Q-values
    for (row, col), actions in q_table.items():
        best_action = max(actions, key=actions.get)  # Get the action with the highest Q-value
        policy_grid[row, col] = action_symbols[best_action]  # Map the action to its symbol

    # Plot the policy grid with increased width
    fig, ax = plt.subplots(figsize=(16, 3))  # Increased width from 12 to 16 for more horizontal stretch
    for i in range(rows):
        for j in range(cols):
            ax.text(j, i, policy_grid[i, j], ha='center', va='center', fontsize=14)  # Slightly larger font
    
    # Create a wider grid with more horizontal space
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.matshow(np.zeros((rows, cols)), cmap='Greys', alpha=0.1)  # Add a faint background grid
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("Learned Policy")
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    """Example for first of init q_table 
    """

    # rows, cols = 4,4
    # terminal_states = [(1,1) , (3,3)]
    # rewards = {(0,0):1 , (3,3):10}
    
    # grids, state_space, actions_space = Environment(rows, cols, terminal_states, rewards)
    
    # current_state = (0,0)
    # action = 'up'
    # next_state = state_transition(current_state, action, rows, cols)
    # reward = get_rewards(current_state, rewards)
    
    # # q_table = init_q_table(state_space, actions_space)   
    # print("Grid worlds:")
    # print(grids)
    # print(f"Current State: {current_state}")
    # print(f"Current action: {action}")
    # print(f"Next state: {next_state}")
    # print(f"Rewards: {reward}")
    # # print(f"q_table: {q_table}")


    """
    Example 2: Example for update_q_table 
    """
    # rows, cols = (4,4)
    # terminal_states = [(0,0), (3,3)]
    # rewards = {(3,3):10}
    # alpha = 0.1
    # gamma = 0.9
    # epsilon = 0.1
    # max_steps = 100
    # episodes = 500
    
    # grids, state_space, action_space = Environment(rows, cols, terminal_states, rewards)
    # q_table = init_q_table(state_space, action_space)
    
    # rewards_per_episode = []
    
    # for episode in range(episodes):
        
    #     total_reward = run_episode(q_table, state_space, action_space, rewards, rows, cols, alpha, gamma, epsilon, max_steps)
        
    #     rewards_per_episode.append(total_reward)
        
        
    # plt.figure(figsize=(20,3))
    # plt.plot(rewards_per_episode)
    # plt.xlabel('Episode')  # Label for the x-axis
    # plt.ylabel('Total Reward')  # Label for the y-axis
    # plt.title('Rewards Over Episodes')  # Title of the plot
    # plt.show()  # Display the plot
    
    """
    Example 3: Example for update epsilon
    """
    
    # rows, cols = (4,4)
    # terminal_states = [(1,1), (3,3)]
    # rewards = {(3,3):10}
    # initial_epsilon: float = 1.0
    # min_epsilon: float = 0.1
    # decay_rate: float = 0.01
    # episodes: int = 500
    
    # epsilon_value: List[float] = []
    # for episode in range(episodes):
    #     epsilon = adjust_epsilon(initial_epsilon, min_epsilon, decay_rate, episode)
    #     epsilon_value.append(epsilon)
        
    # plt.figure(figsize=(20,3))
    
    # plt.plot(epsilon_value)
    # plt.xlabel('Episode')
    # plt.ylabel('Epsilon')
    # plt.title('Epsilon Decay Over Episode')
    # plt.show()
     
     
    """
    Example 4: Example for exploration and exploitation
    
    """
    alpha: float = 0.1
    gamma: float = 0.9
    initial_epsilon: float = 1.0
    min_epsilon: float = 0.05
    decay_rate: float = 0.01
    episodes: int = 500
    max_steps: int = 100
    rows, cols = (4,4)
    terminal_states = [(1,1), (3,3)]
    rewards = {(3,3):10}
    
    grids, state_space, action_space = Environment(rows, cols, terminal_states, rewards)
    # learning_rates = [0.1, 0.5]  # Different learning rates (alpha) to test
    # discount_factors = [0.9]  # Different discount factors (gamma) to test
    # exploration_rates = [0.95]  # Different initial exploration rates (epsilon) to test

    # # Store results for comparison
    # results = []

    # # Run experiments with different hyperparameter combinations
    # for alpha in learning_rates:  # Iterate over different learning rates
    #     for gamma in discount_factors:  # Iterate over different discount factors
    #         for initial_epsilon in exploration_rates:  # Iterate over different initial exploration rates
    #             # Initialize Q-table for the current experiment
    #             q_table = init_q_table(state_space, action_space)
                
    #             # Run Q-Learning with the current set of hyperparameters
    #             rewards_per_episode, episode_lengths = run_episode_v2(
    #                 q_table,state_space,action_space,rewards,rows,cols,
    #                 initial_epsilon,min_epsilon, decay_rate, episodes, alpha,
    #                 gamma,episodes , max_steps)
                
    #             # Store the results of the current experiment
    #             results.append({
    #                 'alpha': alpha,  # Learning rate
    #                 'gamma': gamma,  # Discount factor
    #                 'initial_epsilon': initial_epsilon,  # Initial exploration rate
    #                 'rewards_per_episode': rewards_per_episode,  # Rewards collected per episode
    #                 'episode_lengths': episode_lengths  # Length of each episode
    #             })

    # # Create a larger figure to visualize all hyperparameter combinations
    # plt.figure(figsize=(20, 5))

    # # Calculate the number of rows and columns for the subplot grid
    # num_rows = len(learning_rates)  # Number of rows corresponds to the number of learning rates
    # num_cols = len(discount_factors) * len(exploration_rates)  # Number of columns corresponds to combinations of discount factors and exploration rates

    # # Plot the results of each experiment
    # for i, result in enumerate(results):  # Iterate over all results
    #     plt.subplot(num_rows, num_cols, i + 1)  # Create a subplot for each experiment
    #     plt.plot(result['rewards_per_episode'])  # Plot rewards per episode
    #     plt.title(f"α={result['alpha']}, γ={result['gamma']}, ε={result['initial_epsilon']}")  # Add a title with hyperparameter values
    #     plt.xlabel('Episode')  # Label for the x-axis
    #     plt.ylabel('Total Reward')  # Label for the y-axis

    # # Adjust layout to prevent overlap and display the plots
    # plt.tight_layout()
    # plt.show()
    q_table: Dict[Tuple[int,int], Dict[str,float]] = init_q_table(state_space,action_space)
    rewards_per_episode, episode_lengths = run_episode_v2(
        q_table,state_space,action_space,rewards,rows,cols,
        initial_epsilon,min_epsilon, decay_rate, episodes, alpha,
        gamma, max_steps
    )
    # Plot cumulative rewards over episodes
    plt.figure(figsize=(20, 3))

    # Plot total rewards per episode
    plt.subplot(1, 2, 1)
    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')  # Label for the x-axis
    plt.ylabel('Total Reward')  # Label for the y-axis
    plt.title('Cumulative Rewards Over Episodes')  # Title of the plot

    # Plot episode lengths per episode
    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths)
    plt.xlabel('Episode')  # Label for the x-axis
    plt.ylabel('Episode Length')  # Label for the y-axis
    plt.title('Episode Lengths Over Episodes')  # Title of the plot

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()
    
    fig, axes = plt.subplots(1, len(action_space) + 1, figsize=(20, 5))

    # Plot the Q-value heatmaps for each action
    for i, action in enumerate(action_space):
        q_values = np.zeros((rows, cols))
        for (row, col), actions in q_table.items():
            q_values[row, col] = actions[action]
        cax = axes[i].matshow(q_values, cmap='viridis')
        fig.colorbar(cax, ax=axes[i])
        axes[i].set_title(f"Q-values for action: {action}")
        axes[i].set_xlabel("Columns")
        axes[i].set_ylabel("Rows")

    # Plot the learned policy
    policy_grid = np.empty((rows, cols), dtype=str)
    action_symbols = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
    for (row, col), actions in q_table.items():
        best_action = max(actions, key=actions.get)
        policy_grid[row, col] = action_symbols[best_action]

    axes[-1].matshow(np.zeros((rows, cols)), cmap='Greys', alpha=0.1)
    for i in range(rows):
        for j in range(cols):
            axes[-1].text(j, i, policy_grid[i, j], ha='center', va='center', fontsize=14)
    axes[-1].set_title("Learned Policy")
    axes[-1].set_xlabel("Columns")
    axes[-1].set_ylabel("Rows")

    plt.tight_layout()
    plt.show()
    
    q_policy_data = []
    for state, actions in q_table.items():
        # Append a dictionary for each state containing Q-values for all actions and the optimal action
        q_policy_data.append({
            'State': state,  # The current state (row, col)
            'up': actions['up'],  # Q-value for the 'up' action
            'down': actions['down'],  # Q-value for the 'down' action
            'left': actions['left'],  # Q-value for the 'left' action
            'right': actions['right'],  # Q-value for the 'right' action
            'Optimal Action': max(actions, key=actions.get)  # The action with the highest Q-value
        })

    # Display the Q-table data in a tabular format
    header = ['State', 'up', 'down', 'left', 'right', 'Optimal Action']  # Define the table headers
    # Print the table header with proper spacing
    print(f"{header[0]:<10} {header[1]:<10} {header[2]:<10} {header[3]:<10} {header[4]:<10} {header[5]:<15}")
    print("-" * 65)  # Print a separator line for better readability

    # Iterate through the Q-table data and print each row
    for row in q_policy_data:
        # Print the state, Q-values for all actions, and the optimal action
        print(f"{row['State']!s:<10} {row['up']:<10.2f} {row['down']:<10.2f} {row['left']:<10.2f} {row['right']:<10.2f} {row['Optimal Action']:<15}")