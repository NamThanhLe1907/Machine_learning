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
    rows, cols = (4,4)
    terminal_states = [(0,0), (3,3)]
    rewards = {(3,3):10}
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    max_steps = 100
    episodes = 500
    
    grids, state_space, action_space = Environment(rows, cols, terminal_states, rewards)
    q_table = init_q_table(state_space, action_space)
    
    rewards_per_episode = []
    
    for episode in range(episodes):
        
        total_reward = run_episode(q_table, state_space, action_space, rewards, rows, cols, alpha, gamma, epsilon, max_steps)
        
        rewards_per_episode.append(total_reward)
        
        
    plt.figure(figsize=(20,3))
    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')  # Label for the x-axis
    plt.ylabel('Total Reward')  # Label for the y-axis
    plt.title('Rewards Over Episodes')  # Title of the plot
    plt.show()  # Display the plot
    