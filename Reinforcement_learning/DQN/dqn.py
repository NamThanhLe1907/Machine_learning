import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import namedtuple, deque
from itertools import count
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using deviece: {device}")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
    """
    Custom Grid World environment that simulates a simple grid world. This environment will have a grid of size 10x10
    Where the agent can move in four directions: up, down, left, right. Top-leftt corner (0,0) and bottom-right corner (9, 9)
    """
    
    class GridEnvironment:
        def __init__(self, rows: int = 10, cols: int = 10) -> None:
            self.rows: int = rows
            self.cols: int = cols
            self.start_state: Tuple[int, int] = (0, 0)
            self.goal_state: Tuple[int, int] = (rows - 1, cols - 1)
            self.state: Tuple[int, int] = self.start_state
            self.state_dim: int = 2 # State represented by 2 coordinates (row,col)
            self.action_dim: int = 4 # 4 discrete actions: up, down, left, right
            
            self.action_map: Dict[int, Tuple[int,int]] = {
                0: (-1, 0), # Up
                1: (1, 0), # Down
                2: (0, -1), #Left
                3: (0, 1) #Right
            }
            
        def reset(self) -> torch.Tensor:
            self.state = self.start_state
            return self._get_state_tensor(self.state)
        def _get_state_tensor(self, state_tuple: Tuple[int, int]) -> torch.Tensor:
            """
            Converts a (row, col) tuple to a normalized tensor for the network.
            """
            
            # Normalize coordinates to be between 0 and 1
            normalized_state: List[float] = [
                state_tuple[0] / (self.rows - 1),
                state_tuple[1] / (self.cols - 1)
            ]
            return torch.tensor(normalized_state, dtype=torch.float32, device=device)
        
        def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
            """
            Performs one step in the environment based on the given action.
            """
            #If the goal state is already reached, return the current state.
            if self.state == self.goal_state:
                return self._get_state_tensor(self.state) , 0.0, True
            
            #Get the rows, cols deltas for the action
            
            dr, dc = self.action_map[action]
            current_row, current_col = self.state
            next_row, next_col = current_row + dr, current_col + dc
            
            #Default step cost
            reward: float = -0.1
            hit_wall: bool = False
            
            #Check if the action leads to hitting a wall (out of bounds)
            if not (0 <= next_row < self.rows and 0 <= next_col < self.cols):
                #Stay in the same state and incur a penalty
                next_row, next_col = current_row, current_col
                reward = -1.0
                hit_wall = True
            #Update the state
            self.state = (next_row, next_col)
            next_state_tensor: torch.Tensor = self._get_state_tensor(self.state)
            
            #check if the goal state is reached
            done: bool = (self.state == self.goal_state)
            if done:
                reward += 10.0 
                
            return next_state_tensor, reward, done
        
        def get_action_space_size(self) -> int:
            """
            Returns the size of the action space
            """
            
            return self.action_dim
        
        def get_state_dimension(self) -> int:
            """
            Returns the dimension of the state representation
            """
            
            return self.state_dim
        
    """
    Let's
    """   
        
    class DQN(nn.Module):
        def __init__(self, n_observations: int, n_actions: int):
            """
            Initialize the DQN
            """
            
            super(DQN, self).__init()
            # Define network layers
            # Simple MLP: Input -> Hidden1 -> ReLU -> Hidden2 -> ReLU -> Output
            self.layer1 = nn.Linear(n_observations, 128) #Input Layer
            self.layer2 = nn.Linear(128, 128) #Hidden Layer
            self.layer3 = nn.Linear(128, n_actions)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through the network.
            """
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, device = device)
            elif x.dtype != torch.float32:
                x = x.to(dtype=torch.float32)
            #Apply layers with ReLU activation
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            return self.layer3(x)
    
    """
    Defining the Replay Memory
    """
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward', 'done'))
    
    class ReplayMemory(object):
        """Stores transitions and allows sampling batches."""
        
        def __init__(self, capaciyt: int):
            """
            Initial
            """
        
if __name__ in "__main__":
        # Instantiate the custom grid environment with a 10x10 grid
    custom_env = GridEnvironment(rows=10, cols=10)

    # Get the size of the action space and state dimension
    n_actions_custom = custom_env.get_action_space_size()
    n_observations_custom = custom_env.get_state_dimension()

    # Print basic information about the environment
    print(f"Custom Grid Environment:")
    print(f"Size: {custom_env.rows}x{custom_env.cols}")  # Grid size
    print(f"State Dim: {n_observations_custom}")  # State dimension (2 for row and column)
    print(f"Action Dim: {n_actions_custom}")  # Number of possible actions (4)
    print(f"Start State: {custom_env.start_state}")  # Starting position
    print(f"Goal State: {custom_env.goal_state}")  # Goal position

    # Reset the environment and print the normalized state tensor for the start state
    print(f"Example state tensor for (0,0): {custom_env.reset()}")

    # Take an example step: move 'right' (action=3) and print the result
    next_s, r, d = custom_env.step(3)  # Action 3 corresponds to moving right
    print(f"Step result (action=right): next_state={next_s.cpu().numpy()}, reward={r}, done={d}")

    # Take another example step: move 'up' (action=0) and print the result
    # This should hit a wall since the agent is at the top row
    next_s, r, d = custom_env.step(0)  # Action 0 corresponds to moving up
    print(f"Step result (action=up): next_state={next_s.cpu().numpy()}, reward={r}, done={d}")