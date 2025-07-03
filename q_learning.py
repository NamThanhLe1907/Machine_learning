import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional

class Environment:
    def __init__(self,
                 grid_size: Tuple[int,int],
                 row: int,
                 col: int,
                 start: Tuple[int,int],
                 goal: Tuple[int,int],
                 obstacles: List[Tuple[int,int]],
                 action_space: Optional[List[str]] = None,
                 step_penalty: float = -0.01,
                 goal_reward: float = 10):
        self.grid_size = grid_size(row, col)
        self.start = start
        self.goal = goal
        self.obstacles = obstacles or []
        self.action_space = action_space or ["up", "down", "left", "right"]
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward

        self.current_state = self.start

    def reset(self):
        self.current_state = self.start
        return self.current_state
    def get_valid_action(self, state: Tuple[int,int]) -> List[str]:
        valid_actions = []
        return valid_actions
    def step(self,
              row: int,
              col: int,
              state: Tuple[int,int],
              action: str,
              reward: float,
              done: bool) -> Tuple[Tuple[int,int], float, bool]:
        action = self.action_space[action]
        state = self.get_next_state(state, action)
        if action == "up" and  row > 0:
            row -= 1
        elif action == "down" and row < 0:
            row += 1
        elif action == "left" and col > 0:
            col -= 1
        elif action == "right" and col < 0:
            col += 1

        return state, reward, done