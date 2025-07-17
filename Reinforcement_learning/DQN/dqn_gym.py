import numpy as np
import matplotlib.pyplot as plt
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple, deque
from typing import List, Tuple, Dict, Optional

import gymnasium as gym
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
steps_done_custom = 0

class DQN(nn.Module):
    def __init__(self,n_observations: int, n_actions: int):
        super(DQN, self).__init__()
        # CartPole has 4 observations, 2 actions 0 1
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """
        Save a transition to the memory.
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Sample a batch of transitions from memory.
        """
        return random.sample(self.memory, batch_size)
    def __len__(self) -> int:
        """
        Returns the current size of the memory.
        """
        return len(self.memory)
    

def optimize_model(memory: ReplayMemory,
                   policy_net: nn.Module,
                   target_net: nn.Module,
                   optimizer: optim.Optimizer,
                   batch_size: int,
                   gamma: float,
                   criterion: nn.Module = nn.SmoothL1Loss()) -> Optional[float]:
    """
    Performs one step of optimization on the policy network using a batch of transitions from the replay memory.
    """
    if len(memory) < batch_size:
        return None
    
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions)) #Unpack transitions into seperate components.

    #Identify non-final states (states that are not terminal)
    non_final_mask = torch.tensor(Tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    
    #Stack non-final next states into a tensor
    if any(non_final_mask):
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
    
    #Stack current states, actions, rewards, and done flags into tensors
    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    done_batch = torch.cat(batch.done)
    
    #Compute Q(s_t, a) for the actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    #Compute V(s_{t+1}) for the next states using the target network
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        if any(non_final_mask):
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    #Compute the expected Q values using the Bellman equation
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    #Compute the loss betweeen the predicted and expected Q values
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    #Perform backpropagation and optimization
    optimizer.zero_grad() # Clear previous gradients
    loss.backward() # Backpropagate the loss
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 100) # Clip gradients to prevent explosion
    optimizer.step() # Update the policy network parameters

    return loss.item()  # Return the loss value for monitoring

def update_target_net(policy_net: nn.Module,
                      target_net: nn.Module) -> None:
    """
    Copies the weights from the policy network to the target network.
    """

    target_net.load_state_dict(policy_net.state_dict())


def select_action_custom(state: torch.Tensor,
                         policy_net: nn.Module,
                         epsilon_start: float,
                         epsilon_end: float,
                         epsilon_decay: int,
                         n_actions: int) -> Tuple[torch.Tensor, float]:
    """
    Select an action using the epsilon-greedy strategy for a single state tensor.
    """

    global steps_done_custom #counter to track the number of steps taken
    sample = random.random() #Generate a random number for epsilon-greedy decision.
    #Compute the current epsilon value based on the decay formula.
    epsilon_threshold = epsilon_end + (epsilon_start - epsilon_end) * \
        math.exp(-1. * steps_done_custom / epsilon_decay)
    steps_done_custom += 1  # Increment the step counter

    if sample > epsilon_threshold:
        #Exploitation: Choose the action with the highest Q-value
        with torch.no_grad():
            #Add a batch dimension to the state tensor to make it [1, state_dim]
            state_batch = state.unsqueeze(0)
            #Get the action with thet maximum Q-value (output shape: [1, n_actions])
            action = policy_net(state_batch).max(1)[1].view(1, 1)

    else:
        #Exploration: Choose a random action
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    return action, epsilon_threshold    

class CartPoleEnv:
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.n_observations = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

    def reset(self):
        obs, info = self.env.reset()
        return torch.tensor(obs, dtype=torch.float32, device=device)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action.item())
        done = terminated or truncated
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        return obs_tensor, reward, done
def test_trained_agent(policy_net, num_episodes=5):
    """Test agent với GUI"""
    test_env = gym.make("CartPole-v1", render_mode="human")
    
    for episode in range(num_episodes):
        obs, info = test_env.reset()
        state = torch.tensor(obs, dtype=torch.float32, device=device)
        total_reward = 0
        
        print(f"\n=== Test Episode {episode + 1} ===")
        
        for step in range(500):
            # Chọn action tốt nhất (không random)
            with torch.no_grad():
                action = policy_net(state.unsqueeze(0)).max(1)[1].view(1, 1)
            
            obs, reward, terminated, truncated, info = test_env.step(action.item())
            total_reward += reward
            state = torch.tensor(obs, dtype=torch.float32, device=device)
            
            test_env.render()
            
            if terminated or truncated:
                break
                
        print(f"Episode {episode + 1}: {total_reward} points in {step + 1} steps")
    
    test_env.close()
if __name__ == "__main__":
    # Hyperparameters cho CartPole
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY = 1000
    TAU = 0.0075
    LR = 5e-4
    MEMORY_CAPACITY = 5000
    TARGET_UPDATE_FREQ = 20
    NUM_EPISODES = 200
    
    # Tạo environment
    env = CartPoleEnv()
    n_actions = env.n_actions
    n_observations = env.n_observations
    
    # Tạo networks
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    # Optimizer và memory
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEMORY_CAPACITY)
    
    # Tracking lists
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    
    print("Starting CartPole DQN Training...")


    # Training loop
    for episode in tqdm(range(NUM_EPISODES)):
        state = env.reset()
        total_reward = 0
        losses = []
        
        for step in range(500):  # CartPole max steps = 500
            # Select action
            action, epsilon = select_action_custom(
                state, policy_net, EPS_START, EPS_END, EPS_DECAY, n_actions
            )
            
            # Take step
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # Store in memory
            reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)
            done_tensor = torch.tensor([done], device=device, dtype=torch.bool)
            
            memory_next_state = next_state if not done else None
            memory.push(state, action, memory_next_state, reward_tensor, done_tensor)
            
            state = next_state
            
            # Optimize
            loss = optimize_model(memory, policy_net, target_net, optimizer, BATCH_SIZE, GAMMA)
            if loss is not None:
                losses.append(loss)
                
            if done:
                break
        
        # Store episode stats
        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)
        episode_losses.append(np.mean(losses) if losses else 0)
        
        # Update target network
        if episode % TARGET_UPDATE_FREQ == 0:
            update_target_net(policy_net, target_net)
            
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}/{NUM_EPISODES} | Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.3f}")

        # Plotting
    plt.figure(figsize=(15, 5))
    
    # Rewards
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Episode lengths  
    plt.subplot(1, 3, 2)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    
    # Losses
    plt.subplot(1, 3, 3)
    plt.plot(episode_losses)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("Training completed!")

        # Thêm vào cuối file sau plotting:
    print("Training completed!")
    print("Starting visual test...")
    test_trained_agent(policy_net, num_episodes=3)

    