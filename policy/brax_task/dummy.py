import torch
from torch import nn, optim
import numpy as np
import os
from typing import Any, Callable, Dict, Optional, Sequence

class SimpleBCAgent(nn.Module):
    def __init__(self,  policy_layers: Sequence[int]):
        super(SimpleBCAgent, self).__init__()
        policy = []
        for w1, w2 in zip(policy_layers, policy_layers[1:]):
            policy.append(nn.Linear(w1, w2))
            policy.append(nn.SiLU())
        policy.pop()  # drop the final activation
        self.policy = nn.Sequential(*policy)
    
    def forward(self, x):
        return self.policy(x)

def load_expert_data(path, env):
    # Load expert data
    states = np.load(os.path.join(path, f"{env}_traj_state.npy"))
    actions = np.load(os.path.join(path, f"{env}_traj_action.npy"))
    return torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32)

def train_model(agent, states, actions, optimizer, epochs=10, batch_size=64):
    criterion = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(states, actions)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for state_batch, action_batch in dataloader:
            optimizer.zero_grad()
            action_pred = agent(state_batch)
            loss = criterion(action_pred, action_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

# Example usage
device = 'cuda' if torch.cuda.is_available() else 'cpu'
policy_layers = [
    env.observation_space.shape[-1], 64, 64, env.action_space.shape[-1] * 2
]
input_dim = 17  # Example input dimension, adjust as necessary
output_dim = 6   # Example output dimension, adjust as necessary
agent = SimpleBCAgent(input_dim, output_dim).to(device)
optimizer = optim.Adam(agent.parameters(), lr=0.001)

sample_data_path = "/home/zhangel9/snap/snapd-desktop-integration/83/Desktop/Working/Behaviour-Cloning/BehavoiurCloningTrajectories/policy/brax_task/expert"
env_name = "reacher"
states, actions = load_expert_data(sample_data_path, env_name)
states, actions = states.to(device), actions.to(device)

train_model(agent, states, actions, optimizer, epochs=100, batch_size=128)
