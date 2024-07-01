import torch
from torch import nn, optim
import numpy as np
import os
import brax
from brax import envs
from brax.envs.wrappers import gym as gym_wrapper

from typing import Any, Callable, Dict, Optional, Sequence
import matplotlib.pyplot as plt
from brax.io import image
import numpy as np

import imageio


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

def load_expert_data(path, env, num_envs, ep_len):
    state_path = os.path.join(path, f"{env}_traj_state.npy")
    action_path = os.path.join(path, f"{env}_traj_action.npy")
    states = np.load(state_path)[:ep_len][:, None, ...].repeat(num_envs, axis=1)
    actions = np.load(action_path)[:ep_len][:, None, ...].repeat(num_envs, axis=1)
    return torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32)

def train(agent, states, actions, optimizer, criterion, epochs=10, batch_size=64):
    print("States Size: ", states.shape)
    dataset = torch.utils.data.TensorDataset(states, actions)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for state_batch, action_batch in dataloader:
            optimizer.zero_grad()
            print("State Batch Size: ", state_batch.shape)
            print("Action Batch Size: ", action_batch.shape)
            action_pred = agent(state_batch)
            loss = criterion(action_pred, action_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")


def render_state(env, state):
    """Render the environment state to an RGB array."""
    qp = QP(pos=jp.array(state['pos']),
                 rot=jp.array(state['rot']),
                 vel=jp.array(state['vel']),
                 ang=jp.array(state['ang']))
    return image.render(env.sys, qp)

def simulate_and_render(env, model, initial_state, actions=None, num_steps=100):
    frames = []
    print("Initial State: ", initial_state.shape)
    state = torch.tensor(initial_state, dtype=torch.float32)
    for i in range(num_steps):
        frames.append(render_state(env, state.numpy()))
        if actions is not None:
            action = torch.tensor(actions[i], dtype=torch.float32)
        else:
            action = model(state)
        state, _, _, _ = env.step(state, action)
    return frames

def create_gif(frames, filename):
    with imageio.get_writer(filename, mode='I', fps=20) as writer:
        for frame in frames:
            writer.append_data(frame)

def create_side_by_side(frames1, frames2, filename):
    combined_frames = [np.hstack((f1, f2)) for f1, f2 in zip(frames1, frames2)]
    create_gif(combined_frames, filename)

def simulate_and_collect_states(env, model, initial_state, num_steps=100):
    """Simulate the environment and collect Brax states."""
    states = []
    state = initial_state
    for _ in range(num_steps):
        action = model(state)  # Ensure model output matches what env.step expects
        state = env.step(state, action)
        states.append(state)
    return states

def render_states(sys, states, fmt='gif'):
    """Render states to an image or sequence of images."""
    print("Rendering Size: ", states.shape)
    try:
        # Render and return the image data
        image_data = image.render(sys, states, fmt=fmt)
        return image_data
    except Exception as e:
        print(f"Failed to render states: {e}")
        return None



# Configuration

num_envs= 2048
episode_length = 1000
env_name = "reacher"
env = envs.create(env_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# policy_layers = [
#     env.observation_space.shape[-1], 64, 64, env.action_space.shape[-1] * 2
# ]

# Hardcoded 52 for the given expert data
policy_layers = [
    52, 64, 64, 2 
]
# Initialize the agent and environment
agent = SimpleBCAgent(policy_layers).to(device)
optimizer = optim.Adam(agent.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Load data
sample_data_path = "/home/zhangel9/snap/snapd-desktop-integration/83/Desktop/Working/Behaviour-Cloning/BehavoiurCloningTrajectories/policy/brax_task/expert"
num_envs = 2048
ep_len = 128
expert_states, expert_actions = load_expert_data(sample_data_path, env_name, num_envs, ep_len)
expert_states, expert_actions = expert_states.to(device), expert_actions.to(device)

# Train the model
train(agent, expert_states, expert_actions, optimizer, criterion, epochs=10, batch_size=128)

# Evaluate the model

# Generate frames
print("Expert States Shape: ", expert_states.shape)
# trained_frames = simulate_and_render(env, agent, expert_states[0])
image_data = render_states(env.sys, expert_states)

# Create Gifs
if image_data:
    image_path = 'output_animation.gif'
    with open(image_path, 'wb') as f:
        f.write(image_data)
    print(f"Saved rendered animation to {image_path}")
else:
    print("No image data to save.")

