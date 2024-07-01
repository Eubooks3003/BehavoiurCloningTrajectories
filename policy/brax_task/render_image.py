import torch
from torch import nn, optim
import numpy as np
import os
import brax
from brax import envs
from brax.io import html
from brax.envs.wrappers import gym as gym_wrapper

from typing import Any, Callable, Dict, Optional, Sequence
import matplotlib.pyplot as plt
from brax.io import image
import numpy as np

import imageio

from IPython.display import HTML, clear_output


def render_trajectory(env, states, env_index):
    """
    Renders one environment's trajectory across all time steps.
    """
    trajectory = states[:, env_index, :]  # Extract trajectory for one environment

    # Assuming each state needs to be converted to a Brax "State" object
    # This is hypothetical; actual conversion depends on Brax's current API
    trajectory_states = [state for state in trajectory]

    # Render each state to an image (using hypothetical Brax rendering functionality)
    images = [image.render(env.sys, state) for state in trajectory_states]

    # Combine images into a single figure (using PIL or similar library)
    frame_images = [Image.open(io.BytesIO(img)) for img in images]
    return combine_images_into_one(frame_images)

def load_expert_data(path, env, num_envs, ep_len):
    state_path = os.path.join(path, f"{env}_traj_state.npy")
    action_path = os.path.join(path, f"{env}_traj_action.npy")
    states = np.load(state_path)[:ep_len][:, None, ...].repeat(num_envs, axis=1)
    actions = np.load(action_path)[:ep_len][:, None, ...].repeat(num_envs, axis=1)
    return torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32)

def combine_images_into_one(images):
    """
    Combine a list of PIL Image objects into a single Image object.
    """
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im

def save_image(image, filename):
    """ Save PIL Image to file """
    image.save(filename)

def display_image(filename):
    """ Display image from file using matplotlib """
    img = plt.imread(filename)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

env_name = "reacher"
env = envs.create(env_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sample_data_path = "/home/zhangel9/snap/snapd-desktop-integration/83/Desktop/Working/Behaviour-Cloning/BehavoiurCloningTrajectories/policy/brax_task/expert"
num_envs = 2048
ep_len = 128
expert_states, expert_actions = load_expert_data(sample_data_path, env_name, num_envs, ep_len)
expert_states, expert_actions = expert_states.to(device), expert_actions.to(device)

env_index = 0  # Index of the environment trajectory to render
# rendered_image = render_trajectory(env, expert_states, env_index)

# # Save and/or display the rendered image
# save_image(rendered_image, 'rendered_trajectory.png')
# display_image('rendered_trajectory.png')

HTML(html.render(env.sys.tree_replace({'opt.timestep': env.dt}), expert_states))

