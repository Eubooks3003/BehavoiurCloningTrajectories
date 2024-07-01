#@title Import Brax and some helper modules
from IPython.display import clear_output

import collections
from datetime import datetime
import functools
import math
import os
import time
from typing import Any, Callable, Dict, Optional, Sequence

import brax

from brax import envs
from brax.envs.wrappers import gym as gym_wrapper
from brax.envs.wrappers import torch as torch_wrapper
from brax.io import metrics
from brax.training.agents.ppo import train as ppo
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import jax

class Agent(nn.Module):
  """Standard PPO Agent with GAE and observation normalization."""

  def __init__(self,
               policy_layers: Sequence[int],
               entropy_cost: float,
               discounting: float,
               reward_scaling: float,
               device: str):
    super(Agent, self).__init__()

    policy = []
    for w1, w2 in zip(policy_layers, policy_layers[1:]):
      policy.append(nn.Linear(w1, w2))
      policy.append(nn.SiLU())
    policy.pop()  # drop the final activation
    self.policy = nn.Sequential(*policy)
    self.num_steps = torch.zeros((), device=device)
    self.running_mean = torch.zeros(policy_layers[0], device=device)
    self.running_variance = torch.zeros(policy_layers[0], device=device)

    self.entropy_cost = entropy_cost
    self.discounting = discounting
    self.reward_scaling = reward_scaling
    self.lambda_ = 0.95
    self.epsilon = 0.3
    self.device = device


  def dist_create(self, logits):
    """Normal followed by tanh.

    torch.distribution doesn't work with torch.jit, so we roll our own."""
    loc, scale = torch.split(logits, logits.shape[-1] // 2, dim=-1)
    scale = F.softplus(scale) + .001
    return loc, scale


  def dist_sample_no_postprocess(self, loc, scale):
    return torch.normal(loc, scale)

  @classmethod
  def dist_postprocess(cls, x):
    return torch.tanh(x)

  def dist_entropy(self, loc, scale):
    log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
    entropy = 0.5 + log_normalized
    entropy = entropy * torch.ones_like(loc)
    dist = torch.normal(loc, scale)
    log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
    entropy = entropy + log_det_jacobian
    return entropy.sum(dim=-1)

  def dist_log_prob(self, loc, scale, dist):
    log_unnormalized = -0.5 * ((dist - loc) / scale).square()
    log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
    log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
    log_prob = log_unnormalized - log_normalized - log_det_jacobian
    return log_prob.sum(dim=-1)

  def update_normalization(self, observation):
    self.num_steps += observation.shape[0] * observation.shape[1]
    input_to_old_mean = observation - self.running_mean
    mean_diff = torch.sum(input_to_old_mean / self.num_steps, dim=(0, 1))
    self.running_mean = self.running_mean + mean_diff
    input_to_new_mean = observation - self.running_mean
    var_diff = torch.sum(input_to_new_mean * input_to_old_mean, dim=(0, 1))
    self.running_variance = self.running_variance + var_diff

  def normalize(self, observation):
    variance = self.running_variance / (self.num_steps + 1.0)
    variance = torch.clip(variance, 1e-6, 1e6)
    return ((observation - self.running_mean) / variance.sqrt()).clip(-5, 5)

  def get_logits_action(self, observation):
    observation = self.normalize(observation)
    logits = self.policy(observation)
    loc, scale = self.dist_create(logits)
    action = self.dist_sample_no_postprocess(loc, scale)
    return logits, action

  def chamfer_loss(self, pred, target):
        pred = pred.unsqueeze(2)  # (batch_size, seq_len, 1, feature_dim)
        target = target.unsqueeze(1)  # (batch_size, 1, seq_len, feature_dim)
        pred = pred.repeat(1, 1, target.shape[2], 1)
        target = target.repeat(1, pred.shape[1], 1, 1)
        distances = torch.sqrt(((pred - target) ** 2).mean(dim=-1))  # (batch_size, seq_len, seq_len)
        min_distances, _ = distances.min(dim=-1)
        return min_distances.mean()

  def norm_grad(self, tensor: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        grad_norm = torch.sqrt(torch.sum(tensor ** 2, dim=-1, keepdim=True))
        return tensor / (grad_norm + epsilon)
 
  def do_one_step(self, state, env):
        # assert hasattr(env, 'step'), "env does not have a step method, likely not an environment object in do one step."
    
        normalized_obs = self.normalize(state['obs'].to(self.device))
        logits = self.policy(normalized_obs[:2])
        print("Logits Shape: ", logits.shape)
        loc, scale = self.dist_create(logits)
        print("Loc shape: ", loc.shape)
        actions = self.dist_sample_no_postprocess(loc, scale).cpu()
        print("Action shape before reshape: ", actions.shape)
        actions = actions.view(-1, actions.shape[-1]).to(self.device)
        print("Action shape: ", actions.shape)
        print("state shape: ", state['obs'].shape)
        observation, reward, done, info = env.step(actions.to(self.device))
    
        batch_size = state['obs'].shape[0]
        observation = observation.view(batch_size, -1, observation.shape[-1])
        reward = reward.view(batch_size, -1)
        done = done.view(batch_size, -1)

        next_state = {
            'obs': observation,
            'reward': reward,
            'done': done,
            'pos': info['pos'].view(batch_size, -1, info['pos'].shape[-1]),
            'rot': info['rot'].view(batch_size, -1, info['rot'].shape[-1]),
            'vel': info['vel'].view(batch_size, -1, info['vel'].shape[-1]),
            'ang': info['ang'].view(batch_size, -1, info['ang'].shape[-1])
        }


        actions = norm_grad(actions)
        next_state = {k: norm_grad(v) for k, v in next_state.items()}

        qp_list = {'pos': nstate['pos'], 'rot': nstate['rot'], 'vel': nstate['vel'], 'ang': nstate['ang']}

        return (nstate, params, normalizer_params, key), (next_state['reward'], state, qp_list, logits, actions)

  def loss(self, td: Dict[str, torch.Tensor], demo_traj, demo_traj_action, env):

        print("Hello")
        assert hasattr(env, 'step'), "env does not have a step method, likely not an environment object."
    
        observation = self.normalize(td['observation'])
        print("Keys in td:", td.keys())
        for key in td:
            print(f"Size of tensor associated with {key}: {td[key].shape}")
        print("Td observation: ", td['observation'].shape)
        policy_logits = self.policy(observation[:2])

        # Initial state
        state = {'obs': td['observation'], 'pos': torch.zeros_like(td['observation']), 'rot': torch.zeros_like(td['observation']),
                    'vel': torch.zeros_like(td['observation']), 'ang': torch.zeros_like(td['observation'])}

        # Roll out the trajectory
        rewards = []
        obs = []
        qp_list = {'pos': [], 'rot': [], 'vel': [], 'ang': []}
        logit_list = []
        action_list = []

        for step_index in range(len(demo_traj)):
            state, logits, actions = self.do_one_step(state, env)
            rewards.append(state['reward'])
            obs.append(state['obs'])
            qp_list['pos'].append(state['pos'])
            qp_list['rot'].append(state['rot'])
            qp_list['vel'].append(state['vel'])
            qp_list['ang'].append(state['ang'])
            logit_list.append(logits)
            action_list.append(actions)

        rollout_traj = torch.cat([torch.stack(qp_list['pos']).reshape((len(demo_traj), state['obs'].shape[0], -1)),
                                    torch.stack(qp_list['rot']).reshape((len(demo_traj), state['obs'].shape[0], -1)),
                                    torch.stack(qp_list['vel']).reshape((len(demo_traj), state['obs'].shape[0], -1)),
                                    torch.stack(qp_list['ang']).reshape((len(demo_traj), state['obs'].shape[0], -1))], dim=-1)

        self.update_normalization(observation)
        # demo_traj_ = self.normalize(demo_traj)

        # Calculate state chamfer loss
        cf_loss = self.chamfer_loss(rollout_traj, demo_traj)

        # Calculate action chamfer loss
        pred_action = td['action']
        pred_demo_action = demo_traj_action
        cf_action_loss = self.chamfer_loss(pred_action, pred_demo_action)

        # Calculate entropy loss
        loc, scale = torch.chunk(policy_logits, 2, dim=-1)
        sigma_list = F.softplus(scale) + 1e-6  # Small epsilon for numerical stability
        entropy_loss = -0.5 * torch.log(2 * torch.pi * sigma_list ** 2).mean()

        # Combine losses
        final_loss = cf_loss + cf_action_loss + self.entropy_cost * entropy_loss
        final_loss = torch.tanh(final_loss)

        return final_loss

# Use CPU for JAX
import os
import jax.numpy as jnp
import numpy as np
import torch

os.environ['JAX_PLATFORM_NAME'] = 'gpu'
# Define the arguments
class Args:
    def __init__(self):
        self.env = 'reacher'
        self.ep_len = 128
        self.num_envs = 2048
        self.lr = 3e-4
        self.trunc_len = 5
        self.max_it = 1000
        self.max_grad_norm = 0.5
        self.reverse_discount = 0.99
        self.entropy_factor = 0.01
        self.deviation_factor = 1.0
        self.action_cf_factor = 1.0
        self.l2 = 0.0
        self.il = 0.0
        self.ILD = 0.0
        self.seed = 42

args = Args()

# Set the log directory
args.logdir = f"logs/{args.env}/{args.env}_ep_len{args.ep_len}_num_envs{args.num_envs}_lr{args.lr}_trunc_len{args.trunc_len}" \
              f"_max_it{args.max_it}_max_grad_norm{args.max_grad_norm}_re_dis{args.reverse_discount}_ef_{args.entropy_factor}" \
              f"_df_{args.deviation_factor}_acf_{args.action_cf_factor}_l2loss_{args.l2}_il_{args.il}_ILD_{args.ILD}" \
              f"/seed{args.seed}"

# Define the path to the sample_data directory
sample_data_path = "/home/zhangel9/snap/snapd-desktop-integration/83/Desktop/Working/Behaviour-Cloning/BehavoiurCloningTrajectories/policy/brax_task/expert"

# Load the expert data
print("Loading expert data...")

# Actual Trajectory?
'''
        _, (rewards, obs, qp_list, logit_list, action_list) = jax.lax.scan(
            do_one_step, (state, params, normalizer_params, key),
            (jnp.array(range(episode_length // action_repeat))),
            length=episode_length // action_repeat)

        rollout_traj = jnp.concatenate([qp_list.pos.reshape((qp_list.pos.shape[0], qp_list.pos.shape[1], -1)),
                                        qp_list.rot.reshape((qp_list.rot.shape[0], qp_list.rot.shape[1], -1)),
                                        qp_list.vel.reshape((qp_list.vel.shape[0], qp_list.vel.shape[1], -1)),
                                        qp_list.ang.reshape((qp_list.ang.shape[0], qp_list.ang.shape[1], -1))], axis=-1)
'''
demo_traj = np.load(os.path.join(sample_data_path, f"{args.env}_traj_state.npy"))
demo_traj = jnp.array(demo_traj)[:args.ep_len][:, None, ...].repeat(args.num_envs, 1)
print("Demo Trajectory Shape: ", demo_traj.shape)

# Actions
demo_traj_action = np.load(os.path.join(sample_data_path, f"{args.env}_traj_action.npy"))
demo_traj_action = jnp.array(demo_traj_action)[:args.ep_len][:, None, ...].repeat(args.num_envs, 1)
print("Demo Trajectory Action Shape: ", demo_traj_action.shape)

# Observations
demo_traj_obs = np.load(os.path.join(sample_data_path, f"{args.env}_traj_obs.npy"))
demo_traj_obs = jnp.array(demo_traj_obs)[:args.ep_len][:, None, ...].repeat(args.num_envs, 1)
print("Demo Trajectory Observation Shape: ", demo_traj_obs.shape)

reverse_discounts = jnp.array([args.reverse_discount ** i for i in range(args.ep_len, 0, -1)])[None, ...]
reverse_discounts = reverse_discounts.repeat(args.num_envs, 0)

# Convert JAX arrays to PyTorch tensors
print("Converting JAX arrays to PyTorch tensors...")
demo_traj = torch.tensor(np.array(demo_traj), dtype=torch.float32)
demo_traj_action = torch.tensor(np.array(demo_traj_action), dtype=torch.float32)
demo_traj_obs = torch.tensor(np.array(demo_traj_obs), dtype=torch.float32)
reverse_discounts = torch.tensor(np.array(reverse_discounts), dtype=torch.float32)

print("Expert data loaded and converted successfully.")


StepData = collections.namedtuple(
    'StepData',
    ('observation', 'logits', 'action', 'reward', 'done', 'truncation'))

def sd_map(f: Callable[..., torch.Tensor], *sds) -> StepData:
    """Map a function over each field in StepData."""
    items = {}
    keys = sds[0]._asdict().keys()
    for k in keys:
        items[k] = f(*[sd._asdict()[k] for sd in sds])
    return StepData(**items)

def eval_unroll(agent, env, length):
    observation = env.reset()
    episodes = torch.zeros((), device=agent.device)
    episode_reward = torch.zeros((), device=agent.device)
    for _ in range(length):
        _, action = agent.get_logits_action(observation)
        action_cpu = Agent.dist_postprocess(action).cpu()
        observation, reward, done, _ = env.step(action_cpu)
        episodes += torch.sum(done)
        episode_reward += torch.sum(reward)
    return episodes, episode_reward / episodes

def train_unroll(agent, env, observation, num_unrolls, num_steps, unroll_length):
    """Return step data over multiple unrolls."""
    device = observation.device
    sd = StepData([], [], [], [], [], [])
    for unroll_index in range(num_unrolls):
        one_unroll = StepData([observation], [], [], [], [], [])
        for step_index in range(unroll_length):
            logits, action = agent.get_logits_action(observation)
            action_cpu = Agent.dist_postprocess(action).cpu()
            observation, reward, done, info = env.step(action_cpu)

            # Move tensors to the same device before appending
            one_unroll = one_unroll._replace(
                observation=one_unroll.observation + [observation.to(device)],
                logits=one_unroll.logits + [logits.to(device)],
                action=one_unroll.action + [action.to(device)],
                reward=one_unroll.reward + [reward.to(device)],
                done=one_unroll.done + [done.to(device)],
                truncation=one_unroll.truncation + [info['truncation'].to(device)]
            )
            
            print(f"Unroll {unroll_index+1}/{num_unrolls}, Step {step_index+1}/{unroll_length}:")
            print(f"  Observation shape: {observation.shape}")
            print(f"  Logits shape: {logits.shape}")
            print(f"  Action shape: {action.shape}")
            print(f"  Reward shape: {reward.shape}")
            print(f"  Done shape: {done.shape}")
            print(f"  Truncation shape: {info['truncation'].shape}")
            
        
        one_unroll = sd_map(torch.stack, one_unroll)
        sd = sd_map(lambda x, y: x + [y], sd, one_unroll)
        
        
        print(f"One unroll shape after stacking: Observation: {one_unroll.observation.shape}, "
              f"Logits: {one_unroll.logits.shape}, Action: {one_unroll.action.shape}, "
              f"Reward: {one_unroll.reward.shape}, Done: {one_unroll.done.shape}, "
              f"Truncation: {one_unroll.truncation.shape}")
              

    td = sd_map(torch.stack, sd)
    
    print(f"Final unrolled data shape: Observation: {td.observation.shape}, "
          f"Logits: {td.logits.shape}, Action: {td.action.shape}, "
          f"Reward: {td.reward.shape}, Done: {td.done.shape}, "
          f"Truncation: {td.truncation.shape}")
          
    return observation, td

def train(

    env_name: str = 'reacher',
    num_envs: int = 2048,
    episode_length: int = 1000,
    device: str = 'cpu',
    num_timesteps: int = 30_000_000,
    eval_frequency: int = 10,
    unroll_length: int = 5,
    batch_size: int = 1024,
    num_minibatches: int = 16,
    num_update_epochs: int = 4,
    reward_scaling: float = .1,
    entropy_cost: float = 1e-2,
    discounting: float = .97,
    learning_rate: float = 3e-4,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
):

    env = envs.create(env_name, batch_size=num_envs,
                        episode_length=episode_length,
                        backend='spring')
    env = gym_wrapper.VectorGymWrapper(env)
    # automatically convert between jax ndarrays and torch tensors:
    env = torch_wrapper.TorchWrapper(env, device=device)

    # env warmup
    env.reset()
    action = torch.zeros(env.action_space.shape).to(device)
    action_cpu = action.cpu()

    # Debug: Print action tensor details
    print("Initial PyTorch action tensor details:")
    print(f"Shape: {action.shape}, Device: {action.device}, Type: {action.dtype}")

    
    # env.step(action_cpu)

    print("Stepped through first action")

    # create the agent
    policy_layers = [
        env.observation_space.shape[-1], 64, 64, env.action_space.shape[-1] * 2
    ]

    agent = Agent(policy_layers, entropy_cost, discounting,
                    reward_scaling, device)
    agent = agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

    sps = 0
    total_steps = 0
    total_loss = 0


    for eval_i in range(eval_frequency + 1):
        if progress_fn:
            t = time.time()
            with torch.no_grad():
                episode_count, episode_reward = eval_unroll(agent, env, episode_length)
            duration = time.time() - t
            # TODO: only count stats from completed episodes
            episode_avg_length = env.num_envs * episode_length / episode_count
            eval_sps = env.num_envs * episode_length / duration
            progress = {
                'eval/episode_reward': episode_reward,
                'eval/completed_episodes': episode_count,
                'eval/avg_episode_length': episode_avg_length,
                'speed/sps': sps,
                'speed/eval_sps': eval_sps,
                'losses/total_loss': total_loss,
            }
            progress_fn(total_steps, progress)

        if eval_i == eval_frequency:
            break

        observation = env.reset()
        num_steps = batch_size * num_minibatches * unroll_length
        num_epochs = num_timesteps // (num_steps * eval_frequency)
        num_unrolls = batch_size * num_minibatches // env.num_envs
        total_loss = 0
        t = time.time()
        print(f"Starting training iteration {eval_i+1}/{eval_frequency}")

        for epoch in range(num_epochs):
                print(f"Epoch {epoch+1}/{num_epochs}")
                observation, td = train_unroll(agent, env, observation, num_unrolls, num_steps, unroll_length)
                print("First td: ", td[0].shape)
                # Make unroll first
                def unroll_first(data):
                    data = data.swapaxes(0, 1)
                    return data.reshape([data.shape[0], -1] + list(data.shape[3:]))
                td = sd_map(unroll_first, td)

                print("second td: ", td[0].shape)

                # Update normalization statistics
                agent.update_normalization(td.observation)

                for update_epoch in range(num_update_epochs):
                    print(f"Update Epoch {update_epoch+1}/{num_update_epochs}")
                    # Shuffle and batch the data
                    with torch.no_grad():
                        permutation = torch.randperm(td.observation.shape[1], device=device)
                        def shuffle_batch(data):
                            data = data[:, permutation]
                            data = data.reshape([data.shape[0], num_minibatches, -1] + list(data.shape[2:]))
                            return data.swapaxes(0, 1)
                        epoch_td = sd_map(shuffle_batch, td)

                    for minibatch_i in range(num_minibatches):
                        td_minibatch = sd_map(lambda d: d[minibatch_i], epoch_td)
                        print("third td: ", td_minibatch[0].shape)
                        loss = agent.loss(td_minibatch._asdict(), demo_traj, demo_traj_action, env)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        print(f"Minibatch {minibatch_i+1}/{num_minibatches}, Loss: {loss.item()}")

        duration = time.time() - t
        total_steps += num_epochs * num_steps
        total_loss = total_loss / (num_epochs * num_update_epochs * num_minibatches)
        sps = num_epochs * num_steps / duration

        print(f"Iteration {eval_i+1}/{eval_frequency} complete. Total Loss: {total_loss}, SPS: {sps}")

    print("Training complete!")


# temporary fix to cuda memory OOM
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

xdata = []
ydata = []
eval_sps = []
train_sps = []
times = [datetime.now()]

def progress(num_steps, metrics):
  times.append(datetime.now())
  xdata.append(num_steps)
  ydata.append(metrics['eval/episode_reward'].cpu())
  eval_sps.append(metrics['speed/eval_sps'])
  train_sps.append(metrics['speed/sps'])
  clear_output(wait=True)
  plt.xlim([0, 30_000_000])
  plt.ylim([0, 6000])
  plt.xlabel('# environment steps')
  plt.ylabel('reward per episode')
  plt.plot(xdata, ydata)
  plt.show()

train(progress_fn=progress)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')
print(f'eval steps/sec: {np.mean(eval_sps)}')
print(f'train steps/sec: {np.mean(train_sps)}')