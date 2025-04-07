"""
Utility functions for RL algorithms
"""
import sys
from pathlib import Path
import gymnasium as gym
import numpy as np
from dataclasses import dataclass
import os
from typing import Optional
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
import torch, random
from pcse_gym import wrappers
import copy
sys.path.append(str(Path(__file__).parent.parent))
import utils

@dataclass 
class RL_Args:
    # Experiment configuration
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    wandb_project_name: str = "npk"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

class Agent(ABC):
    """
    Abstract agent class to enforce the get_action function
    """
    
    @abstractmethod
    def get_action(self, obs):
        pass

def make_env(kwargs, idx=1, capture_video=False, run_name=None):
    """
    Environment constructor for SyncVectorEnv
    """
    def thunk():
        if capture_video and idx == 0:
            env = utils.make_gym_env(kwargs, run_name=run_name)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = utils.make_gym_env(kwargs, run_name=run_name)
        env = utils.wrap_env_reward(env, kwargs)
        env = wrappers.NormalizeObservation(env)
        env = wrappers.NormalizeReward(env)
        return env
    return thunk

def setup(kwargs, args, run_name): 

    if kwargs.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"{kwargs.save_folder}{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(kwargs, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    return writer, device, envs

def eval_policy(policy, eval_env, kwargs, device, eval_episodes=5):
    """
    Evaluate a policy. Don't perform domain randomization (ie evaluate performance on the base environment)
    And don't perform limited weather resets (ie evaluate performance on the full weather data)
    """
    avg_reward = 0.

    if isinstance(eval_env, gym.vector.SyncVectorEnv):
        env_constr = type(eval_env.envs[0].unwrapped)
        args = eval_env.envs[0].unwrapped.args
        base_fpath = eval_env.envs[0].unwrapped.base_fpath
        agro_fpath = eval_env.envs[0].unwrapped.agro_fpath
        site_fpath = eval_env.envs[0].unwrapped.site_fpath
        crop_fpath = eval_env.envs[0].unwrapped.crop_fpath
        name_fpath = eval_env.envs[0].unwrapped.name_fpath
        unit_fpath = eval_env.envs[0].unwrapped.unit_fpath
        range_fpath = eval_env.envs[0].unwrapped.range_fpath
        render_mode = eval_env.envs[0].unwrapped.render_mode
        config = eval_env.envs[0].unwrapped.config
    else:
        env_constr = type(eval_env.unwrapped)
        args = eval_env.unwrapped.args
        base_fpath = eval_env.unwrapped.base_fpath
        agro_fpath = eval_env.unwrapped.agro_fpath
        site_fpath = eval_env.unwrapped.site_fpath
        crop_fpath = eval_env.unwrapped.crop_fpath
        name_fpath = eval_env.unwrapped.name_fpath
        unit_fpath = eval_env.unwrapped.unit_fpath
        range_fpath = eval_env.unwrapped.range_fpath
        render_mode = eval_env.unwrapped.render_mode
        config = eval_env.unwrapped.config

    new_args = copy.deepcopy(args)
    new_args.random_reset = True
    new_args.train_reset = False
    new_args.domain_rand = False
    env = env_constr(new_args, base_fpath, agro_fpath, site_fpath, crop_fpath, name_fpath, unit_fpath, range_fpath, render_mode, config)
    
    env = utils.wrap_env_reward(env, kwargs)
    env = wrappers.NormalizeObservation(env)
    env = wrappers.NormalizeReward(env)
    
    for i in range(eval_episodes):
        
        state, _, term, trunc = *env.reset(), False, False
        while not (term or trunc):
            if isinstance(state, np.ndarray):
                state = torch.Tensor(state).reshape((-1, *env.observation_space.shape)).to(device)
            action = policy.get_action(state)
            state, reward, term, trunc, _ = env.step(action.detach().cpu().numpy())

            if isinstance(eval_env, gym.vector.SyncVectorEnv):
                avg_reward += eval_env.envs[0].unnormalize(reward)
            else: 
                avg_reward += eval_env.unnormalize(reward)
    
    avg_reward /= eval_episodes
    return avg_reward
