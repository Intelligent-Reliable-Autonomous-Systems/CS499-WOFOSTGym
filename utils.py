"""File for utils functions. Importantly contains:
    - Args: Dataclass for configuring paths for the WOFOST Environment
    - get_gym_args: function for getting the required arguments for the gym 
    environment from the Args dataclass 

Written by: Will Solow, 2024
"""

import gymnasium as gym
import warnings
import numpy as np 
import torch
from dataclasses import dataclass, is_dataclass, fields, is_dataclass
from typing import Optional
from omegaconf import OmegaConf
import os

import pcse_gym.wrappers.wrappers as wrappers
from pcse_gym.args import NPK_Args
from inspect import getmembers, isclass, isfunction, getmodule
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class Args:
    """
    Dataclass for configuration a Gym environment
    """

    """Parameters for the NPK Gym Environment"""
    npk: NPK_Args

    """Environment ID"""
    env_id: str = "lnpkw-v0"
    """Env Reward Function. Do not specify for default reward"""
    env_reward: Optional[str] = None
    """Rendering mode. `None` or `human`"""
    render_mode: Optional[str] = "None"

    """Relative location of run, either agent or data, where to save config file 
        and associated run (policies, .npz files, etc)"""
    save_folder: Optional[str] = None
    """Name of data file to save save in save_folder"""
    data_file: Optional[str] = None
    
    """Agromanagement file"""
    agro_file: str = "wheat_agro.yaml"

    """Path configuration, generally do not change these """
    """Base filepath"""
    base_fpath: str = f"{os.getcwd()}/"
    """Relative path to agromanagement configuration file"""
    agro_fpath: str = "env_config/agro/"
    """Relative path to crop configuration folder"""
    crop_fpath: str = "env_config/crop/"
    """Relative path to site configuration foloder"""
    site_fpath: str = "env_config/site/"
    """Relative path to the state units """
    unit_fpath: str = "env_config/state_units.yaml"
    """Relative path to the state names"""
    name_fpath: str = "env_config/state_names.yaml"
    """Relative path to the state ranges for normalization"""
    range_fpath: str = "env_config/state_ranges.yaml"

HATCHES = [
   
    ".",      # Small dots
    "",
    "+",      # Crossing diagonal lines
     "O",      # Large circles
    "o",      # Small circles
    "O",      # Large circles
    "/",      # Diagonal lines (forward slash)
    "\\",     # Diagonal lines (backslash)
    "|",      # Vertical lines
    "*",      # Stars
    "-",      # Horizontal lines
    "x",      # Crossing lines (horizontal and vertical)
]

COLORS = [
    "#ff0000",  # Red
    "#0000ff",  # Blue
    "#daa520",  # Goldenrod 
    "#4B0082", # Dark violet
    #"#7700ff",  # Violet
    

    "#008000",  # Dark Green
    "#ff00ff",  # Magenta
    "#00ff00",  # Green
    
    "#00ffff",  # Cyan
    
   
    
    "#ff7700",  # Orange
    
    
    
    
    "#000000",  # Black
]

def wrap_env_reward(env: gym.Env, args):
    """
    Function to wrap the environment with a given reward function
    Based on the reward functions created in the pcse_gym/wrappers/
    """
    # Default environment
    if not args.env_reward:
        return env
    # Reward wrapper
    try:
        reward_constr = get_reward_wrappers(wrappers)[args.env_reward]
        return reward_constr(env, args)
    # Incorrectly specified reward
    except:
        msg = f"Incorrectly specified RewardWrapper args.env_reward: `{args.env_reward}`"
        raise Exception(msg)

def make_gym_env(args, run_name=None):
    """
    Make a gym environment. Ensures that OrderEnforcing and PassiveEnvChecker
    are not applied to environment
    """

    assert args.save_folder is not None, "Specify `save_folder` to save config file."
    assert isinstance(args.save_folder, str), f"`args.save_folder` must be of type `str` but is of type `{type(args.save_folder)}`."
    assert args.save_folder.endswith("/"), f"Folder args.save_folder `{args.save_folder}` must end with `/`"
    
    env_id, env_kwargs = get_gym_args(args)
    env = gym.make(env_id, **env_kwargs).unwrapped 

    # Save configuration
    os.makedirs(args.save_folder, exist_ok=True)
    
    # Save configuration
    if run_name is None:
        save_file = f"{args.save_folder}config.yaml"
    else:
        save_file = f"{args.save_folder}/{run_name}/config.yaml"
    with open(save_file, "w") as fp:
        OmegaConf.save(config=args, f=fp.name)

    return env

def get_gym_args(args: Args):
    """
    Returns the Environment ID and required arguments for the WOFOST Gym
    Environment

    Arguments:
        Args: Args dataclass
    """
    env_kwargs = {'args': correct_commandline_lists(args.npk), 'base_fpath': args.base_fpath, \
                  'agro_fpath': f"{args.agro_fpath}{args.agro_file}",'site_fpath': args.site_fpath, 
                  'crop_fpath': args.crop_fpath, 'unit_fpath':args.unit_fpath, 
                  'name_fpath':args.name_fpath, 'range_fpath':args.range_fpath, 'render_mode':args.render_mode}
    
    return args.env_id, env_kwargs

def correct_commandline_lists(d):
    """
    Correct any lists passed by command line
    """
    def iterate_dataclass(obj, prefix=""):
        if not is_dataclass(obj):
            return

        for field in fields(obj):
            value = getattr(obj, field.name)
            full_key = f"{prefix}.{field.name}" if prefix else field.name

            if is_dataclass(value):  # If the value is another dataclass, recurse
                iterate_dataclass(value, prefix=full_key)
            else:
                if isinstance(value, list):
                    if len(value) != 0:
                        if isinstance(value[0], str) and len(value) == 1:
                            values = value[0].split(",")
                            for i, v in enumerate(values):
                                values[i] = v.strip("[], ")
                                try:
                                    values[i] = float(values[i])
                                except:
                                    pass
                            setattr(obj, field.name, values)
                        elif isinstance(value[0], str):
                            for i,v in enumerate(value):
                                value[i] = v.strip("[], ")
                                try:
                                    value[i] = float(value[i])
                                except:
                                    pass
                            setattr(obj, field.name, value)   
        
    iterate_dataclass(d)
    return d

def save_file_npz(args:Args, obs:np.ndarray|list, actions, rewards, next_obs, dones, output_vars):
    """
    Save observations and rewards as .npz file
    """
    assert isinstance(args.save_folder, str), f"Folder args.save_folder `{args.save_folder}` must be of type `str`"
    assert args.save_folder.endswith("/"), f"Folder args.save_folder `{args.save_folder}` must end with `/`"

    assert isinstance(args.data_file, str), f"args.data_file must be of type `str` but is of type `{type(args.data_file)}`"

    np.savez(f"{args.save_folder}{args.data_file}.npz", obs=np.array(obs), next_obs=np.array(next_obs), \
            actions=np.array(actions), rewards=np.array(rewards), dones=np.array(dones),\
            output_vars=np.array(output_vars))
    
def load_data_file(fname):
    """
    Load the data file and get the list of variables for graphing"""
    assert isinstance(fname, str), f"File (args.data_file) `{fname}` is not of type String"
    assert fname.endswith(".npz"), f"File `{fname}` does not end with `.npz`, cannot load."

    data = np.load(fname, allow_pickle=True)

    try: 
        obs = data["obs"]
        actions = data["actions"]
        rewards = data["rewards"]
        next_obs = data["next_obs"]
        dones = data["dones"]
        output_vars = data["output_vars"]
    except:
        msg = f"`{fname}` missing one of the following keys: `obs`, `actions`, `rewards`, `next_obs`, `dones`, `output_vars`. Cannot load data"
        raise Exception(msg)
    
    return obs, actions, rewards, next_obs, dones, output_vars


def get_functions(file):
    """
    Get the functions that correspond only to a specific file
    """
    functions = {name: obj
                for name, obj in getmembers(file, isfunction)
                if getmodule(obj) == file}
    return functions

def get_reward_wrappers(file):
    """
    Get the classes that are declared in a specific file
    """
    classes = {name: obj
                for name, obj in getmembers(file, isclass)
                if getmodule(obj) == file and 
                issubclass(obj, wrappers.RewardWrapper)}
    return classes

def normalize(arr):
    """
    Min-Max normalize array
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-12)

def obs_to_numpy(obs):
    """
    Convert observation to numpy.ndarray based on type
    """
    if isinstance(obs, dict):
        return np.squeeze(np.array(list(obs.values())))
    elif isinstance(obs, torch.Tensor):
        return np.squeeze(obs.cpu().numpy())
    else:
        return np.squeeze(obs)
    
def action_to_numpy(env, act):
    """
    Converts the dicionary action to an integer to be pased to the base
    environment.
    
    Args:
        action
    """
    if isinstance(act, float):
       return np.array([act])
    elif isinstance(act, torch.Tensor):
        return act.cpu().numpy()
    elif isinstance(act, np.ndarray):
        return act
    elif isinstance(act, dict): 
        act_vals = list(act.values())
        for v in act_vals:
            if not isinstance(v, int):
                msg = "Action value must be of type int"
                raise Exception(msg)
        if len(np.nonzero(act_vals)[0]) > 1:
            msg = "More than one non-zero action value for policy"
            raise Exception(msg)
        # If no actions specified, assume that we mean the null action
        if len(np.nonzero(act_vals)[0]) == 0:
            return np.array([0])
    else:
        msg = f"Unsupported Action Type `{type(act)}`. See README for more information"
        raise Exception(msg)
    
    if not "n" in act.keys():
        msg = "Nitrogen action \'n\' not included in action dictionary keys"
        raise Exception(msg)
    if not "p" in act.keys():
        msg = "Phosphorous action \'p\' not included in action dictionary keys"
        raise Exception(msg)
    if not "k" in act.keys():
        msg = "Potassium action \'k\' not included in action dictionary keys"
        raise Exception(msg)
    if not "irrig" in act.keys():
        msg = "Irrigation action \'irrig\' not included in action dictionary keys"
        raise Exception(msg)

    if len(act.keys()) != env.unwrapped.NUM_ACT:
        msg = "Incorrect action dictionary specification"
        raise Exception(msg)
    # Set the offsets to support converting to the correct action
    offsets = [env.unwrapped.num_fert,env.unwrapped.num_fert,env.unwrapped.num_fert,env.unwrapped.num_irrig]
    act_values = [act["n"],act["p"],act["k"],act["irrig"]]
    offset_flags = np.zeros(env.env.unwrapped.NUM_ACT)
    offset_flags[:np.nonzero(act_values)[0][0]] = 1
        
    return np.array([np.sum(offsets*offset_flags) + act_values[np.nonzero(act_values)[0][0]]])

def load_scalars_from_runs(log_dirs:list, scalar_name:str):
    """
    Load scalars from multiple TensorBoard log directories.

    Args:
        log_dirs (list of str): List of TensorBoard log directories.
        scalar_name (str): Name of the scalar to extract.

    Returns:
        dict: A dictionary with log directory names as keys and (steps, values) tuples as values.
    """
    data = {}
    for log_dir in log_dirs:
        # Load TensorBoard events
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        
        # Check if the scalar exists
        if scalar_name not in event_acc.Tags().get('scalars', []):
            continue

        # Retrieve scalar data
        scalar_events = event_acc.Scalars(scalar_name)
        steps = [e.step for e in scalar_events]
        values = [e.value for e in scalar_events]
        data[log_dir] = (steps, values)

    return data