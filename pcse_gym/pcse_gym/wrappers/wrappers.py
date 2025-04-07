"""Core API for environment wrappers for handcrafted policies and varying rewards."""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Box
from abc import abstractmethod, ABC
import torch

from pcse_gym import exceptions as exc
          
class RewardWrapper(gym.Wrapper, ABC):
    """ Abstract class for all reward wrappers
    
    Given how the reward wrapper functions, it must be applied BEFORE any
    observation or action wrappers. 
    
    This _validate() function ensures that is the case and will throw and error
    otherwise 
    """
    def __init__(self, env: gym.Env):
        """Initialize the :class:`RewardWrapper` wrapper with an environment.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self._validate(env)
        self.env = env

    @abstractmethod
    def _get_reward(self, output:dict, act_tuple):
        """
        The get reward function shaping the reward. Implement this.
        """
        pass

    def _validate(self, env: gym.Env):
        """Validates that the environment is not wrapped with an Observation or 
        Action Wrapper
        
        Args: 
            env: The environment to check
        """
        if isinstance(env, gym.ActionWrapper) or isinstance(env, gym.ObservationWrapper):
            msg = f"Cannot wrap a `{type(self)}` around `{type(env)}`. Wrap Env with `{type(self)}` before wrapping with `{type(env)}`."
            raise exc.WOFOSTGymError(msg)
        if isinstance(env, RewardWrapper):
            msg = "Cannot wrap environment with another reward wrapper."
            raise exc.WOFOSTGymError(msg)
        
    def step(self, action:int):
        """Run one timestep of the environment's dynamics.

        Sends action to the WOFOST model and recieves the resulting observation
        which is then processed to the _get_reward() function and _process_output()
        function for a reward and observation

        Args:
            action: integer
        """
        if isinstance(action, dict):
            msg = f"Action must be of type `int` but is of type `dict`. Wrap environment in `pcse_gym.wrappers.NPKDictActionWrapper` before proceeding."
            raise Exception(msg)
        # Send action signal to model and run model
        act_tuple = self.env.unwrapped._take_action(action)
  
        output = self.env.unwrapped._run_simulation()

        observation = self.env.unwrapped._process_output(output)
        
        reward = self._get_reward(output, act_tuple) 

        # Terminate based on crop finishing
        termination = output[-1]['FIN'] == 1.0 or output[-1]['FIN'] is None
        if output[-1]['FIN'] is None:
            observation = np.nan_to_num(observation)

        # Truncate based on site end date
        truncation = self.env.unwrapped.date >= self.env.unwrapped.site_end_date

        self.env.unwrapped._log(output[-1]['WSO'], act_tuple, reward)

        return observation, reward, termination, truncation, self.env.unwrapped.log
        
    def reset(self, **kwargs):
        """
        Forward keyword environments to base env
        """
        return self.env.reset(**kwargs)

class RewardFertilizationCostWrapper(RewardWrapper):
    """ Modifies the reward to be a function of how much fertilization and irrigation
    is applied
    """
    def __init__(self, env: gym.Env, args):
        """Initialize the :class:`RewardFertilizationCostWrapper` wrapper with an environment.

        Args: 
        """

        super().__init__(env)
        self.env = env

        self.cost = 10

    def _get_reward(self, output: dict, act_tuple:tuple):
        """Gets the reward as a penalty based on the amount of NPK/Water applied
        
        Args:
            output: dict     - output from model
            act_tuple: tuple -  NPK/Water amounts"""
        act_tuple = tuple(float(x) for x in act_tuple)


        reward = output[-1]['WSO'] - \
                    (np.sum(self.cost * np.array([act_tuple[:-1]])))  if output[-1]['WSO'] \
                    is not None else -np.sum(self.cost * np.array([act_tuple[2:]]))
        return reward
         
class RewardFertilizationThresholdWrapper(RewardWrapper):
    """ Modifies the reward to be a function with high penalties for if a 
     threshold is crossed during fertilization or irrigation
    """
    def __init__(self, env: gym.Env, args):
        """Initialize the :class:`RewardFertilizationThresholdWrapper` wrapper with an environment.

        Args: 
        """
        super().__init__(env)
        self.env = env

        # Thresholds for nutrient application
        self.max_n = 20
        self.max_p = 20
        self.max_k = 20
        self.max_w = 20

        # Set the reward range in case of normalization
        self.reward_range = [4*-1e4, 10000]

    def _get_reward(self, output, act_tuple):
        """Convert the reward by applying a high penalty if a fertilization
        threshold is crossed
        
        Args:
            output     - of the simulator
            act_tuple  - amount of NPK/Water applied
        """
        if output[-1]['TOTN'] > self.max_n and act_tuple[self.env.unwrapped.N] > 0:
            return -1e4 * act_tuple[self.env.unwrapped.N]
        if output[-1]['TOTP'] > self.max_p and act_tuple[self.env.unwrapped.P] > 0:
            return -1e4 * act_tuple[self.env.unwrapped.P]
        if output[-1]['TOTK'] > self.max_k and act_tuple[self.env.unwrapped.K] > 0:
            return -1e4 * act_tuple[self.env.unwrapped.K]
        if output[-1]['TOTIRRIG'] > self.max_w and act_tuple[self.env.unwrapped.I] > 0:
            return -1e4 * act_tuple[self.env.unwrapped.I]

        return output[-1]['WSO'] if output[-1]['WSO'] is not None else 0
    
class RewardLimitedRunoffWrapper(RewardWrapper):
    """ Modifies the reward to be a function with high penalties for if Nitrogen Runoff Occurs
    """
    def __init__(self, env: gym.Env, args):
        """Initialize the :class:`RewardFertilizationThresholdWrapper` wrapper with an environment.

        Args: 
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.env = env

        # Thresholds for nutrient application

        # Set the reward range in case of normalization
        self.reward_range = [4*-1e5, 10000]

    def _get_reward(self, output, act_tuple):
        """Convert the reward by applying a high penalty if a fertilization
        threshold is crossed
        
        Args:
            output     - of the simulator
            act_tuple  - amount of NPK/Water applied
        """
        if output[-1]['RRUNOFF_N'] > 0:
            return -1e5 * output[-1]['RRUNOFF_N']
        return output[-1]['WSO'] if output[-1]['WSO'] is not None else 0

class NormalizeObservation(gym.Wrapper):

    def __init__(self, env:gym.Env):
        """
        Initialize normalization wrapper
        """
        super().__init__(env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        self.env = env
        self.output_vars = self.env.unwrapped.output_vars
        self.weather_vars = self.env.unwrapped.weather_vars

        self.all_vars = self.output_vars + self.weather_vars + ["DAYS"]

        self.ploader = self.env.unwrapped.ploader

        self.ranges = np.stack([self.ploader.get_range(k) for k in self.all_vars], dtype=np.float64)

        if hasattr(env, "reward_range"):
            self.reward_range = env.reward_range
        else:
            self.reward_range = [0,10000]

    def normalize(self, obs):
        """
        Normalize the observation
        """

        obs = (obs - self.ranges[:,0]) / (self.ranges[:,1] - self.ranges[:,0] +1e-12)

        return obs
    
    def unnormalize(self, obs):
        """
        Normalize the observation
        """
        obs = obs * (self.ranges[:,1] - self.reward_range[:,0] + 1e-12) + self.ranges[:,0]

        return obs
    
    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)

        if self.is_vector_env:
            return self.normalize(obs), info
        else:
            return self.normalize(np.array([obs]))[0], info

class NormalizeReward(gym.Wrapper):

    def __init__(self, env:gym.Env):
        """
        Initialize normalization wrapper for rwards
        """
        super().__init__(env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        if hasattr(env, "reward_range"):
            self.reward_range = env.reward_range
            if self.reward_range == (float('-inf'), float('inf')):
                self.reward_range = [0,10000]
        else:
            self.reward_range = [0,10000]

        if hasattr(env, "ranges"):
            self.ranges = env.ranges
        
    def unnormalize_obs(self, obs):
        """
        Normalize the observation
        """
        obs = obs * (self.ranges[:,1] - self.ranges[:,0] + 1e-12) + self.ranges[:,0]

        return obs

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if isinstance(rews, torch.Tensor):
            rews = rews.cpu()
        if self.is_vector_env:
            rews = self.normalize(rews)
        else:
            rews = self.normalize(np.array([rews]))[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)

        return obs, info
    
    def normalize(self, rews):
        """
        Normalize the observation
        """
        rews = (rews - self.reward_range[0]) / (self.reward_range[1] - self.reward_range[0] +1e-12)

        return rews

    def unnormalize(self, rews):
        """
        Unnormalize the reward
        """
        rews = rews * (self.reward_range[1] - self.reward_range[0] + 1e-12) + self.reward_range[0]

        return rews


