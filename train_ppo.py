"""
File for training a PPO RL Agent

Written by: Will Solow, 2025

To run: python3 train_agent.py --save-folder <logs>
"""
import tyro
from dataclasses import dataclass, field

from rl_algs.PPO import Args as PPO_Args
from rl_algs import PPO
import utils
from typing import Optional

@dataclass
class AgentArgs(utils.Args):
    # Agent Args configurations
    """Algorithm Parameters for PPO"""
    PPO: PPO_Args = field(default_factory=PPO_Args)

    """Tracking Flag, if True will Track using Weights and Biases"""
    track: bool = False

    """Render mode, default to None for no rendering"""
    render_mode: Optional[str] = None


if __name__ == "__main__":
    
    args = tyro.cli(AgentArgs)

    # Get the agent trainer from RL_Alg
    PPO.train(args)
