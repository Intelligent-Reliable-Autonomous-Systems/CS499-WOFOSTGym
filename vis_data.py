"""
Module for visualizing data

Written by Will Solow, 2024

To run: python3 vis_data.py --data-file <file-name> --plt <plot-type> --fig-folder <save folder>
"""

import tyro, yaml, os
from dataclasses import dataclass
from typing import Optional
import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pcse_gym.utils import ParamLoader

@dataclass
class PlotArgs(utils.Args):
    """Type of plot"""
    plt: Optional[str] = None

    """Where to save figures"""
    fig_folder: Optional[str] = None

    """Path to data file"""
    data_file: Optional[str] = None

    """Variable to plot when visualizing output of policy"""
    policy_var: Optional[str] = "WSO"

    """Figure size"""
    figsize: tuple = tuple((8,6))

def plot_output(args:utils.Args, output_vars:np.ndarray|list=None, obs:np.ndarray=None, 
                actions: np.ndarray=None, rewards:np.ndarray=None, next_obs:np.ndarray=None,
                dones:np.ndarray=None, figsize=(8,6), save=False, **kwargs):
    """
    Plot the output variables for a single simulation. 
    Requires observations to be 2-dimensional of (step, obs)
    """     
    assert isinstance(obs, np.ndarray), "Observations are not of type np.ndarray"   
    assert len(obs.shape) == 2, "Shape of Observations are not 2 dimensional"
    assert obs.shape[-1] == len(output_vars), "Length of Output Variables does not match length of observations"

    ploader = ParamLoader(args.base_fpath, args.name_fpath, args.unit_fpath, args.range_fpath)

    # Set color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=utils.COLORS)

    for i,v in enumerate(output_vars):
        
        # Plot figures
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.plot(obs[:,i], label=f"{v}")
        ax.set_xlabel("Days Elapsed")
        ax.set_ylabel(ploader.get_unit(v))
        ax.set_title(ploader.get_name(v))
        ax.legend()

        if save:
            assert isinstance(args.save_folder, str), f"Folder args.fig_folder `{args.save_folder}` must be of type `str`"
            assert args.save_folder.endswith("/"), f"Folder args.fig_folder `{args.save_folder}` must end with `/`"

            os.makedirs(f"{args.base_fpath}{args.save_folder}", exist_ok=True)
            plt.savefig(f"{args.save_folder}{v}.png", bbox_inches='tight')

        # Save if has attribute
        if hasattr(args, "fig_folder"):
            if args.fig_folder is not None:
                assert isinstance(args.fig_folder, str), f"Folder args.fig_folder `{args.fig_folder}` must be of type `str`"
                assert args.fig_folder.endswith("/"), f"Folder args.fig_folder `{args.fig_folder}` must end with `/`"

                os.makedirs(f"{args.base_fpath}{args.fig_folder}", exist_ok=True)
                plt.savefig(f"{args.fig_folder}{v}.png", bbox_inches='tight')
    plt.show()
    plt.close()

    if rewards is not None:
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.plot(rewards, label="Rewards")
        ax.set_ylabel("Weeks Elapsed")
        ax.set_ylabel("Rewards")
        ax.set_title("Rewards")
        ax.legend()

        if save:
            assert isinstance(args.save_folder, str), f"Folder args.fig_folder `{args.save_folder}` must be of type `str`"
            assert args.save_folder.endswith("/"), f"Folder args.fig_folder `{args.save_folder}` must end with `/`"

            os.makedirs(f"{args.base_fpath}{args.save_folder}", exist_ok=True)
            plt.savefig(f"{args.save_folder}rewards.png", bbox_inches='tight')

        # Save if has attribute
        if hasattr(args, "fig_folder"):
            if args.fig_folder is not None:
                assert isinstance(args.fig_folder, str), f"Folder args.fig_folder `{args.fig_folder}` must be of type `str`"
                assert args.fig_folder.endswith("/"), f"Folder args.fig_folder `{args.fig_folder}` must end with `/`"

                os.makedirs(f"{args.base_fpath}{args.fig_folder}", exist_ok=True)
                plt.savefig(f"{args.fig_folder}/rewards.png", bbox_inches='tight')
    plt.show()
    plt.close()

def plot_policy(args:utils.Args, output_vars:np.ndarray|list=None, obs:np.ndarray=None, 
                actions: np.ndarray=None, rewards:np.ndarray=None, next_obs:np.ndarray=None,
                dones:np.ndarray=None, figsize=(8,6), **kwargs):
    """
    Plot the total growth of the crop and the respective actions taken
    """

    """Assert all necessary params are present"""
    assert "twinax_varname" in kwargs, f"Keyword argument `twinax_varname` not present in `kwargs`"

    twinax_varname = kwargs["twinax_varname"]

    REQUIRED_VARS = ["TOTN", "TOTP", "TOTK", "TOTIRRIG"]+[twinax_varname]
    for v in REQUIRED_VARS:
        assert v in output_vars, f"Required var {v} not in `output_vars`."

    """Create fertilizer/irrigation values"""
    new_totn = obs[:,np.argwhere(output_vars == "TOTN").flatten()[0]]
    new_totp = obs[:,np.argwhere(output_vars == "TOTP").flatten()[0]]
    new_totk = obs[:,np.argwhere(output_vars == "TOTK").flatten()[0]]
    new_totirrig = obs[:,np.argwhere(output_vars == "TOTIRRIG").flatten()[0]]
    
    new_totn[1:] -= new_totn[:-1].copy()
    new_totp[1:] -= new_totp[:-1].copy()
    new_totk[1:] -= new_totk[:-1].copy()
    new_totirrig[1:] -= new_totirrig[:-1].copy()

    """Create indicies for graphing"""
    totn_inds = np.argwhere(new_totn != 0).flatten()
    totn_vals = new_totn[totn_inds]
    totp_inds = np.argwhere(new_totp != 0).flatten()
    totp_vals = new_totp[totp_inds]
    totk_inds = np.argwhere(new_totk != 0).flatten()
    totk_vals = new_totk[totk_inds]
    totirrig_inds = np.argwhere(new_totirrig != 0).flatten()
    totirrig_vals = new_totirrig[totirrig_inds]

    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=utils.COLORS)

    ploader = ParamLoader(args.base_fpath, args.name_fpath, args.unit_fpath, args.range_fpath)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.set_xlabel('Weeks Elapsed')
    ax.set_ylabel('Mineral Applied (kg/ha) and (cm/ha)')
    ax.set_title('Yield vs Mineral Application')

    max_y = np.max([np.max(new_totn), np.max(new_totp), np.max(new_totk), np.max(new_totirrig)])
    ax.set_ylim(0, max_y+10)
    twinax = plt.twinx(ax)
    twinax.set_ylabel(ploader.get_unit(twinax_varname))

    wso = twinax.plot(obs[:,np.argwhere(output_vars==twinax_varname).flatten()[0]], label=twinax_varname)
    
    """Add fertilizer and irrigation patches to plot"""
    n = [[patches.Rectangle((totn_inds[i],0), 1, totn_vals[i], facecolor=('g',.5), edgecolor=('k',.7)) \
          for i in range(len(totn_inds))]]
    [[ax.add_patch(ni) for ni in nj] for nj in n]
    p = [[patches.Rectangle((totp_inds[i],0), 1, totp_vals[i], facecolor=('m',.5), edgecolor=('k',.7)) \
          for i in range(len(totp_inds))] for j in range(len(totn_inds))]
    [[ax.add_patch(pi) for pi in pj] for pj in p]
    k = [[patches.Rectangle((totk_inds[i],0), 1, totk_vals[i], facecolor=('y',.5), edgecolor=('k',.7)) \
          for i in range(len(totk_inds))] for j in range(len(totn_inds))]
    [[ax.add_patch(ki) for ki in kj] for kj in k]
    w = [[patches.Rectangle((totirrig_inds[i],0), 1, totirrig_vals[i], facecolor=('b',.5), edgecolor=('k',.7)) \
          for i in range(len(totirrig_inds))] for j in range(len(totn_inds))]
    [[ax.add_patch(wi) for wi in wj] for wj in w]

    
    n = patches.Patch(color='g', alpha=.6, label='Nitrogen')
    p = patches.Patch(color='m', alpha=.6, label='Phosphorous')
    k = patches.Patch(color='y', alpha=.6, label='Potassium')
    w = patches.Patch(color='b', alpha=.6, label='Water')
    hands = [n,p,k,w, wso[0]]
    plt.legend(handles=hands)

    # Save if has attribute
    if hasattr(args, "fig_folder"):
        if args.fig_folder is not None:
            assert isinstance(args.fig_folder, str), f"Folder args.fig_folder `{args.fig_folder}` must be of type `str`"
            assert args.fig_folder.endswith("/"), f"Folder args.fig_folder `{args.fig_folder}` must end with `/`"

            os.makedirs(f"{args.base_fpath}{args.fig_folder}", exist_ok=True)
            plt.savefig(f"{args.fig_folder}policy_yield.png", bbox_inches='tight')

    plt.show()
    plt.close()

if __name__ == "__main__":
    args = tyro.cli(PlotArgs)

    obs, actions, rewards, next_obs, dones, output_vars = utils.load_data_file(args.data_file)

    try:
        plot_func = utils.get_functions(__import__(__name__))[args.plt]
    except:
        msg = f"Plot Type `{args.plt}` not supported. Ensure that `--plt` is not `None` and the function exists in `vis_data.py`."
        raise Exception(msg)
    
    kwargs = {"twinax_varname":args.policy_var, "figsize":tuple(args.figsize)}
    
    plot_func(args, output_vars, obs, actions, rewards, next_obs, dones, **kwargs)