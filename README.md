# WOFOST-Gym

DISCLAIMER: This repository is still actively under development. Please email 
Will Solow (soloww@oregonstate.edu) with use-case questions and bug reports. 

This is the WOFOSTGym Crop Simulator for the joint [AgAid](https://agaid.org/) Project between Oregon State University (OSU) and Washington State University (WSU). 

See our documentation:

https://intelligent-reliable-autonomous-systems.github.io/WOFOSTGym-Docs/

See our website:

https://intelligent-reliable-autonomous-systems.github.io/WOFOSTGym-Site/

### Citing
```bibtex
@article{solow_wofostgym_2025,
      title={WOFOSTGym: A Crop Simulator for Learning Annual and Perennial Crop Management Strategies}, 
      author={William Solow and Sandhya Saisubramanian and Alan Fern},
      year={2025},
      eprint={2502.19308},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2502.19308}, 
}
```

## Description

Disclaimer: This package has been simplified for use in homework assignments in CS499 Special Topics: Intelligent
Decision Making, taught in Spring 2025 by Dr. Sandhya Saisubramanian.

This package provides the following main features:
1. A crop simulation environment based off of the WOFOST8 crop simulation model
    which simulates the growth of various crops in N/P/K and water limited conditions. 
2. A Gymansium environment which contains the WOFOST8 crop simulation environment.
3. The training of a PPO agent.

Our aim with this project is to support the AgAid community, and other researchers, 
by enabling easy evaluation of decision making systems in the agriculture environment.

## Getting Started

### Dependencies

* This project is entirely self contained and built to run with Python 3.12
* Install using miniconda3 

### Installing

Recommended Installation Method:

1. Navigate to desired installation directory
2. git clone git@github.com:Intelligent-Reliable-Autonomous-Systems/CS499-WOFOSTGym.git
3. conda create -n wofost_env python=3.12
4. conda activate wofost_env
5. pip install -e pcse -e pcse_gym
6. pip install tyro torch omegaconf wandb tensorboard

These commands will install all the required packages into the conda environment
needed to run all scripts in the WOFOSTGym package

## Executing Programs

After following the above installation instructions: 
1. Navigate to the base directory ../CS499-WOFOSTGym/
2. Run the testing domain with: python3 test_sim.py --save-folder test/ --data-file test. This will generate a sample output using default configurations and save a year of data in test/test.npz.
3. This may take a few seconds initially to configure the weather directory

### Use Cases:
* To train an RL Agent: 
    1. `python3 train_agent.py --save-folder <Location>`
    2. Use `--<Agent_Type: PPO.<Agent_Specific_Args>` to specify algorithm specific arguments
    3. To track using Weights and Biases add `--PPO.track`

## Help

Email soloww@oregonstate.edu with any further questions

## Authors

Will Solow (soloww@oregonstate.edu) - Principle Developer

Dr. Sandhya Saisubramanian (sandhya.sai@oregonstate.edu) - Project Lead

## Version History

* 1.0.0
    * Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

The Original PCSE codebase and WOFOST8 Crop Simulator can be found at:
* [PCSE](https://github.com/ajwdewit/pcse)

While we made substantial modifications to the PCSE codebase to suit our needs, 
a large portion of the working code in the PCSE directory is the property of
Dr. Allard de Wit and Wageningen-UR Group. Please see the following paper for an
overview of WOFOST:
* [WOFOST](https://www-sciencedirect-com.oregonstate.idm.oclc.org/science/article/pii/S0308521X17310107)

The original inspiration for a crop simulator gym environment came from the paper:
* [CropGym](https://arxiv.org/pdf/2104.04326)

We have since extended their work to interface with multiple Reinforcement Learning Agents, 
have added support for perennial fruit tree crops, grapes, multi-year simulations, and different sowing
and harvesting actions. 

The Python Crop Simulation Environment (PCSE) is well documented. Resources can 
be found here:
* [PCSE Docs](https://pcse.readthedocs.io/en/stable/)

The WOFOST crop simulator is also well documented, and we use the WOFOST8 model
in our crop simulator. Documentation can be found here:
* [WOFOST Docs](https://wofost.readthedocs.io/en/latest/)