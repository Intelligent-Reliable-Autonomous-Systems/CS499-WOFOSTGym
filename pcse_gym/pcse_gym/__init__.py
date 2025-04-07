"""Entry point for pcse_gym package. Handles imports and Gym Environment
registration.
"""

from gymnasium.envs.registration import register

# Default Annual Environments
register(
    id='lnpkw-v0',
    entry_point='pcse_gym.envs.wofost_annual:Limited_NPKW_Env',
)
