"""Args configurations file includes: 
    - PCSE configuration file for WOFOST 8.0 Water and NPK limited Production
    - WOFOST Gym parameter configurations
"""

from dataclasses import dataclass, field
import os

@dataclass
class NPK_Args:
    """
    Arguments for the WOFOST Gym environment
    """

    """Environment seed"""
    seed: int = 0

    """Output Variables"""
    output_vars: list = field(default_factory = lambda: ['FIN', 'DVS', 'WSO', 'WLV', 'WRT', 'WST', 'NAVAIL', 'PAVAIL', 'KAVAIL', 'SM', 'TOTN', 'TOTP', 'TOTK', 'TOTIRRIG', 'RRUNOFF_N', 'RRUNOFF_P', 'RRUNOFF_K'])
    """Weather Variables"""
    weather_vars: list = field(default_factory = lambda: ['IRRAD', 'TEMP', 'RAIN'])

    """Flag for resetting to random year"""
    random_reset: bool = False
    
    """Path to assets file"""
    assets_fpath: str = f"{os.getcwd()}/pcse_gym/pcse_gym/assets/"

