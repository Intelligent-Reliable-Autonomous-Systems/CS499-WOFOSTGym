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
    """Randomization scale for domain randomization"""
    scale: float = 0.1
    """Number of farms for multi farm environment"""
    num_farms = 5

    """Output Variables"""
    output_vars: list = field(default_factory = lambda: ['FIN', 'DVS', 'WSO', 'NAVAIL', 'PAVAIL', 'KAVAIL', 'SM', 'TOTN', 'TOTP', 'TOTK', 'TOTIRRIG'])
    """Weather Variables"""
    weather_vars: list = field(default_factory = lambda: ['IRRAD', 'TEMP', 'RAIN'])

    """Intervention Interval"""
    intvn_interval: int = 1
    """Weather Forecast length in days (min 1)"""
    forecast_length: int = 1
    forecast_noise: list = field(default_factory = lambda: [0, 0.2])
    """Number of NPK Fertilization Actions"""
    """Total number of actions available will be 3*num_fert + num_irrig"""
    num_fert: int = 4
    """Number of Irrgiation Actions"""
    num_irrig: int = 4

    """Flag for resetting to random year"""
    random_reset: bool = False
    """Flag for resetting to a specified group of years"""
    train_reset: bool = False
    """Flag for randomizing a subset of the parameters each reset"""
    domain_rand: bool = False
    """Flag for randomizing a subset of the parameters on initialization - for data generation"""
    crop_rand: bool = False
    
    """Harvest Effiency in range (0,1)"""
    harvest_effec: float = 1.0
    """Irrigation Effiency in range (0,1)"""
    irrig_effec: float = 0.7

    """Coefficient for Nitrogen Recovery after fertilization"""
    n_recovery: float = 0.7
    """Coefficient for Phosphorous Recovery after fertilization"""
    p_recovery: float = 0.7
    """Coefficient for Potassium Recovery after fertilization"""
    k_recovery: float = 0.7
    """Amount of fertilizer coefficient in kg/ha"""
    fert_amount: float = 2
    """Amount of water coefficient in cm/water"""
    irrig_amount: float  = 0.5

    """Path to assets file"""
    assets_fpath: str = f"{os.getcwd()}/pcse_gym/pcse_gym/assets/"

