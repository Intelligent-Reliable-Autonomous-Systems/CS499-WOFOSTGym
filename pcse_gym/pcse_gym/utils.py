
"""Utils file for making model configurations and setting parameters from arguments
"""
import yaml
import numpy as np

from pcse.soil.soil_wrappers import BaseSoilModuleWrapper, SoilModuleWrapper_LNPKW
from pcse.crop.wofost8 import BaseCropModel, Wofost80
from pcse.agromanager import BaseAgroManager, AgroManagerAnnual

def make_config(soil: BaseSoilModuleWrapper=SoilModuleWrapper_LNPKW, crop: BaseCropModel=Wofost80, \
                agro: BaseAgroManager=AgroManagerAnnual):
    """Makes the configuration dictionary to be used to set various values of
    the model.
    
    Further modified in the WOFOST Gym delcaration.
    
    Args:
        None
    """

    # Module to be used for water balance
    SOIL = soil

    # Module to be used for the crop simulation itself
    CROP = crop

    # Module to use for AgroManagement actions
    AGROMANAGEMENT = agro

    # interval for OUTPUT signals, either "daily"|"dekadal"|"monthly"
    # For daily output you change the number of days between successive
    # outputs using OUTPUT_INTERVAL_DAYS. For dekadal and monthly
    # output this is ignored.
    OUTPUT_INTERVAL = "daily"
    OUTPUT_INTERVAL_DAYS = 1
    # Weekday: Monday is 0 and Sunday is 6
    OUTPUT_WEEKDAY = 0

    # variables to save at OUTPUT signals
    # Set to an empty list if you do not want any OUTPUT
    OUTPUT_VARS = [ 
        # WOFOST STATES 
        "TAGP", "GASST", "MREST", "CTRAT", "CEVST", "HI", "DOF", "FINISH_TYPE", "FIN",
        # WOFOST RATES 
        "GASS", "PGASS", "MRES", "ASRC", "DMI", "ADMI",
        # EVAPOTRANSPIRATION STATES
        "IDOST", "IDWST",
        # EVAPOTRANSPIRATION RATES  
        "EVWMX", "EVSMX", "TRAMX", "TRA", "IDOS", "IDWS", "RFWS", "RFOS", 
        "RFTRA",
        # LEAF DYNAMICS STATES
        "LV", "SLA", "LVAGE", "LAIEM", "LASUM", "LAIEXP", "LAIMAX", "LAI", "WLV", 
        "DWLV", "TWLV",
        # LEAF DYNAMICS RATES
        "GRLV", "DSLV1", "DSLV2", "DSLV3", "DSLV4", "DSLV", "DALV", "DRLV", "SLAT", 
        "FYSAGE", "GLAIEX", "GLASOL",
        # NPK DYNAMICS STATES
        "NAMOUNTLV", "PAMOUNTLV", "KAMOUNTLV", "NAMOUNTST", "PAMOUNTST", "KAMOUNTST",
        "NAMOUNTSO", "PAMOUNTSO", "KAMOUNTSO", "NAMOUNTRT", "PAMOUNTRT", "KAMOUNTRT",
        "NUPTAKETOTAL", "PUPTAKETOTAL", "KUPTAKETOTAL", "NFIXTOTAL", "NLOSSESTOTAL", 
        "PLOSSESTOTAL", "KLOSSESTOTAL", 
        # NPK DYNAMICS RATES
        "RNAMOUNTLV", "RPAMOUNTLV", "RKAMOUNTLV", 
        "RNAMOUNTST", "RPAMOUNTST", "RKAMOUNTST", "RNAMOUNTRT", "RPAMOUNTRT",  
        "RKAMOUNTRT", "RNAMOUNTSO", "RPAMOUNTSO", "RKAMOUNTSO", "RNDEATHLV", 
        "RNDEATHST", "RNDEATHRT", "RPDEATHLV", "RPDEATHST", "RPDEATHRT", "RKDEATHLV",
        "RKDEATHST", "RKDEATHRT", "RNLOSS", "RPLOSS", "RKLOSS", 
        # PARTIONING STATES
        "FR", "FL", "FS", "FO", 
        # PARTIONING RATES
            # NONE
        # VERNALIZATION STATES
        "VERN", "ISVERNALISED",
        # VERNALIZATION RATES
        "VERNR", "VERNFAC",   
        # PHENOLOGY STATES
        "DVS", "TSUM", "TSUME", "STAGE", "DSNG",
        "DSD", "AGE", "DOP", "DATBE", "DOC", "DON", "DOB", "DOV", "DOR", "DOL",
        # PHENOLOGY RATES
        "DTSUME", "DTSUM", "DVR", "RDEM",
        # RESPIRATION STATES
            # NONE
        # RESPIRATION RATES
        "PMRES",
        # ROOT DYNAMICS STATES
        "RD", "RDM", "WRT", "DWRT", "TWRT", 
        # ROOT DYNAMICS RATES
        "RR", "GRRT", "DRRT1", "DRRT2", "DRRT3", "DRRT", "GWRT", 
        # STEM DYNAMICS STATES
        "WST", "DWST", "TWST", "SAI", 
        # STEM DYNAMICS RATES
        "GRST", "DRST", "GWST",
        # STORAGE ORGAN DYNAMICS STATES
        "WSO", "DWSO", "TWSO", "HWSO", "PAI","LHW",
        # STORAGE ORGAN DYNAMICS RATES
        "GRSO", "DRSO", "GWSO", "DHSO",
        # NPK NUTRIENT DEMAND UPTAKE STATES
            # NONE
        # NPK NUTRIENT DEMAND UPTAKE RATES
        "RNUPTAKELV", "RNUPTAKEST", "RNUPTAKERT", "RNUPTAKESO", "RPUPTAKELV", 
        "RPUPTAKEST", "RPUPTAKERT", "RPUPTAKESO", "RKUPTAKELV", "RKUPTAKEST", 
        "RKUPTAKERT", "RKUPTAKESO", "RNUPTAKE", "RPUPTAKE", "RKUPTAKE", "RNFIXATION",
        "NDEMANDLV", "NDEMANDST", "NDEMANDRT", "NDEMANDSO", "PDEMANDLV", "PDEMANDST", 
        "PDEMANDRT", "PDEMANDSO", "KDEMANDLV", "KDEMANDST", "KDEMANDRT","KDEMANDSO", 
        "NDEMAND", "PDEMAND", "KDEMAND", 
        # NPK STRESS STATES
            # NONE
        # NPK STRESS RATES
        "NNI", "PNI", "KNI", "NPKI", "RFNPK", 
        # NPK TRANSLOCATION STATES
        "NTRANSLOCATABLELV", "NTRANSLOCATABLEST", "NTRANSLOCATABLERT", "PTRANSLOCATABLELV",
        "PTRANSLOCATABLEST", "PTRANSLOCATABLERT", "KTRANSLOCATABLELV", "KTRANSLOCATABLEST",
        "KTRANSLOCATABLERT", "NTRANSLOCATABLE", "PTRANSLOCATABLE", "KTRANSLOCATABLE", 
        # NPK TRANSLOCATION RATES
        "RNTRANSLOCATIONLV", "RNTRANSLOCATIONST", "RNTRANSLOCATIONRT", "RPTRANSLOCATIONLV",
        "RPTRANSLOCATIONST", "RPTRANSLOCATIONRT", "RKTRANSLOCATIONLV", "RKTRANSLOCATIONST",
        "RKTRANSLOCATIONRT",
        # SOIL STATES
        "SM", "SS", "SSI", "WC", "WI", "WLOW", "WLOWI", "WWLOW", "WTRAT", "EVST", 
        "EVWT", "TSR", "RAINT", "WART", "TOTINF", "TOTIRR", "PERCT", "LOSST", "WBALRT", 
        "WBALTT", "DSOS", "TOTIRRIG",
        # SOIL RATES
        "EVS", "EVW", "WTRA", "RIN", "RIRR", "PERC", "LOSS", "DW", "DWLOW", "DTSR", 
        "DSS", "DRAINT", 
        # NPK SOIL DYNAMICS STATES
        "NSOIL", "PSOIL", "KSOIL", "NAVAIL", "PAVAIL", "KAVAIL", "TOTN", "TOTP", "TOTK",
        "SURFACE_N", "SURFACE_P", "SURFACE_K", "TOTN_RUNOFF", "TOTP_RUNOFF", "TOTK_RUNOFF",
        # NPK SOIL DYNAMICS RATES
        "RNSOIL", "RPSOIL", "RKSOIL", "RNAVAIL", "RPAVAIL", "RKAVAIL", "FERT_N_SUPPLY",
        "FERT_P_SUPPLY", "FERT_K_SUPPLY", "RNSUBSOIL", "RPSUBSOIL", "RKSUBSOIL",
        "RRUNOFF_N", "RRUNOFF_P", "RRUNOFF_K",
        ]

    # Summary variables to save at CROP_FINISH signals
    # Set to an empty list if you do not want any SUMMARY_OUTPUT
    SUMMARY_OUTPUT_VARS = OUTPUT_VARS

    # Summary variables to save at TERMINATE signals
    # Set to an empty list if you do not want any TERMINAL_OUTPUT
    TERMINAL_OUTPUT_VARS = OUTPUT_VARS

    return {'SOIL': SOIL, 'CROP': CROP, 'AGROMANAGEMENT': AGROMANAGEMENT, 'OUTPUT_INTERVAL': OUTPUT_INTERVAL, \
            'OUTPUT_INTERVAL_DAYS':OUTPUT_INTERVAL_DAYS, 'OUTPUT_WEEKDAY': OUTPUT_WEEKDAY, \
                'OUTPUT_VARS': OUTPUT_VARS, 'SUMMARY_OUTPUT_VARS': SUMMARY_OUTPUT_VARS, \
                    'TERMINAL_OUTPUT_VARS': TERMINAL_OUTPUT_VARS}

class ParamLoader():
    """
    Class to handle loading the parameter and state/rate names of all variables
    """
    def __init__(self, base_fpath:str=None, name_fpath:str=None, unit_fpath:str=None, range_fpath:str=None):
        """
        Initialize State by loading respective .yaml files as specified in utils.Args
        """
        try: 
            with open(f"{base_fpath}{name_fpath}", "rb") as f:
                self.state_names = yaml.safe_load(f) 
        except:
            msg = f"Error loading State Names file `{base_fpath}{name_fpath}`. Check correct path or that file exists."
            raise Exception(msg)
        
        try: 
            with open(f"{base_fpath}{unit_fpath}", "rb") as f:
                self.state_units = yaml.safe_load(f) 
        except:
            msg = f"Error loading State Names file `{base_fpath}{unit_fpath}`. Check correct path or that file exists."
            raise Exception(msg)
        
        try: 
            with open(f"{base_fpath}{range_fpath}", "rb") as f:
                self.state_range = yaml.safe_load(f) 
        except:
            msg = f"Error loading State Ranges file `{base_fpath}{range_fpath}`. Check correct path or that file exists."
            raise Exception(msg)
        
    def get_name(self, key:str):
        """
        Get the name of a specific key
        """
        try:
            return self.state_names[key]
        except:
            msg = f"`{key}` not found in State Names YAML file. Check that key exists in file."
            raise Exception(msg)
        
    def get_unit(self, key:str):
        """
        Get the unit of a specific key
        """
        try:
            return self.state_units[key]
        except:
            msg = f"`{key}` not found in State Units YAML file. Check that key exists in file."
            raise Exception(msg)
        
    def get_range(self, key:str):
        """
        Get the range of a specific key
        """
        try: 
            val = self.state_range[key]
            return np.array(list(val), dtype=np.float64)
        except:
            msg = f"`{key}` not found in State Range YAML file. Check that key exists in file."
            raise Exception(msg)
        
    def normalize(self, key:str, obs:float):
        """
        Get the range of a specific key and normalize
        """
        try: 
            range = np.array(list(self.state_range[key]), dtype=np.float64)
            return (obs - range[0]) / (range[1]-range[0] + 1e-12)
        except:
            msg = f"`{key}` not found in State Range YAML file. Check that key exists in file."
            raise Exception(msg)

