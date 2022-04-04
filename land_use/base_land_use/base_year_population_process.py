# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28

@author: adamtebbs
Version number:

Written using: Python 3.7.1

Module versions used for writing:
    pandas v0.25.3
    numpy v1.17.3

Includes functions that were first defined in main_build.py:
    Created on Tue May 26 09:05:40 2020

    @author: mags15
    Version number:

    Written using: Python 3.7.3

    Module versions used for writing:
        pandas v0.24.2

    Main Build:
        - imports addressbase
        - applies household occupancy
        - applies 2011 census segmentation
        - and ns-sec and soc segmentation
        - distinguishes communal establishments

Updates here relative to main_build.py are:
    - Reads in f and P from 2011 Census year outputs
    - Revised processes for uplifting Base Year population based on Base Year MYPE
    - Revised expansion of NTEM population to full dimensions
"""

import pandas as pd
import numpy as np
import os
from ipfn import ipfn
import datetime
import pyodbc

import land_use
# from land_use.utils import file_ops as utils
from land_use.utils import compress
from land_use.utils import general as gen
from land_use import lu_constants as const
from land_use.audits import audits
# from land_use.base_land_use import by_lu


# constants


# Set Model Year
# TODO - ART, 16/02/2022: Make this a variable that is set in run_by_lu
#  Maybe ModelYear should be BaseYear or BaseModelYear too...?
ModelYear = '2019'

# Directory and file paths for the MYPE section

# Directory Paths


# File names

# la_to_msoa_path_og = r'lad_to_msoa.csv'  # Original path, proportions don't quite add to 1



