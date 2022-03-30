"""
Created on: Fri March 25 2022
Updated on:

Original author: Nirmal Kumar
Last update made by:
Other updates made by:

File purpose:
Module of errors for the pathing modules
"""
# Builtins

# Third Party

# Local imports
import land_use as lu


class PathingError(lu.LandUseError):
    """
    Base Exception for all NorMITs LandUse Pathing errors
    """

    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)
