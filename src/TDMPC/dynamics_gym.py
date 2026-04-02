# -*- coding: utf-8 -*-
"""
Init function for the environments
"""

from env_beam import EulerBeam
from env_tallstorey import Skyscraper
from env_tallstorey import Skyscraper_rom

env = {'EulerBeam' : EulerBeam,
       'Skyscraper' : Skyscraper,
       'Skyscraper_rom': Skyscraper_rom}

