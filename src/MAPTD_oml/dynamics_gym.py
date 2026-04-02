# -*- coding: utf-8 -*-
"""
Init function for the environments
"""

from env_beam import EulerBeam
from env_beam_nn import EulerBeam as EulerBeamNN
from env_tallstorey import Skyscraper
from env_tallstorey import Skyscraper_rom
from env_tallstorey_nn import Skyscraper_rom as Skyscraper_romNN

env = {'EulerBeam' : EulerBeam,
       'Skyscraper' : Skyscraper,
       'Skyscraper_rom': Skyscraper_rom,
       'EulerBeamNN': EulerBeamNN,
       'Skyscraper_romNN': Skyscraper_romNN}
