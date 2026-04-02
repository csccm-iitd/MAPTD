"""
Utility function to parse the confugaration.yaml file
"""

import os
import re
import torch
from omegaconf import OmegaConf


def parse_cfg(cfg_path: str, task_name: str) -> OmegaConf:
    """Parses a config file and returns an OmegaConf object."""
    base = OmegaConf.load(cfg_path / f'{task_name}.yaml')
    cli = OmegaConf.from_cli()
    
    # Algebraic expressions
    for k,v in base.items():
        if isinstance(v, str):
            match = re.match(r'(\d+)([+\-*/])(\d+)', v)
            if match:
                base[k] = eval(match.group(1) + match.group(2) + match.group(3))
                if isinstance(base[k], float) and base[k].is_integer():
                    base[k] = int(base[k])

    # Convenience
    domain, task = base.task.split('-', 1)
    
    base.domain = domain
    base.task_title = task
    base.device = 'cuda:' + str(base.deviceid) if torch.cuda.is_available() else 'cpu'
    base.exp_name = str(base.get('exp_name', 'default'))

    return base
