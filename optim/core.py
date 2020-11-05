import random
import uuid


def sample_random_hyperconfig(grid:dict, cfg:dict) -> dict:
    """ Generate a random hyper configuration based on grid and base config

    Parameters
    ----------
    grid : dict
        Search grid
    cfg : dict
        Basic Configuration

    Returns
    -------
    dict
        Hyper config

    Raises
    ------
    Exception
        Invalid grid settings
    """
    hypercfg = { 
        "class": cfg['class'],
        "environment": cfg['environment'],
        "mdl_file": cfg['mdl_file'] + '-'+ str(uuid.uuid4()),
        "hp_params": {}
    }
    for k, v in grid.items():
        if isinstance(v, list):
            hypercfg['hp_params'][k] = random.choice(v)
        elif isinstance(v, dict):
            hypercfg['hp_params'][k] = v['min'] + random.uniform(0, 1) * (v['max'] - v['min'])
        else:
            raise Exception(f"Invalid grid type: {k}")
    return hypercfg

def has_converged(metric: float, prev_metric:float, convergence:float) -> bool:
    """ Convergence validation

    Parameters
    ----------
    metric : float
        Measured metric
    prev_metric : float
        Measured metric in prev step
    convergence : float
        Convergence threshold

    Returns
    -------
    bool
        True if metric has converged
    """
    return abs(metric - prev_metric) <= convergence