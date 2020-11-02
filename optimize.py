""" RLiable optimization module
"""
import time
import random
import argparse
import uuid

from pyspark import SparkConf, SparkContext

from config.config import APP_NAME, load_conf
from models import models
from utils.misc import log, read_json

def parse_args() -> argparse.Namespace:
    """ Method to parse cmd arguments

    Returns
    -------
    argparse.Namespace
        Arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--optim-config", type=str, required=True,
         help="Optimizatiion configuration path")
    return parser.parse_args()

def create_spark(optim_name:str, exec_workers:int ) -> SparkContext:
    """ Method to create Spark Context

        Returns:
        -----
        sc : pyspark.SparkContext
    """
    conf = SparkConf()\
        .setAppName(APP_NAME+"-"+optim_name)\
        .setMaster(f"local[{exec_workers}]")\
        .set("spark.executor.memory", "4g")\
        .set("spark.executor.cores", f"{exec_workers}")\
        .set("spark.driver.cores",  "1")\
        .set("spark.driver.memory", "2g")
    sc = SparkContext(conf=conf)
    return sc

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
        "mdl_file": cfg['mdl_file'] + '-'+ str(uuid.uuid4()),
        "hp_params": {}
    }
    for k,v in grid.items():
        if isinstance(v, list):
            hypercfg['hp_params'][k] = random.choice(v)
        if isinstance(v, tuple):
            hypercfg['hp_params'][k] = v[0] + random.uniform(0,1) * (v[1] - v[0])
        else:
            raise Exception(f"Invalid grid type: {k}")
    return hypercfg

def run_crossvalidation(sc: SparkContext, training: pyspark.rdd, optim: dict, cfg: dict):
    """ Main method to submit crossvalidation process 

    Parameters
    ----------
    sc : SparkContext
        App context
    training : pyspark.rdd
        Training data or data config
    optim: dict
        Optimization config
    cfg : dict
        Base config
    """
    hcfgs = {}
    for itrs in range(optim['max_iters']):
        _hcfg = sample_random_hyperconfig(optim['grid'], cfg)
        hcfgs[itrs] = _hcfg
        model = models[_hcfg['class']](sc, _hcfg)
        model.train(training)
        model.save()

if __name__ == '__main__':
    log(f"Starting {APP_NAME} optimization ...")
    st_time = time.time()
    # read arguments
    args = parse_args()
    # load config
    cfg = load_conf(args.optim_config['config'])
    log(f"Using {cfg['class']}")
    # create spark
    sc = create_spark(args.optim_config['optim_name'], 
                    args.optim_config['num_workers'])
    # Load environment configuration  
    # - [TODO] : need to initialize environment based on env params
    training = read_json(sc, cfg['environment'])
    # Init model
    model = models[cfg['class']](sc, cfg)
    # Start training
    model.train(training)
    model.save()
    log(f"Finished training in {time.time()- st_time }")