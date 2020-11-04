""" RLiable optimization module
"""
import time
import random
import argparse
import uuid
import json
from pprint import pformat

from pyspark import SparkConf, SparkContext
from pyspark.rdd import RDD

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
    for k, v in grid.items():
        if isinstance(v, list):
            hypercfg['hp_params'][k] = random.choice(v)
        elif isinstance(v, tuple):
            hypercfg['hp_params'][k] = v[0] + random.uniform(0, 1) * (v[1] - v[0])
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

def run_crossvalidation(sc: SparkContext, 
                    training: RDD,
                    testing: RDD,
                    optim: dict, cfg: dict):
    """ Main method to submit crossvalidation process 

    Parameters
    ----------
    sc : SparkContext
        App context
    training : pyspark.rdd.RDD
        Training data or data config
    optim: dict
        Optimization config
    cfg : dict
        Base config
    """
    hcfgs = {}
    metric_series = []
    for itrs in range(optim['max_iters']):
        log(f"Running CV-{itrs}")
        _hcfg = sample_random_hyperconfig(optim['grid'], cfg)
        hcfgs[itrs] = _hcfg
        # instance and train model
        model = models[_hcfg['class']](sc, _hcfg)
        model.train(training)
        model.save()
        # run evaluation in testing env
        _preds, metric = model.evaluate(testing)
        hcfgs[itrs]['metric'] = metric
        # convergence validation
        if itrs > 1:
            if has_converged(metric, metric_series[-1][1], optim['convergence']):
                log(f"Optimization has converged in {itrs} iterations")
                break
        metric_series.append((itrs, metric))
    # best model selection based metric
    best_model = hcfgs[
        sorted(
            metric_series, key=lambda s: s[1], 
            reverse=(optim['metric']['criteria'] == 'max')
        )[0][0]
    ]
    log("Best performed model:\n", pformat(best_model))

if __name__ == '__main__':
    log(f"Starting {APP_NAME} optimization ...")
    st_time = time.time()
    # read arguments
    args = parse_args()
    with open(args.optim_config, 'r') as f:
        optconfig = json.load(f)
    # load config
    cfg = load_conf(optconfig['config'])
    log(f"Using {cfg['class']}")
    # create spark
    sc = create_spark(optconfig['optim_name'], 
                    optconfig['num_workers'])
    # Load environment configuration  
    # - [TODO] : need to initialize environment based on env params
    training = read_json(sc, cfg['environment'])
    testing = read_json(sc, cfg['environment'])
    # Run CV 
    run_crossvalidation(sc, training, testing, optconfig, cfg)
    log(f"Finished optimization in {time.time()- st_time }")