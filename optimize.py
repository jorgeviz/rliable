""" RLiable optimization module
"""
import time
import random
import argparse
import uuid
import json
from pprint import pformat

from pyspark import SparkConf, SparkContext

from models import models
from config.config import APP_NAME, load_con
from utils.misc import log, read_env
from optim.serial import serial_run_crossvalidation

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
    parser.add_argument("--parallelized", action='store_true',
        default=False, help="Parallelization flag")
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

if __name__ == '__main__':
    log(f"Starting {APP_NAME} optimization ...")
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
    st_time = time.time()
    # Load environment configuration  
    training = read_env(sc, cfg['environment'])
    testing = read_env(sc, cfg['environment'])
    # Run CV 
    if args.parallelized:
        raise NotImplementedError
    else:
        serial_run_crossvalidation(sc, training, testing, optconfig, cfg)
    log(f"Finished optimization in {time.time()- st_time }")