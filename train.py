""" RLiable single training module
"""
import time

from pyspark import SparkConf, SparkContext

from config.config import APP_NAME, load_conf
from models import models
from utils.misc import log, read_env

def create_spark():
    """ Method to create Spark Context

        Returns:
        -----
        sc : pyspark.SparkContext
    """
    conf = SparkConf()\
        .setAppName(APP_NAME)\
        .setMaster("local[4]")\
        .set("spark.executor.memory", "4g")\
        .set("spark.executor.cores", "4")\
        .set("spark.driver.cores",  "2")\
        .set("spark.driver.memory", "2g")
    sc = SparkContext(conf=conf)
    return sc

if __name__ == '__main__':
    log(f"Starting {APP_NAME} training ...")
    st_time = time.time()
    # load config
    cfg = load_conf()
    log(f"Using {cfg['class']}")
    # create spark
    sc = create_spark()
    # Load environment configuration  
    training = read_env(sc, cfg['environment'])
    evaluation = read_env(sc, cfg['environment'])
    # Init model
    model = models[cfg['class']](sc, cfg)
    # Start training
    model.train(training, evaluation)
    model.save()
    log(f"Finished training in {time.time()- st_time }")