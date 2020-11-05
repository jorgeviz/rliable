""" RLiable single evaluation module
"""
import time
from pyspark import SparkConf, SparkContext
from models import models
from config.config import APP_NAME, load_conf
from utils.misc import parse_predit_args, log, read_env

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
    log(f"Starting {APP_NAME} evaluation ...")
    args = parse_predit_args()
    # load config
    cfg = load_conf()
    log(f"Using {cfg['class']}")
    # create spark
    sc = create_spark()
    st_time = time.time()
    # Load testing data
    testing = read_env(sc, args['test_file'])
    # Init model
    model = models[cfg['class']](sc, cfg)
    # Load model  and eval
    model.load_model()
    model.evaluate(testing)
    log(f"Finished predicting in {time.time() - st_time}")