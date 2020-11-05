from pprint import pformat
import math

from pyspark import SparkContext

from optim.core import sample_random_hyperconfig, has_converged
from models import models
from utils.misc import log, read_env

def train_eval_mapper(optim_row: tuple) -> tuple:
    """ Train-eval mapper function, 
        it runs together training and evaluation process 
        within same worker

    Parameters
    ----------
    optim_row : tuple
        (iter number, hyperconfig dict)

    Returns
    -------
    tuple
        (iter number, hyperconfig dict with eval metric)
    """
    itrs, _hcfg = optim_row
    training_env = read_env('sc', _hcfg['environment'])
    testing_env = read_env('sc', _hcfg['environment'])
    # instance and train model
    model = models[_hcfg['class']]('sc', _hcfg)
    model.train(training_env, testing_env)
    model.save()
    # run evaluation in testing env
    _preds, metric = model.evaluate(testing_env)
    _hcfg['metric'] = metric
    return (itrs, _hcfg)

def train_mapper(optim_row: tuple) -> tuple:
    """ Train mapper function, 
        it runs only training in one worker

    Parameters
    ----------
    optim_row : tuple
        (iter number, hyperconfig dict)

    Returns
    -------
    tuple
        (iter number, hyperconfig dict)
    """
    itrs, _hcfg = optim_row
    training_env = read_env('sc', _hcfg['environment'])
    # instance and train model
    model = models[_hcfg['class']]('sc', _hcfg)
    model.train(training_env, training_env)
    model.save()
    return (itrs, _hcfg)

def eval_mapper(optim_row: tuple) -> tuple:
    """ Eval mapper function, 
        it runs only testing in one worker

    Parameters
    ----------
    optim_row : tuple
        (iter number, hyperconfig dict, num_evals)

    Returns
    -------
    tuple
        (iter number, hyperconfig dict with eval metric)
    """
    itrs, _hcfg, n_eval_eps = optim_row
    testing_env = read_env('sc', _hcfg['environment'])
    # instance, load model and eval model
    model = models[_hcfg['class']]('sc', _hcfg)
    model.load_model()
    _preds, metric = model.evaluate(testing_env, n_eval_eps)
    _hcfg['metric'] = metric
    return (itrs, _hcfg)

def parallel_run_crossvalidation(sc: SparkContext, 
                    training,
                    testing,
                    optim: dict, cfg: dict):
    """ Parallel MapReduce implementation of crossvalidation process.
        The jobs are splitted in batches depending on the amount of resources
        available, it performs training and evaluation process in one worker.
    Parameters
    ----------
    sc : SparkContext
        App context
    training : pyspark.rdd.RDD | tf_agents.environments.TFPyEnvironment
        Training data or data config
    Testing : pyspark.rdd.RDD | tf_agents.environments.TFPyEnvironment
        Eval/Testing data or data config
    optim: dict
        Optimization config
    cfg : dict
        Base config
    """
    global training_env, testing_env
    if (optim['num_workers'] < 2):
        raise Exception("MapReduce optimization needs at least 2 workers!")
    hcfgs = {}
    metric_series = []
    for itrs in range(math.ceil(optim['max_iters'] / optim['num_workers'])):
        log(f"Running CV-{itrs} batch ({itrs * optim['num_workers']} - {(itrs+1) * optim['num_workers']})")
        # generate Iters / Num_Workers hyperconfigs
        mpr_hcfgs = sc.parallelize([
            (_j, sample_random_hyperconfig(optim['grid'], cfg))
            for _j in range(itrs * optim['num_workers'], (itrs+1) * optim['num_workers'])
        ])
        hcfgs.update(
            mpr_hcfgs.map(train_eval_mapper).collectAsMap()
        )
        metric_series = [(_k, _h['metric']) for _k, _h in hcfgs.items()]
        # convergence validation
        if itrs > 1:
            if has_converged(metric_series[-2][1], metric_series[-1][1], optim['convergence']):
                log(f"Optimization has converged in {itrs} batch iterations")
                break
    # best model selection based metric
    best_model = hcfgs[
        sorted(
            metric_series, key=lambda s: s[1], 
            reverse=(optim['metric']['criteria'] == 'max')
        )[0][0]
    ]
    log("Best performed model:\n", pformat(best_model))


def parallel_run_crossvalidation_v2(sc: SparkContext, 
                    training,
                    testing,
                    optim: dict, cfg: dict):
    """ Parallel MapReduce implementation of crossvalidation process 
        he jobs are splitted in batches depending on the amount of resources
        available. It performs training in one process, and it extends another 
        level of parallelization performing a split of the evaluation rollouts 
        to all the workers across the cluster.

    Parameters
    ----------
    sc : SparkContext
        App context
    training : pyspark.rdd.RDD | tf_agents.environments.TFPyEnvironment
        Training data or data config
    Testing : pyspark.rdd.RDD | tf_agents.environments.TFPyEnvironment
        Eval/Testing data or data config
    optim: dict
        Optimization config
    cfg : dict
        Base config
    """
    global training_env, testing_env
    if (optim['num_workers'] < 2):
        raise Exception("MapReduce optimization needs at least 2 workers!")
    hcfgs = {}
    metric_series = []
    for itrs in range(math.ceil(optim['max_iters'] / optim['num_workers'])):
        log(f"Running CV-{itrs} batch ({itrs * optim['num_workers']} - {(itrs+1) * optim['num_workers']})")
        # generate Iters / Num_Workers hyperconfigs
        mpr_hcfgs = sc.parallelize([
            (_j, sample_random_hyperconfig(optim['grid'], cfg))
            for _j in range(itrs * optim['num_workers'], (itrs+1) * optim['num_workers'])
        ])
        _eval_episodes = min(optim['grid'].get("num_eval_episodes", [16]))
        hcfgs.update(
            mpr_hcfgs.map(train_mapper)\
                    .flatMap(lambda q: [
                        (q[0], q[1], _eval_episodes//optim['num_workers'])
                        for __w in range(optim['num_workers'])
                    ])\
                    .map(eval_mapper)\
                    .collectAsMap()
        )
        metric_series = [(_k, _h['metric']) for _k, _h in hcfgs.items()]
        # convergence validation
        if itrs > 1:
            if has_converged(metric_series[-2][1], metric_series[-1][1], optim['convergence']):
                log(f"Optimization has converged in {itrs} batch iterations")
                break
    # best model selection based metric
    best_model = hcfgs[
        sorted(
            metric_series, key=lambda s: s[1], 
            reverse=(optim['metric']['criteria'] == 'max')
        )[0][0]
    ]
    log("Best performed model:\n", pformat(best_model))