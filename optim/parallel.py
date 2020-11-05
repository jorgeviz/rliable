from pprint import pformat
from pyspark import SparkContext

from optim.core import sample_random_hyperconfig, has_converged
from models import models
from utils.misc import log

training_env = None
testing_env = None

def train_eval_mapper(optim_row: tuple) -> tuple:
    itrs, _hcfg = optim_row
    # instance and train model
    model = models[_hcfg['class']]('sc', _hcfg)
    model.train(training_env.value, testing_env.value)
    model.save()
    # run evaluation in testing env
    _preds, metric = model.evaluate(testing_env.value)
    _hcfg['metric'] = metric
    return (itrs, _hcfg)

def parallel_run_crossvalidation(sc: SparkContext, 
                    training,
                    testing,
                    optim: dict, cfg: dict):
    """ Parallel MapReduce implementation of crossvalidation process 

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
    # broadcast envs
    training_env = sc.broadcast(training)
    testing_env = sc.broadcast(testing)
    hcfgs = {}
    metric_series = []
    for itrs in range(int(optim['max_iters'] / optim['num_workers'])):
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