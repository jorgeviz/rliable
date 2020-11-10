from pprint import pformat
import json
import uuid
from pathlib import Path
from pyspark import SparkContext
from pyspark.rdd import RDD

from rliable.core import sample_random_hyperconfig, has_converged
from models import models
from utils.misc import log


def serial_run_crossvalidation(sc: SparkContext, 
                    training,
                    testing,
                    optim: dict, cfg: dict):
    """ Serial implementation of crossvalidation process 

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
    hcfgs = {}
    metric_series = []
    for itrs in range(int(optim['max_iters'])):
        log(f"Running CV-{itrs}")
        _hcfg = sample_random_hyperconfig(optim['grid'], cfg)
        hcfgs[itrs] = _hcfg
        # instance and train model
        model = models[_hcfg['class']](sc, _hcfg)
        model.train(training, testing)
        model.save()
        # run evaluation in testing env
        _preds, metric = model.evaluate(testing)
        hcfgs[itrs]['metric'] = float(metric)
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
    cv_results_path = (Path(cfg['mdl_file']).parent / f'single_cv-{uuid.uuid4()}.json').as_posix()
    with open(cv_results_path, 'w') as f:
        f.write(json.dumps(hcfgs))