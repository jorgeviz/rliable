# RLiable 

<div align="center">
<img src="docs/assets/rliable-logo.png" style="zoom:30%;" />
</div>

![Code linting](https://github.com/jorgeviz/rliable/workflows/Code%20Linting/badge.svg)

RLiable is an experiment parallelization framework for fast hyper-parameter tuning of reinforcement learning agents. It aims to fill the need for a distributable Spark/TF-compatible models that allows to scale experimentation in an easy and `reliable` way.


## Execution

Prerequisites:

|        | Version |
| ------ | ------- |
| Python | >=3.6   |
| Spark | 3.0.1  |
| Hadoop | 2.7  |
| Java | 1.8  |
| Scala | 2.11  |


### Running experiments

Current optimization algorithm uses a naive random search based on the range or categorical selections of the configuration flag `optim/config/optim_*.json`.  

An example for the DQN implementation of the CartPole OpenAI environment has the following schema:

```js
{
    "config": "config/config_dqn.json",
    "optim_name": "RandomOptimDQN",
    "num_workers": 2,
    "grid": {
        "train_iterations": {
            "min": 1000,
            "max": 4000
        },
        "collect_steps_per_iteration": [1],
        "batch_size": [32, 64],
        "learning_rate": {
            "min": 0.001, 
            "max": 0.005
        },
        "qnet_fc_hidden_size": [20, 100, 400],
        "num_eval_episodes": [200]
    },
    "metric": {
        "name": "reward",
        "criteria": "max"
    },
    "max_iters": 3,
    "convergence": 0.01
}
```

Based on the specified optimization config file, we can run the process with the following command:

```bash
python optimize.py optim/config/optim_dqn.json
```

The implementation allows 3 different modes, single process, and parallelized level 1 and 2. For the level 1, the system spans batches of training-evaluation jobs per each of the workers available. For the level 2, it divides the training and evaluation process to allow parallelization on the rollouts dividing the load of interaction with the environment.

Alternative flags:

- `--parallelized`: Sets parallelized mode on
- `--plevel=LEVEL`: (Default 1), sets the level of parallelization described in train and eval processes.


### Visualizing Results

- TODO

---

## Project Structure

```
├── README.md
├── docs
│   └── assets
├── config
│   ├── config.py
│   ├── config_base.json
│   └── ...
├── models
│   ├── __init__.py
│   ├── base_model.py
│   └── dqn.py
├── evaluate.py
├── scripts
│   └── ...
├── train.py
└── utils
    ├── metrics.py
    └── misc.py
```

### Custom Models

Each model has 3 main methods `train`, `predict` and  `load_model`, and most inherit from the `BaseModel` class or ensure to handle within the class the configuration dict and PySpark Context.  

```python
# models/base_model.py

class BaseModel(object):

    def __init__(self, sc, cfg):
        """ Base model constructor
        """
        self._sc = sc
        self.cfg = cfg

    def train(self, data, *args):
        """ Training method

            Parameters:
            -----
            data: pyspark.rdd | tf_agents.environments.TFPyEnvironment
                Training Data or environment
        """
        pass
    
    def evaluate(self, data):
        """ Evaluation method

            Parameters:
            -----
            data: pyspark.rdd | tf_agents.environments.TFPyEnvironment
                Training Data or environment
        """
        pass
```



For model's addition to the pipeline, it has to be registered in the `__init__.py` file from the models directory, and define a configuration as the one in the `config`folder to point to the key class of the registered model, the trining data file, the model output file and its respective hyper parameters.

```json
{
    "class": "BaseModel",
    "environment": "CartPole-V1",
    "mdl_file": "weights/base.model",
    "hp_params": {
        "learning_rate": 0.001
    }
}
```

## Use References

- [Local Properties](https://mallikarjuna_g.gitbooks.io/spark/content/spark-sparkcontext-local-properties.html)
- [Spark Job Scheduling](https://spark.apache.org/docs/latest/job-scheduling.html)
