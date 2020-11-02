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

- TODO

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

Each model has 3 main methods `train`, `predict` and  `load_model`, and most inherit from the `BaseModel` class or ensure to handle within the class the configuration dict and PySpark Context.  

```python
# models/base_model.py

class BaseModel(object):

    def __init__(self, sc, cfg):
        """ Base model constructor
        """
        self._sc = sc
        self.cfg = cfg

    def train(self, data):
        """ Training method

            Params:
            -----
            data: pyspark.rdd
                Input Data
        """
        pass
    
    def predict(self, data):
        """ Prediction method
        """
        pass
```



For model's addition to the pipeline, it has to be registered in the `__init__.py` file from the models directory, and define a configuration as the one in the `config`folder to point to the key class of the registered model, the trining data file, the model output file and its respective hyper parameters.

```json
{
    "class": "BaseModel",
    "environment": "../../data/project/cart-pole.json",
    "mdl_file": "weights/base.model",
    "hp_params": {
    }
}
```

