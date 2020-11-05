import os
import json
from pathlib import Path


os.environ['PYSPARK_PYTHON'] = 'python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python3'

# -- Application vars
APP_NAME = "RLiable-v0.0.1"

# -- Model configurations
# model_conf = "config/config_base.json"
model_conf = "config/config_dqn.json"
# model_conf = "config/config_count.json"

def validate_dirs(mcf):
    mdl_f = Path(os.path.abspath(mcf))
    if not os.path.exists(mdl_f.parent):
        os.makedirs(mdl_f.parent)

def load_conf(mconf=model_conf):
    with open(mconf, 'r') as mf:
        mdl_cnf = json.load(mf)
    assert 'class' in mdl_cnf, "No Model Class!"
    assert 'mdl_file' in mdl_cnf, "No Model output file!"
    assert 'hp_params' in mdl_cnf, "No Hyperparameters!"
    assert 'environment' in mdl_cnf, "No environment specification!"
    validate_dirs(mdl_cnf['mdl_file'])
    return mdl_cnf
