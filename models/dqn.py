import re
import os
import json
from pprint import pprint
import math
from collections import OrderedDict, Counter, namedtuple
from pyspark.mllib.linalg import SparseVector
from pyspark.sql import SQLContext, Row

from models.base_model import BaseModel
from utils.misc import log, debug, read_json


class DQN(BaseModel):
    """ DQN Model Wrapper 
    """

    def __init__(self, sc, cfg):
        """ DQN constructor
        """
        # [TODO]
        super().__init__(sc,cfg)

    
    def save(self, *args):
        """ Save model values
        """
        # [TODO]
        raise NotImplementedError

    def train(self, data):
        """ Training method

            Params:
            ----
            data: pyspark.rdd
        """
        # [TODO]
        raise NotImplementedError

    def load_model(self):
        """ Load model from config defined model file
        """
        # [TODO]
        raise NotImplementedError

    def evaluate(self, test, outfile):
        """ Evaluate method

            Params:
            ----
            test: pyspark.rdd
                Test Data
            outfile: str
                Output file Path
        """
        # [TODO]
        raise NotImplementedError