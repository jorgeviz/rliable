from models.base_model import BaseModel
from utils.misc import log, debug, read_json


class CountModel(BaseModel):
    """ Simple Count Model example:
        - It counts elements in the dataset and returns the
            next counts based on the current count (at evaluation)

    """

    def __init__(self, sc, cfg):
        """ Count Model constructor
        """
        super().__init__(sc,cfg)
        self._mdl_count = 0
    
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
                Counting json data
        """
        log("Training data:\n", data.take(10))

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