import json
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
        self._mdl_sum = 0
    
    def save(self, *args):
        """ Save model values
        """
        with open(self.cfg['mdl_file'], 'w') as joi:
            joi.write(json.dumps({
                "count": self._mdl_count,
                "sum": self._mdl_sum
            }))

    def train(self, data):
        """ Training method

            Params:
            ----
            data: pyspark.rdd
                Counting json data
        """
        log("Training data:\n", data.take(4))
        ds = data.map(lambda j: j['x'])
        self._mdl_count = ds.count()
        self._mdl_sum = ds.sum()

    def load_model(self):
        """ Load model from config defined model file
        """
        with open(self.cfg['mdl_file'], 'r') as joi:
            jd = json.loads(joi.read())
            self._mdl_count = jd['count']
            self._mdl_sum = jd['sum']

    def evaluate(self, test, outfile):
        """ Evaluate method

            Params:
            ----
            test: pyspark.rdd
                Test Data
            outfile: str
                Output file Path
        """
        _mdl_count = self._mdl_count
        preds = test.ZipWithIndex()\
                    .map(lambda y: y[1] + _mdl_count)

        def _persist_preds(buff, val):
            buff.write(f"{val}\n")
            return val

        with open(outfile, 'w') as joi:
            save_op = preds.map(lambda y: _persist_preds(joi, y))
            log("Saved", save_op.count(), "predictions!")