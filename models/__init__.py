from models.base_model import BaseModel
from models.dqn import DQN
from models.count_model import CountModel

models = {
    "BaseModel": BaseModel,
    "DQN": DQN,
    "CountModel": CountModel
}