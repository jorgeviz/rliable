from models.base_model import BaseModel
from models.dqn import DQN

models = {
    "BaseModel": BaseModel,
    "QLearning": DQN
}