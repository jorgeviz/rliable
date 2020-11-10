import sys
import os
import json
from pathlib import Path
from collections import OrderedDict

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import trajectory


MAX_PART_SIZE = 10 * (1024**2)

def fetch_arg(_pos):
    """ Fetch arg from sys args
    """
    if len(sys.argv) <= _pos:
        raise Exception(
            "Missing arguments: <test_file> <output_file>")
    return sys.argv[_pos]

def parse_predit_args():
    """ Parse arguments
    """
    args = OrderedDict({
        "test_file": str
    })
    for i, a in enumerate(args):
        args[a] = args[a](fetch_arg(i + 1))
    return args

def log(*msg, lvl="INFO"):
    """ Log message with visual help
    """
    print("-"*50)
    print("[{}]".format(lvl), end=" ")
    for m in msg:
        print("{}".format(m), end=" ")
    print("\n" + "-"*50)

def read_file(sc, fpath):
    """ Read a file
    """
    _fsize = Path(fpath).stat().st_size
    return sc.textFile(fpath, _fsize // MAX_PART_SIZE )

def read_json(sc, fpath):
    """ Read JSON-rowed file parsed in to RDD
    """
    data = read_file(sc, fpath)\
            .map(lambda x: json.loads(x))
    return data

def read_csv(sc, fpath, with_heads=False):
    """ Read and parse CSV into RDD
    """
    def filter_heads(z): 
        return z[1] > 0
    data = read_file(sc, fpath)\
        .zipWithIndex()\
        .filter(lambda z: True if with_heads else filter_heads(z))\
        .map(lambda z: tuple(z[0].split(',')))
    return data

def read_env(sc, environmnt, max_episode_steps=None):
    """ Read environment 
    """
    # static files
    if os.path.exists(environmnt):
        if ".json" in environmnt:
            return read_json(sc, environmnt)
        return read_file(sc, environmnt)
    # suite env
    _py_env = suite_gym.load(environmnt, max_episode_steps=max_episode_steps)
    return tf_py_environment.TFPyEnvironment(_py_env)

def collect_step(environment, policy, buffer):
    """ Collects a timesetp interaction with the environment and the agent

    Parameters
    ----------
    environment : tf.agent.Environment
        Environment
    policy : tf.agent.Policy
        Agent policy
    buffer : tf.agent.ReplayBuffer
        Replay buffer with collected trajectories
    """
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    # Add trajectory to the replay buffer
    buffer.add_batch(traj)

def debug():
    """ Local debug
    """
    import ipdb; ipdb.set_trace()