""" Single execution script for DQN policy execution
"""
import os

import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment


######################################
#  Hyper parameters
######################################
collect_steps_per_iteration = 1 

num_eval_episodes = 10  # -- eval params 
eval_interval = 1000  
output_dir = "output"

######################################
# Environments definition
######################################
env_name = 'CartPole-v0'
eval_py_env = suite_gym.load(env_name)
# - tf wrapper
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

######################################
# Evaluation metrics    
######################################
def compute_avg_return(environment, policy, num_episodes=10):
    """ Computing average reward in N episodes

    Parameters
    ----------
    environment: tf.agents.Environment
        Environment
    policy : tf.agents.Policy
        Agent policy
    num_episodes : int, optional
        Num of episodes to eval, by default 10

    Returns
    -------
    float
        Average reward
    """
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


######################################
#  Evaluating pretrained policy
######################################

policy_dir = os.path.join(output_dir, 'policy')
saved_policy = tf.compat.v2.saved_model.load(policy_dir)
eval_avg_return = compute_avg_return(eval_env, saved_policy, num_eval_episodes)
print("Eval Avg Reward:", eval_avg_return)