""" Single execution script for DQN training and evaluation
"""
import os
import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.policies import policy_saver

######################################
#  Hyper parameters
######################################
num_iterations = 2000  # -- training iterations

initial_collect_steps = 100  # -- data collection vars
collect_steps_per_iteration = 1 
replay_buffer_max_length = 100000 

batch_size = 64   # -- training params
learning_rate = 1e-3  
log_interval = 200  

qnet_fc_layer_params = (100,) # -- Q-network params

num_eval_episodes = 10  # -- eval params 
eval_interval = 1000  

output_dir = "output"

######################################
# Environments definition
######################################
env_name = 'CartPole-v0'
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)
# - tf wrapper
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

######################################
# Agent Model initalization
######################################
# Q network
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=qnet_fc_layer_params)
# Optimizer
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)
# DQN agent
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter
)
agent.initialize()
# Policies
eval_policy = agent.policy
collect_policy = agent.collect_policy
random_policy = random_tf_policy.RandomTFPolicy(
    train_env.time_step_spec(),
    train_env.action_spec())
# Replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length
)

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
# Data collection
######################################
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

def collect_data(env, policy, buffer, steps):
    """ Collect N steps from the environment into the Replay buffer

    Parameters
    ----------
    env : tf.agent.Environment
        Environment
    policy : tf.agent.Policy
        Agent policy
    buffer : tf.agent.ReplayBuffer
        Replay buffer with collected trajectories
    steps : int
        Number of steps
    """
    for _ in range(steps):
        collect_step(env, policy, buffer)

# Collect initial steps from random policy
collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)
# Convert replay buffer into tf dataset
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2
    ).prefetch(3)
iterator = iter(dataset)

######################################
#  Training process
######################################
def train(returns):
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns.append(avg_return)

    for _ in range(num_iterations):
        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:     
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)

# call training
train_eval_returns = []
train(train_eval_returns)
print("Best Avg Reward:", max(train_eval_returns))
######################################
# Results
######################################
try:
    os.mkdir(output_dir)
except FileExistsError:
    pass

# Save training reward curve
iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, train_eval_returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=250)
plt.savefig(output_dir+"/dqn_train_eval_returns.png", dpi=100)

# Save agent and policy
checkpoint_dir = os.path.join(output_dir, 'checkpoint')
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=train_step_counter
)
policy_dir = os.path.join(output_dir, 'policy')
tf_policy_saver = policy_saver.PolicySaver(agent.policy)

train_checkpointer.save(train_step_counter)
tf_policy_saver.save(policy_dir)
