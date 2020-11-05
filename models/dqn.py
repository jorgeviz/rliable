import os
from pathlib import Path
from collections import OrderedDict, Counter, namedtuple

import tensorflow as tf
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.policies import policy_saver
import matplotlib.pyplot as plt


from models.base_model import BaseModel
from utils.misc import log, debug, collect_step
from utils.metrics import compute_avg_return


class DQN(BaseModel):
    """ DQN Model Wrapper 
    """

    def __init__(self, sc, cfg):
        """ DQN constructor
        """
        super().__init__(sc,cfg)
        # hyper params
        self.train_iters = cfg['hp_params']['train_iterations']
        self.collect_steps_per_iter = cfg['hp_params'].get('collect_steps_per_iteration', 1)
        self.batch_size = cfg['hp_params'].get('batch_size', 64)
        self.lr = cfg['hp_params'].get('learning_rate', 1e-3)
        self.log_interval = cfg['hp_params'].get('log_interval', 200)
        self.qnet_fc_hidden_size = (cfg['hp_params'].get('qnet_fc_hidden_size', 100),)
        self.num_eval_episodes = cfg['hp_params'].get('num_eval_episodes', 10)
        self.eval_interval = cfg['hp_params'].get('eval_interval', 1000)
        self.replay_buffer_max_length = cfg['hp_params'].get('replay_buffer_max_length', 100000)
        self.output_dir = Path(cfg['mdl_file'])
        # output dir validation
        if not self.output_dir.is_dir():
            self.output_dir.mkdir()
        # model vars
        self.q_net = None
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step_counter = tf.Variable(0)
        self.agent = None
        self.policy = None
        # data vars
        self.train_env = None
        self.eval_env = None
        self.replay_buffer = None
        self.dataset = None
        self.iterator = None
        self.eval_rewards = []

    def initialize_agent(self):
        """ Instance TF agent with hparams
        """
        # Q network
        self.q_net = q_network.QNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            fc_layer_params=self.qnet_fc_hidden_size
        )
        # DQN agent
        self.agent = dqn_agent.DqnAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            q_network=self.q_net,
            optimizer=self.optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.train_step_counter
        )
        self.agent.initialize()
        self.policy = self.agent.policy
    
    def initialize_collection(self):
        """ Instance replay buffer and populate first batch with random policy
        """
        random_policy = random_tf_policy.RandomTFPolicy(
                    self.train_env.time_step_spec(),
                    self.train_env.action_spec()
        )
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self.replay_buffer_max_length
        )
        # collect data and initialize iterator
        for _ in range(self.batch_size):
            collect_step(self.train_env, random_policy, self.replay_buffer)
        self.dataset = self.replay_buffer\
                            .as_dataset(
                                num_parallel_calls=3, 
                                sample_batch_size=self.batch_size, 
                                num_steps=2)\
                            .prefetch(3)
        self.iterator = iter(self.dataset)

    def save(self, *args):
        """ Save model values
        """
        # Save training reward curve
        iterations = range(0, self.train_iters + 1, self.eval_interval)
        plt.plot(iterations, self.eval_rewards)
        plt.ylabel('Average Return')
        plt.xlabel('Iterations')
        plt.ylim(top=250)
        plt.savefig(self.output_dir.as_posix()+"/dqn_train_eval_returns.png", dpi=100)
        # Save agent and policy
        checkpoint_dir = os.path.join(self.output_dir.as_posix(), 'checkpoint')
        train_checkpointer = common.Checkpointer(
            ckpt_dir=checkpoint_dir,
            max_to_keep=1,
            agent=self.agent,
            policy=self.agent.policy,
            replay_buffer=self.replay_buffer,
            global_step=self.train_step_counter
        )
        policy_dir = os.path.join(self.output_dir.as_posix(), 'policy')
        tf_policy_saver = policy_saver.PolicySaver(self.agent.policy)
        train_checkpointer.save(self.train_step_counter)
        tf_policy_saver.save(policy_dir)

    def _train(self):
        """ Private training method
        """
        # tf function wrapper
        self.agent.train = common.function(self.agent.train)
        # Reset the train step
        self.agent.train_step_counter.assign(0)
        # Evaluate the agent's policy once before training.
        avg_return = compute_avg_return(self.eval_env, self.agent.policy, self.num_eval_episodes)
        self.eval_rewards.append(avg_return)
        # train loop
        for _ in range(int(self.train_iters)):
            # Collect a few steps using collect_policy and save to the replay buffer.
            for _s in range(self.collect_steps_per_iter):
                collect_step(self.train_env, self.agent.collect_policy, self.replay_buffer)
            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(self.iterator)
            train_loss = self.agent.train(experience).loss
            step = self.agent.train_step_counter.numpy()
            if step % self.log_interval == 0:
                log('step = {0}: loss = {1}'.format(step, train_loss))
            if step % self.eval_interval == 0:     
                avg_return = compute_avg_return(self.eval_env, self.agent.policy, self.num_eval_episodes)
                log('step = {0}: Average Return = {1}'.format(step, avg_return))
                self.eval_rewards.append(avg_return)
        log("Best episode avg reward:", max(self.eval_rewards))

    def train(self, tenv, eenv):
        """ Training method

            Params:
            ----
            tenv: tf_agent.environments.TFPyEnvironment
                Training Environment
            eenv: tf_agent.environments.TFPyEnvironment
                Validation Environment
        """
        self.train_env = tenv
        self.eval_env = eenv
        self.initialize_agent()
        self.initialize_collection()
        self._train()

    def load_model(self):
        """ Load model from config defined model file (only policy)
        """
        policy_dir = os.path.join(self.output_dir.as_posix(), 'policy')
        self.policy = tf.compat.v2.saved_model.load(policy_dir)

    def evaluate(self, test):
        """ Evaluate method

            Params:
            ----
            test: tf_agents.environments.TFPyEnvironment
                Test environments
        """
        eval_avg_return = compute_avg_return(test, self.policy, self.num_eval_episodes)
        log("Eval Avg Reward:", eval_avg_return)
        return (eval_avg_return, eval_avg_return)