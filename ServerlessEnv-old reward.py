import copy
import logging
import os

import math
import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from gym.utils import seeding

import constants
from gym.spaces import Discrete, Box, MultiDiscrete
import ServEnv_base
import definitions as defs
from sys import maxsize
from ServEnv_base import Worker
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
# from keras.optimizers import Adam, RMSprop
from kubernetes import client, config, watch
from datetime import datetime
import time
import json
import threading
from collections import deque
from queue import Queue
import xlwt
from tensorflow.python.keras.layers import LSTM
from xlwt import Workbook
from tensorflow.keras.callbacks import TensorBoard
# import keras.backend.tensorflow_backend as backend
import tensorflow as tf
import subprocess

MODEL_NAME = "Serverless_Scaling"
# PATH = 'D:/WL generation/Third_work/Simulator/training_agent'
PATH = ''

FN_TYPE = ""

actions_dict = {0: (-920, -3), 1: (-920, -2), 2: (-920, -1), 3: (-920, 0), 4: (-920, 1), 5: (-920, 2), 6: (-920, 3),
                7: (-460, -3), 8: (-460, -2), 9: (-460, -1), 10: (-460, 0), 11: (-460, 1), 12: (-460, 2), 13: (-460, 3),
                14: (0, -3), 15: (0, -2), 16: (0, -1), 17: (0, 0), 18: (0, 1), 19: (0, 2), 20: (0, 3), 21: (460, -3),
                22: (460, -2), 23: (460, -1), 24: (460, 0), 25: (460, 1), 26: (460, 2), 27: (460, 3), 28: (920, -3),
                29: (920, -2), 30: (920, -1), 31: (920, 0), 32: (920, 1), 33: (920, 2), 34: (920, 3)}
# 45

# action_cpu = [-9200, -8740, -8280, -7820, -7360, -6900, -6440, -5980, -5520, -5060, -4600, -4140, -3680, -3220, -2760,
#               -2300, -1840, -1380, -920, -460, -230, -115, 0, 115, 230, 460, 920, 1380, 1840, 2300, 2760, 3220, 3680, 4140, 4600, 5060, 5520, 5980,
#               6440, 6900, 7360, 7820, 8280, 8740, 9200]
# action_mem = [-3000, -2850, -2700, -2550, -2400, -2250, -2100, -1950, -1800, -1650, -1500, -1350, -1200, -1050, -900,
#               -750, -600, -450, -300, -150, -75, 0, 75, 150, 300, 450, 600, 750, 900, 1050, 1200, 1350, 1500, 1650, 1800, 1950,
#               2100, 2250, 2400, 2550, 2700, 2850, 3000]

action_cpu = [-920, -690, -460, -230, -115, 0, 115, 230, 460, 690, 920]

action_mem = [-375, -300, -225, -150, -75, 0, 75, 150, 225, 300, 375]

# 43
action_rep = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

action_util = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
# 17

# 0 > S 1 > NS
action_env = [0, 1]

wb_summary = Workbook()
ep_reward_sheet = wb_summary.add_sheet('Episodic_reward')
ep_lat_sheet = wb_summary.add_sheet('Episodic_latency')
ep_failure_rate_sheet = wb_summary.add_sheet('Episodic_failure_rate')
ep_ec2_cost_sheet = wb_summary.add_sheet('Episodic_ec2_vm_cost')
ep_ec2_time_sheet = wb_summary.add_sheet('Episodic_ec2_vm_uptime')
ep_serverless_cost_sheet = wb_summary.add_sheet('Episodic_serverless_fn_cost')
ep_t_cost_sheet = wb_summary.add_sheet('Episodic_total_user_cost')
ep_env_action_sheet = wb_summary.add_sheet('Deploy_env')
ep_second_action_sheet = wb_summary.add_sheet('Second_action')
# step_latency_sheet = wb_summary.add_sheet('Fn_Latency_step_based')
step_failure_rate_sheet = wb_summary.add_sheet('Fn_fail_rate_step_based')
step_t_costdiff_sheet = wb_summary.add_sheet('total_vm_cost_diff')


# def OurModel(input_shape, action_space):
#     X_input = Input(input_shape)
#
#     # 'Dense' is the basic form of a neural network layer
#     # Input Layer of state size(4) and Hidden Layer with 512 nodes
#
#     # only FC layers
#     X = Dense(92, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)
#     # X = Dense(137, activation="relu", kernel_initializer='he_uniform')(X)
#     X = Dense(92, activation="relu", kernel_initializer='he_uniform')(X)
#     # Output Layer with # of actions: 2 nodes (left, right)
#     X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)
#
#     model = Model(inputs=X_input, outputs=X)
#     model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
#
#     model.summary()
#     return model


# class ActorCriticModel(Model):
#     def __init__(self, state_size, action_size):
#         super(ActorCriticModel, self).__init__()
#         self.state_size = state_size
#         self.action_size = action_size
#         self.total_action_size = action_size[0] + action_size[1] + action_size[2]
#         self.dense1 = tf.keras.layers.Dense(100, activation='relu')
#         self.dense2 = tf.keras.layers.Dense(100, activation='relu')
#         self.policy_logits_act = tf.keras.layers.Dense(self.total_action_size)
#
#         # self.dense2 = tf.keras.layers.Dense(100, activation='relu')
#         # self.policy_logits_mem = tf.keras.layers.Dense(action_size[1])
#         #
#         # self.dense3 = tf.keras.layers.Dense(100, activation='relu')
#         # self.policy_logits_rep = tf.keras.layers.Dense(action_size[2])
#
#         self.dense3 = tf.keras.layers.Dense(100, activation='relu')
#         self.dense4 = tf.keras.layers.Dense(100, activation='relu')
#         self.values = tf.keras.layers.Dense(1, activation='linear')
#
#     def call(self, inputs):
#         # Forward pass
#         x1 = self.dense1(inputs)
#         x2 = self.dense2(x1)
#         logits_act = self.policy_logits_act(x2)
#
#         # x2 = self.dense2(inputs)
#         # logits_mem = self.policy_logits_mem(x2)
#         #
#         # x3 = self.dense3(inputs)
#         # logits_rep = self.policy_logits_rep(x3)
#
#         x3 = self.dense3(inputs)
#         v1 = self.dense4(x3)
#         values = self.values(v1)
#         # return logits_cpu, logits_mem, logits_rep, values
#
#         return logits_act, values


class ModifiedTensorBoard(TensorBoard):
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, name)
        print("location:" + str(self._log_write_dir))

        # self._train_dir = 'train'

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        # self.model = model
        self.model = model
        self._log_write_dir = self.log_dir
        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter
        # pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


class ThreadLogFilter(logging.Filter):
    """
    This filter only show log entries for specified thread name
    """

    def __init__(self, thread_name, *args, **kwargs):
        logging.Filter.__init__(self, *args, **kwargs)
        self.thread_name = thread_name

    def filter(self, record):
        return record.threadName == self.thread_name


def start_thread_logging(id, level, type):
    thread_name = threading.Thread.getName(threading.current_thread())
    log_file = "log/" + str(id) + "/" + level + "-" + type + "-logfile.log"
    log_handler = logging.FileHandler(log_file)
    log_handler.setLevel(logging.DEBUG)
    log_filter = ThreadLogFilter(thread_name)
    log_handler.addFilter(log_filter)
    logger = logging.getLogger()
    logger.addHandler(log_handler)

    return log_handler


class ServerlessEnv(gym.Env):
    """A serverless environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.fn_type = ""
        self.current_request = None
        self.episode_no = 0
        # self.env_base = ServEnv_base(self.fn_type, self.episode_no)
        logging.basicConfig(filename="log/" + self.fn_type + "-logfile.log", filemode="w",
                            level=logging.DEBUG)

        logging.info("Serverlessenv is initialized")
        # self.lr = 0.001
        self.num_max_replicas = constants.max_num_replicas
        self.num_max_serverless_vms = constants.max_num_serverless_vms

        # State includes server metrics, individual function metrics
        self.observation_space = Box(low=np.array(np.zeros(124)),
                                     high=np.array([maxsize] * 124),
                                     dtype=np.float32)
        self.state_size = self.observation_space.shape[0]
        self.state = np.zeros(124)
        # self.opt = tf.optimizers.Adam(self.lr)
        # self.gamma = 0.95  # discount rate
        self.done = False
        self.action = 0
        self.act_type = ""
        self.reward = 0
        self.episode_success = False
        self.current_count = 1
        self.episode_cost = 0
        self.simulation_running = True
        self.worker = ServEnv_base.Worker(self.episode_no)
        self.clock = self.worker.clock
        # self.tensorboard = ModifiedTensorBoard(MODEL_NAME, log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
        features = 20
        container_id = 1

    def save(self, name):
        self.main_network.save(name)

    def action_spec(self):
        return self.action_space

    def observation_spec(self):
        return self.observation_space

    def reset(self):
        self.episode_no += 1
        self.worker = ServEnv_base.Worker(self.episode_no)
        self.simulation_running = True
        # self.serverless_vms = copy.deepcopy(ServEnv_base.serverless_vms)
        # self.state = copy.deepcopy(np.array(ServEnv_base.gen_serv_env_state(FN_TYPE)))
        # self.state = copy.deepcopy(np.array(ServEnv_base.gen_serv_env_state(self.fn_type)))
        self.done = False
        self.reward = 0

        # self.jobs = copy.deepcopy(cluster.JOBS)
        self.clock = self.worker.clock
        self.episode_success = False
        self.current_count = 1
        self.info = {}
        # ServEnv_base.sorted_request_history_per_window[self.fn_type] = []
        #       self.app_idx = self.app_idx+1
        # return self.state

    def load(self, name):
        self.main_network = load_model(name)

    # def replay(self, terminal_state, mem, train_s):
    #     if len(mem) >= train_s:
    #         print("Memory replaying...")
    #         # logging.info(
    #         #     "Now memory replaying")
    #         # minibatch = random.sample(memory, min(len(memory), self.batch_size))
    #         minibatch = random.sample(mem, self.batch_size)
    #
    #         state = np.zeros((self.batch_size, self.state_size))
    #         # next_state = np.zeros((self.batch_size, self.state_size))
    #         next_state = np.zeros((self.batch_size, self.state_size))
    #         action, reward, done = [], [], []
    #
    #         # do this before prediction
    #         # for speedup, this could be done on the tensor level
    #         # but easier to understand using a loop
    #         for i in range(self.batch_size):
    #             state[i] = minibatch[i][0]
    #             action.append(minibatch[i][1])
    #             reward.append(minibatch[i][2])
    #             next_state[i] = minibatch[i][3]
    #             done.append(minibatch[i][4])
    #
    #         # do batch prediction to save speed - here target means a Q value.
    #         target = self.main_network.predict(state)
    #         target_next = self.target_network.predict(next_state)
    #
    #         for i in range(self.batch_size):
    #             # correction on the Q value for the action used
    #             if done[i]:
    #                 target[i][action[i]] = reward[i]
    #             else:
    #                 # Standard - DQN
    #                 # DQN chooses the max Q value among next actions
    #                 # selection and evaluation of action is on the target Q Network
    #                 # Q_max = max_a' Q_target(s', a')
    #                 target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))
    #
    #         self.main_network.fit(state, target, batch_size=self.batch_size, verbose=0, shuffle=False,
    #                               callbacks=[self.tensorboard] if terminal_state else None)
    #
    #     if terminal_state:
    #
    #         total_req_count = 0
    #         failed_count = 0
    #         fn_failure_rate = 0
    #         req_info = {}
    #         fn_latency = 0
    #         with ServEnv_base.history_lock:
    #             for req in ServEnv_base.sorted_request_history_per_window:
    #                 total_req_count += 1
    #                 if req.status != "Dropped":
    #                     if req.type in req_info:
    #                         # self.logging.info(
    #                         #     "Req id: {}, arrival time: {}, finish time: {}".format(req.id, req.arrival_time,
    #                         #                                                            req.finish_time))
    #                         req_info[req.type]['execution_time'] += req.finish_time - req.arrival_time
    #                         req_info[req.type]['req_count'] += 1
    #                     else:
    #                         req_info[req.type] = {}
    #                         # self.logging.info(
    #                         #     "Req id: {}, arrival time: {}, finish time: {}".format(req.id, req.arrival_time,
    #                         #                                                            req.finish_time))
    #                         req_info[req.type]['execution_time'] = req.finish_time - req.arrival_time
    #                         req_info[req.type]['req_count'] = 1
    #                 else:
    #                     failed_count += 1
    #
    #         for req_type, req_data in req_info.items():
    #             # self.info(
    #             #     "FN type: {}, Total exec time: {}, Req count: {}, MIPS for type: {}".format(req_type, req_data[
    #             #         'execution_time'], req_data['req_count'], ServEnv_base.fn_features[str(
    #             #         req_type) + "_req_MIPS"]))
    #             fn_latency += (req_data['execution_time'] / req_data['req_count']) / (
    #                     int(self.worker.fn_features[str(req_type) + "_req_MIPS"]) / constants.MIPS_for_one_request)
    #
    #         if len(req_info) != 0:
    #             ServEnv_base.Episodic_latency = fn_latency / len(req_info) / constants.max_step_latency_perfn
    #         logging.info("CLOCK: {} Overall latency: {}".format(self.worker.clock, fn_latency))
    #         ec2_vm_up_time_cost, ec2_vm_up_time, serverless_fn_cost = self.calc_total_user_cost()
    #         if total_req_count != 0:
    #             ServEnv_base.Episodic_failure_rate = failed_count / total_req_count
    #         logging.info("CLOCK: {} Cum vm_up_time_cost: {}".format(self.worker.clock, ec2_vm_up_time_cost))
    #         logging.info("CLOCK: {} fn_failure_rate: {}".format(self.worker.clock, fn_failure_rate))
    #
    #         self.tensorboard.update_stats(Episodic_reward=self.worker.episodic_reward)
    #         self.tensorboard.update_stats(Function_Latency_step_based=self.worker.function_latency)
    #         self.tensorboard.update_stats(Function_failure_rate_total_step_based=self.worker.fn_failures)
    #         # self.tensorboard.update_stats(Total_VM_TIME_DIFF=ServEnv_base.total_vm_time_diff)
    #         self.tensorboard.update_stats(Total_VM_COST_DIFF=self.worker.total_vm_cost_diff)
    #         self.tensorboard.update_stats(Vertical_cpu_action_total=self.worker.ver_cpu_action_total)
    #         self.tensorboard.update_stats(Vertical_mem_action_total=self.worker.ver_mem_action_total)
    #         self.tensorboard.update_stats(Horizontal_action_total=self.worker.hor_action_total)
    #         self.tensorboard.update_stats(Episodic_latency=self.worker.Episodic_latency)
    #         self.tensorboard.update_stats(Episodic_failure_rate=self.worker.Episodic_failure_rate)
    #         # self.tensorboard.update_stats(Active_node_penalty=act_node_penalty)
    #         # agent.tensorboard.update_stats(CPU_util_penalty=cpu_util_penalty)
    #
    #     print("Tensorboard step: " + str(self.tensorboard.step))

    def graphs(self, write_graph, ep):
        total_req_count = 0
        failed_count = 0
        fn_failure_rate = 0
        req_info = {}
        fn_latency = 0

        # with ServEnv_base.history_lock:
        if self.worker.fn_type in self.worker.sorted_request_history_per_window:
            for req in self.worker.sorted_request_history_per_window[self.worker.fn_type]:
                total_req_count += 1
                # if req.status != "Dropped":
                #     if req.type in req_info:
                #         # self.logging.info(
                #         #     "Req id: {}, arrival time: {}, finish time: {}".format(req.id, req.arrival_time,
                #         #                                                            req.finish_time))
                #         req_info[req.type]['execution_time'] += req.finish_time - req.arrival_time
                #         req_info[req.type]['req_count'] += 1
                #     else:
                #         req_info[req.type] = {}
                #         # self.logging.info(
                #         #     "Req id: {}, arrival time: {}, finish time: {}".format(req.id, req.arrival_time,
                #         #                                                            req.finish_time))
                #         req_info[req.type]['execution_time'] = req.finish_time - req.arrival_time
                #         req_info[req.type]['req_count'] = 1
                # else:
                if req.status == "Dropped":
                    failed_count += 1

        # for req_type, req_data in req_info.items():
        #     # self.logging.info(
        #     #     "FN type: {}, Total exec time: {}, Req count: {}, MIPS for type: {}".format(req_type, req_data[
        #     #         'execution_time'], req_data['req_count'], ServEnv_base.fn_features[str(
        #     #         req_type) + "_req_MIPS"]))
        #     fn_latency += (req_data['execution_time'] / req_data['req_count']) / (
        #             int(self.worker.fn_features[str(req_type) + "_req_MIPS"]) / constants.MIPS_for_one_request)

        # if len(req_info) != 0:
        #     self.worker.Episodic_latency = fn_latency / len(req_info)
        # logging.info("CLOCK: {} Overall latency: {}".format(self.worker.clock, fn_latency))
        self.worker.ec2_vm_up_time_cost, self.worker.ec2_vm_up_time, self.worker.serverless_fn_cost = self.calc_total_user_cost()
        if total_req_count != 0:
            self.worker.Episodic_failure_rate = failed_count / total_req_count
        logging.info(
            "CLOCK: {} Cum ec2 vm_up_time_cost {} serverless fn cost: {} at episode end".format(self.worker.clock,
                                                                                                self.worker.ec2_vm_up_time_cost,
                                                                                                self.worker.serverless_fn_cost))
        logging.info("CLOCK: {} fn_failure_rate: {}".format(self.worker.clock, fn_failure_rate))

        if write_graph:
            print("GRAPHS")
            ep_reward_sheet.write(ep + 1, 0, ep)
            ep_reward_sheet.write(ep + 1, 1, self.worker.episodic_reward)
            ep_lat_sheet.write(ep + 1, 0, ep)
            ep_lat_sheet.write(ep + 1, 1, self.worker.function_latency)
            ep_failure_rate_sheet.write(ep + 1, 0, ep)
            ep_failure_rate_sheet.write(ep + 1, 1, self.worker.Episodic_failure_rate)
            ep_ec2_cost_sheet.write(ep + 1, 0, ep)
            ep_ec2_cost_sheet.write(ep + 1, 1, self.worker.ec2_vm_up_time_cost)
            ep_ec2_time_sheet.write(ep + 1, 0, ep)
            ep_ec2_time_sheet.write(ep + 1, 1, self.worker.ec2_vm_up_time)
            ep_serverless_cost_sheet.write(ep + 1, 0, ep)
            ep_serverless_cost_sheet.write(ep + 1, 1, self.worker.serverless_fn_cost)
            ep_t_cost_sheet.write(ep + 1, 0, ep)
            ep_t_cost_sheet.write(ep + 1, 1, self.worker.serverless_fn_cost + self.worker.ec2_vm_up_time_cost)
            ep_env_action_sheet.write(ep + 1, 0, ep)
            ep_env_action_sheet.write(ep + 1, 1, int(self.worker.deploy_env_action_total))
            ep_second_action_sheet.write(ep + 1, 0, ep)
            ep_second_action_sheet.write(ep + 1, 1, int(self.worker.second_action_total))
            # step_latency_sheet.write(ep + 1, 0, ep)
            # step_latency_sheet.write(ep + 1, 1, self.worker.function_latency)
            step_failure_rate_sheet.write(ep + 1, 0, ep)
            step_failure_rate_sheet.write(ep + 1, 1, self.worker.fn_failures)
            step_t_costdiff_sheet.write(ep + 1, 0, ep)
            step_t_costdiff_sheet.write(ep + 1, 1, self.worker.total_vm_cost_diff)

            wb_summary.save("summary.xls")
            # self.tensorboard.update_stats(Episodic_reward=self.worker.episodic_reward)
            # self.tensorboard.update_stats(Function_Latency_step_based=self.worker.function_latency)
            # self.tensorboard.update_stats(Function_failure_rate_total_step_based=self.worker.fn_failures)
            # self.tensorboard.update_stats(Total_VM_COST_DIFF=self.worker.total_vm_cost_diff)
            # # self.tensorboard.update_stats(Vertical_cpu_action_total=self.worker.ver_cpu_action_total)
            # # self.tensorboard.update_stats(Vertical_mem_action_total=self.worker.ver_mem_action_total)
            # # self.tensorboard.update_stats(Horizontal_action_total=self.worker.hor_action_total)
            # self.tensorboard.update_stats(Episodic_latency=self.worker.Episodic_latency)
            # self.tensorboard.update_stats(Episodic_failure_rate=self.worker.Episodic_failure_rate)
            # self.tensorboard.update_stats(Episodic_ec2_vm_cost=self.worker.ec2_vm_up_time_cost)
            # self.tensorboard.update_stats(Episodic_ec2_vm_uptime=self.worker.ec2_vm_up_time)
            # self.tensorboard.update_stats(Episodic_serverless_fn_cost=self.worker.serverless_fn_cost)
            # self.tensorboard.update_stats(Episodic_total_user_cost=self.worker.serverless_fn_cost + self.worker.ec2_vm_up_time_cost)

    # def reset(self):
    #     """
    #     Reset the state of the environment and returns an initial observation.
    #
    #     Returns
    #     -------
    #     observation (object): the initial observation of the space.
    #     """
    #     # if str(train_or_eval_flag) == "eval":
    #     #     self.rng = np.random.RandomState(constants.seed_random_number_evaluation)
    #
    #     if self.go_to_next_DAG == True:
    #         self.app_idx = self.rng.choice(
    #             np.arange(len(FogEnv_base.DAG_list)), 1, replace=False)
    #         print("the selected DAG id is: {}".format(int(self.app_idx)))
    #         self.go_to_next_DAG = False
    #         self.iteration_counter_for_one_DAG = 0
    #
    #     self.application = copy.deepcopy(
    #         FogEnv_base.DAG_list[int(self.app_idx)])
    #     self.task_idx = self.task_idx_selector(self.application)
    #     self.MAX_STEPS = len(FogEnv_base.DAG_list[int(self.app_idx)].list_prioritized_tasks) - len(
    #         FogEnv_base.DAG_list[int(self.app_idx)].list_local_execution_tasks)
    #     self.serverless_vms = copy.deepcopy(FogEnv_base.serverless_vms)
    #     self.state = copy.deepcopy(np.array(FogEnv_base.gen_FogEnv_state(
    #         self.task_idx, FogEnv_base.DAG_list[int(self.app_idx)], self.serverless_vms,
    #         FogEnv_base.DAG_list[int(self.app_idx)].dict_solutionConfig)))
    #     self.done = False
    #     self.reward = 0
    #     self.list_sorted_tag_tasks = FogEnv_base.DAG_list[int(
    #         self.app_idx)].list_prioritized_tasks
    #
    #     # self.jobs = copy.deepcopy(cluster.JOBS)
    #     self.clock = 0
    #     # self.job_idx = 0
    #
    #     self.task_list_queue = PriorityQueue()
    #     self.episode_success = False
    #     self.current_count = 1
    #     self.info = {}
    #     self.ave_task_time = 0
    #     self.episode_cost = 0
    #     self.average_task_duration = []
    #
    #     # self.greedyCostCalculator(self.application)
    #     self.start_task_local_cost_calculator(
    #         self.application.list_local_execution_tasks)
    #     # print(self.jobs[self.job_idx].ex_placed)
    #     return self.state

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    # def take_step(self, act_type, act, step, id_w):
    #
    #     # global episodes
    #     # logging.debug("Worker: {} Current ServEnv State: {}".format(id_w, self.state))
    #     # print("Current FogEnv State: {}".format(self.state))
    #     if self.done:
    #         self.reward = self.calculate_reward(step, id_w)
    #         return [self.reward, self.done, self.info]
    #         # # code should never reach this point
    #         # # if done the code shoudnt come to step method (because there is no action), instead we should directly calculate reward for thr previous step
    #         # print(
    #         #     "EPISODE DONE..code should not come here....going to reset the episode!!!")
    #         # # return self.reset()
    #         # self.reset()
    #
    #     # elif self.current_count == self.MAX_STEPS:
    #     #     self.done = True
    #     # if act_action > (self.num_max_replicas - 1) or action < -(self.num_max_replicas - 1):
    #     #     raise ValueError(
    #     #         "action should be between {} and {}, but the action is {}".format((self.num_max_replicas - 1),
    #     #                                                                           -(self.num_max_replicas - 1), action))
    #
    #     # elif self.count == self.MAX_STEPS:
    #     #    self.done = True
    #
    #     else:
    #         assert self.action_space.contains(act)
    #         # self.logging.info("CLOCK: {}: Action: {}".format(ServEnv_base.clock, act))
    #         # print("CLOCK: {}: Action: {}".format(self.clock, action))
    #         # self.current_count += 1
    #         # if valid placement, place 1 ex in the VM chosen, update cluster states -> "self._state";
    #         # check for episode end  -> update self.done
    #         self.execute_scaling(act, step, id_w)
    #
    #     # try:
    #     #     self.state = np.reshape(self.state, [1, self.state_size])
    #     #     assert self.observation_space.contains(np.array(self.state))
    #     # except AssertionError:
    #     #     print("INVALID STATE", np.array(self.state))
    #
    #     # return [self.state, self.reward, self.done, self.info]
    #     return [self.reward, self.done, self.info]

    def filtered_unavail_action_list(self, act_selected):
        unavail_action_list_serv = []
        unavail_action_list_nserv = []
        print("Action selected is {}".format(act_selected))
        if act_selected == 0:
            for vm in self.worker.serverless_vms:
                print("serverless: cpu left after allocation in vm {} is {}".format(vm.id,
                                                                                    str(vm.cpu - vm.cpu_allocated)))
                print("serverless: mem left after allocation in vm {} is {}".format(vm.id,
                                                                                    str(vm.ram - vm.mem_allocated)))

                if self.fn_type not in vm.idle_containers:
                    if ((vm.cpu - vm.cpu_allocated) < self.worker.fn_features[str(self.fn_type) + "_cpu_req"]) or (
                            (vm.ram - vm.mem_allocated) < self.worker.fn_features[str(self.fn_type) + "_req_ram"]):
                        unavail_action_list_serv.append(vm.id)
                        print("Serverless vm Action {} added to unavail list".format(vm.id))
            return unavail_action_list_serv
        else:
            for vm in self.worker.ec2_vms:
                print(
                    "serverful: cpu left after allocation in vm {} is {}".format(vm.id, str(vm.cpu - vm.cpu_allocated)))
                print("serverful: mem left after allocation in vm {} is {}".format(vm.id,
                                                                                   str(vm.ram - vm.mem_allocated)))
                if ((vm.cpu - vm.cpu_allocated) < self.worker.fn_features[str(self.fn_type) + "_cpu_req"]) or (
                        (vm.ram - vm.mem_allocated) < self.worker.fn_features[str(self.fn_type) + "_req_ram"]):
                    unavail_action_list_nserv.append(vm.id)
                    print("Serverful vm Action {} added to unavail list".format(vm.id))
            return unavail_action_list_nserv

    # def filtered_unavail_action_list(self, fn_type):
    #     global action_cpu
    #     global action_mem
    #     global action_util
    #     unavail_action_list_cpu = []
    #     unavail_action_list_mem = []
    #     # unavail_action_list_rep = []
    #     unavail_action_list_util = []
    #     # global max_replicas
    #     # global vertical_actions
    #     # with ServEnv_base.vm_lock:
    #     for vm in self.worker.serverless_vms:
    #         cpu_added = 0
    #         pod_count = 0
    #         fntype_pod_cpu_req_total = 0
    #         fntype_pod_mem_req_total = 0
    #         pod_cpu_util_min = 0
    #         pod_ram_util_min = 0
    #         if fn_type in vm.running_list:
    #             for pod in vm.running_list[fn_type]:
    #                 pod_count += 1
    #                 fntype_pod_cpu_req_total += pod.cpu_req
    #                 fntype_pod_mem_req_total += pod.ram_req
    #                 if pod.cpu_util > pod_cpu_util_min:
    #                     pod_cpu_util_min = pod.cpu_util
    #                 if pod.ram_util > pod_ram_util_min:
    #                     pod_ram_util_min = pod.ram_util
    #         if pod_count * self.worker.fn_features[str(fn_type) + "_pod_cpu_req"] > vm.cpu_allocated:
    #             print("debug at unav action list")
    #         for a in range(len(action_cpu)):
    #             if a not in unavail_action_list_cpu:
    #                 if a == 5:
    #                     logging.info(
    #                         "CLOCK: {} Worker: {} fn_type: {} Pod count: {} vm.cpu allocated: {} vm.cpu: {} fn_pod_cpu_req: {} pod_cpu_util_min: {} pod_ram_util_min: {} fn_pod_ram_req: {} fn_scale_cpu_threshold: {}".format(self.worker.clock,
    #                             self.worker_id, fn_type, pod_count, vm.cpu_allocated, vm.cpu, self.worker.fn_features[
    #                                 str(fn_type) + "_pod_cpu_req"], pod_cpu_util_min, pod_ram_util_min,
    #                             self.worker.fn_features[str(fn_type) + "_pod_ram_req"],
    #                             self.worker.fn_features[str(fn_type) + "_scale_cpu_threshold"]))
    #                 if action_cpu[a] * pod_count + vm.cpu_allocated > vm.cpu or self.worker.fn_features[
    #                     str(fn_type) + "_pod_cpu_req"] + action_cpu[a] > constants.max_pod_cpu_req or \
    #                         action_cpu[a] * pod_count + vm.cpu_allocated < 0 or self.worker.fn_features[
    #                     str(fn_type) + "_pod_cpu_req"] + action_cpu[a] < constants.min_pod_cpu_req or \
    #                         self.worker.fn_features[
    #                             str(fn_type) + "_pod_cpu_req"] + action_cpu[a] < pod_cpu_util_min:
    #                     unavail_action_list_cpu.append(a)
    #                     # logging.info("Worker: {} Adding cpu Action: {} to unavail list".format(self.worker_id, a))
    #
    #         for b in range(len(action_mem)):
    #             if b not in unavail_action_list_mem:
    #                 if b == 5:
    #                     logging.info(
    #                         "Worker: {} Pod count: {} vm.mem allocated: {} vm.ram: {} fn_pod_cpu_req: {} fn_pod_ram_req: {} fn_scale_cpu_threshold: {}".format(
    #                             self.worker_id, pod_count, vm.mem_allocated, vm.ram, self.worker.fn_features[
    #                                 str(fn_type) + "_pod_cpu_req"],
    #                             self.worker.fn_features[str(fn_type) + "_pod_ram_req"],
    #                             self.worker.fn_features[str(fn_type) + "_scale_cpu_threshold"]))
    #                 if action_mem[b] * pod_count + vm.mem_allocated > vm.ram or action_mem[
    #                     b] * pod_count + vm.mem_allocated < 0 or self.worker.fn_features[
    #                     str(fn_type) + "_pod_ram_req"] + math.floor(
    #                         action_mem[b]) > constants.max_total_pod_mem or self.worker.fn_features[
    #                     str(fn_type) + "_pod_ram_req"] + math.floor(
    #                     action_mem[b]) < constants.min_pod_mem_req or self.worker.fn_features[
    #                     str(fn_type) + "_pod_ram_req"] + math.floor(
    #                     action_mem[b]) < pod_ram_util_min:
    #                     unavail_action_list_mem.append(b)
    #                     # logging.info("Worker: {} Adding mem Action: {} to unavail list".format(self.worker_id, b))
    #
    #         # for c in range(len(action_rep)):
    #         #     if c not in unavail_action_list_rep:
    #         #         # if c == 10:
    #         #         #     # logging.info(
    #         #         #     #     "Pod count: {} vm.cpu allocated: {} vm.cpu: {} fn0_pod_cpu_req: {} fn0_pod_ram_req: {} fn0_num_max_replicas: {}".format(
    #         #         #     #         pod_count, vm.cpu_allocated, vm.cpu, self.worker.fn_features[
    #         #         #     #             str(self.fn_type) + "_pod_cpu_req"],
    #         #         #     #         self.worker.fn_features[str(self.fn_type) + "_pod_ram_req"],
    #         #         #     #         self.worker.fn_features[str(self.fn_type) + "_num_max_replicas"]))
    #         #         if self.worker.fn_features[str(self.fn_type) + "_num_max_replicas"] + action_rep[c] < 1 or \
    #         #                 self.worker.fn_features[str(self.fn_type) + "_num_max_replicas"] + action_rep[c] > constants.max_num_replicas:
    #         #             unavail_action_list_rep.append(c)
    #         #             # logging.info("Worker: {} Adding rep Action: {} to unavail list".format(self.worker_id, c))
    #
    #         for c in range(len(action_util)):
    #             if c not in unavail_action_list_util:
    #                 # if c == 5:
    #                 #     # logging.info(
    #                 #     #     "Pod count: {} vm.cpu allocated: {} vm.cpu: {} fn0_pod_cpu_req: {} fn0_pod_ram_req: {} fn0_num_max_replicas: {}".format(
    #                 #     #         pod_count, vm.cpu_allocated, vm.cpu, self.worker.fn_features[
    #                 #     #             str(self.fn_type) + "_pod_cpu_req"],
    #                 #     #         self.worker.fn_features[str(self.fn_type) + "_pod_ram_req"],
    #                 #     #         self.worker.fn_features[str(self.fn_type) + "_num_max_replicas"]))
    #                 if self.worker.fn_features[str(fn_type) + "_scale_cpu_threshold"] + action_util[c] > constants.pod_scale_cpu_util_high or \
    #                         self.worker.fn_features[str(fn_type) + "_scale_cpu_threshold"] + action_util[c] < constants.pod_scale_cpu_util_low:
    #                     unavail_action_list_util.append(c)
    #
    #     return unavail_action_list_cpu, unavail_action_list_mem, unavail_action_list_util

    # def act(self, step, w_id, fn):
    #     print("Action selection")
    #     sel_action = []
    #     # with ServEnv_base.vm_lock:
    #     discarded_action_list_c, discarded_action_list_m, discarded_action_list_r = self.filtered_unavail_action_list(fn)
    #
    #     # logits_c, logits_m, logits_r, v = self.local_model.call(
    #     #     tf.convert_to_tensor(self.state[None, :], dtype=tf.float32))
    #
    #     logits_action, v = self.local_model.call(
    #         tf.convert_to_tensor(self.state[None, :], dtype=tf.float32))
    #
    #     # print(logits_action)
    #
    #     # logits_c = []
    #     # logits_m = []
    #     # logits_r = []
    #
    #     logits_c, logits_m, logits_r = tf.split(logits_action, [self.action_size[0], self.action_size[1], self.action_size[2]], 2)
    #     # print(logits_c)
    #     # print(logits_m)
    #     # print(logits_r)
    #
    #     # range1 = self.action_size[0]
    #     # range2 = self.action_size[0] + self.action_size[1]
    #     # range3 = self.action_size[0] + self.action_size[1] + self.action_size[2]
    #     # for x in range(0, range1):
    #     #     logits_c.append(logits_action[0][x])
    #     #
    #     # for x in range(range1, range2):
    #     #     logits_m.append(logits_action[0][x])
    #     #
    #     # for x in range(range2, range3):
    #     #     logits_r.append(logits_action[0][x])
    #
    #     probs_c = tf.nn.softmax(logits_c)
    #     probs_m = tf.nn.softmax(logits_m)
    #     probs_r = tf.nn.softmax(logits_r)
    #
    #     # print("prob"+ str(probs_c.numpy()[0]))
    #
    #     probs_c = probs_c.numpy()[0].flatten()
    #     probs_m = probs_m.numpy()[0].flatten()
    #     probs_r = probs_r.numpy()[0].flatten()
    #
    #     # set the probabilities of all illegal moves to zero and renormalise the output vector before we choose our move.
    #
    #     # print("prob_c before " + str(probs_c))
    #     # print("prob_m before " + str(probs_m))
    #     # print("prob_r before " + str(probs_r))
    #
    #     for a in discarded_action_list_c:
    #         probs_c[a] = 0
    #     for a in discarded_action_list_m:
    #         probs_m[a] = 0
    #     for a in discarded_action_list_r:
    #         probs_r[a] = 0
    #
    #     # print("prob_c after " + str(probs_c))
    #     # print("prob_m after " + str(probs_m))
    #     # print("prob_r after " + str(probs_r))
    #
    #     probs_c /= np.array(probs_c).sum()
    #     probs_m /= np.array(probs_m).sum()
    #     probs_r /= np.array(probs_r).sum()
    #
    #     # print("Prob"+str(probs_c.numpy()[0]))
    #     action_c = np.random.choice(self.action_size[0], p=probs_c)
    #     # print("Discarded actions for thread %s: %s" % (str(step-1), str(discarded_action_list)))
    #
    #     # print("Prob"+str(probs_c.numpy()[0]))
    #     action_m = np.random.choice(self.action_size[1], p=probs_m)
    #
    #     # print("Prob"+str(probs_c.numpy()[0]))
    #     action_r = np.random.choice(self.action_size[2], p=probs_r)
    #
    #     action_t = "Network"
    #     sel_action = [action_c, action_m, action_r]
    #     print("worker: %s Selected action for step %s: %s" % (str(self.worker_id), step, str(sel_action)))
    #     # sch_data.write(step, 9, "Random")
    #     # wb.save("Episodic_Data" + str(ep) + ".csv")
    #     # print("Random action selected: ", sel_action)
    #     self.worker.ver_cpu_action_total += action_cpu[sel_action[0]]
    #     self.worker.ver_mem_action_total += action_mem[sel_action[1]]
    #     self.worker.hor_action_total += action_util[sel_action[2]]
    #
    #     self.execute_scaling(sel_action, step, w_id, fn)
    #     self.worker.pod_scaler()
    #
    #     return sel_action, action_t, self.done

    def act_test(self, step, state, app):
        sel_action = np.argmax(self.main_network.predict(state))
        action_t = "Network"
        self.worker.ver_cpu_action_total += action_cpu[sel_action[0]]
        self.worker.ver_mem_action_total += action_mem[sel_action[1]]
        self.worker.hor_action_total += action_util[sel_action[2]]
        return sel_action, action_t

    def get_state(self, step_c):
        state_orig = self.worker.gen_serv_env_state()
        self.state = np.reshape(state_orig, [1, self.state_size])
        # logging.debug("Worker: {}  Current ServEnv State: {}".format(self.worker_id, self.state))
        return state_orig, self.state, self.worker.clock

    def select_action(self, step_c):
        # logging.info("CLOCK: {} worker: {} Generating state for step: {}".format(
        #     self.clock, self.worker_id, step_c))
        state = self.worker.gen_serv_env_state()
        state = np.reshape(state, [1, self.state_size])
        # logging.debug("worker: {} Current ServEnv State: {}".format( self.worker_id, state))
        # self.action = self.action_space.sample()
        self.action, self.act_type = self.act_test(step_c, state, self.fn_type)
        # self.actual_action = self.action - (constants.max_num_replicas - 1)
        return state, self.act_type, self.action, self.worker.clock

    def select_action_test(self, step_c, ep):
        # logging.info("CLOCK: {} worker: {} Generating state for step: {}".format(
        #     self.clock, self.worker_id, step_c))
        state = self.worker.gen_serv_env_state()
        state = np.reshape(state, [1, self.state_size])
        # logging.debug("worker: {} Current ServEnv State: {}".format( self.worker_id, state))
        # self.action = self.action_space.sample()
        self.action, self.act_type = self.act_test(step_c, state, self.fn_type, ep)
        # self.actual_action = self.action - (constants.max_num_replicas - 1)
        return state, self.act_type, self.action, self.worker.clock

    # def execute_scaling(self, action, step_c, idx, fn_type):
    #     # print("DOne status: {}".format(self.done))
    #     # self.reward = self.calculate_reward(step_c, idx)
    #     # logging.info(
    #     #     "CLOCK: {} Worker:  {} Now executing scaling, type: {}, previous max replicas: {}".format(self.worker.clock,
    #     #                                                                                               idx,
    #     #                                                                                               self.fn_type,
    #     #                                                                                               self.worker.fn_features[
    #     #                                                                                                   str(
    #     #                                                                                                       self.fn_type) + "_num_max_replicas"]))
    #
    #     # if (ServEnv_base.fn_features[str(self.fn_type) + "_num_max_replicas"] + action) >= 1:
    #     self.worker.fn_features[str(fn_type) + "_scale_cpu_threshold"] += action_util[action[2]]
    #     new_cpu = self.worker.fn_features[str(fn_type) + "_pod_cpu_req"] + action_cpu[action[0]]
    #     self.worker.fn_features[str(fn_type) + "_pod_cpu_req"] = new_cpu
    #     new_mem = self.worker.fn_features[str(fn_type) + "_pod_ram_req"] + action_mem[action[1]]
    #     self.worker.fn_features[str(fn_type) + "_pod_ram_req"] = new_mem
    #
    #     # with ServEnv_base.vm_lock:
    #     if fn_type in self.worker.cont_object_dict_by_type:
    #         for pod in self.worker.cont_object_dict_by_type[fn_type]:
    #             pod.allocated_vm.cpu_allocated += new_cpu - pod.cpu_req
    #             pod.cpu_req = new_cpu
    #             pod.allocated_vm.mem_allocated += new_mem - pod.ram_req
    #             pod.ram_req = new_mem
    #
    #     # logging.info("Worker: {} Now max replicas: {}".format(idx, self.worker.fn_features[str(self.fn_type) + "_num_max_replicas"]))
    #     # logging.info("Now cpu req: {}".format(self.worker.fn_features[str(self.fn_type) + "_pod_cpu_req"]))
    #     # logging.info("Now mem req: {}".format(self.worker.fn_features[str(self.fn_type) + "_pod_ram_req"]))
    #     # self.worker.pod_scaler()
    #
    #     # self.reward = 10
    #     # logging.info("CLOCK: {}  Worker:  {} Finished scaling step: {}".format(
    #     #     self.worker.clock, idx, self.current_count))
    #
    #     # Allocate the current reward to the previous step
    #     tuple0 = (self.current_count - 1, self.reward)
    #     self.info["step_reward"] = tuple0
    #
    #     # logging.info("Worker: {} Current count: {} Max steps: {}".format(idx, step_c, self.MAX_STEPS))
    #     # self.current_count += 1
    #
    #     if step_c == self.MAX_STEPS:
    #         self.done = True
    #         self.episode_success = True
    #         # return True
    #
    #     # self.state = ServEnv_base.gen_serv_env_state()
    #     # self.state = ServEnv_base.gen_serv_env_state(self.fn_type)
    #     # self.logging.debug("CLOCK: {}: Current ServEnv State: {}".format(ServEnv_base.clock, self.state))
    #     return True

    # def create_scaling_events(self):
    #     scaling_time = constants.step_interval
    #     for i in range(self.MAX_STEPS):
    #         # create events for each scaling step
    #         # self.logging.info("Scaling event created at: {}".format(scaling_time))
    #         self.worker.sorted_events.append(defs.EVENT(scaling_time, constants.invoke_step_scaling, None))
    #         scaling_time += constants.step_interval
    #
    #     self.worker.sorted_events = sorted(self.worker.sorted_events, key=self.worker.sorter_events)
    #     # print(ServEnv_base.sorted_events)

    def execute_events(self):
        while self.worker.sorted_events:
            # print(len(ServEnv_base.sorted_events))
            ev = self.worker.sorted_events.pop(0)
            prev_clock = self.worker.clock
            self.worker.clock = round(float(ev.received_time), 2)
            # print("Now time: " + str(ServEnv_base.clock))
            time_diff = self.worker.clock - prev_clock
            # pod_scaler()
            ev_name = ev.event_name
            # print("Event is " + ev_name)
            if ev_name == "SCHEDULE_REQ":
                self.fn_type = ev.entity_object.type
                self.worker.fn_type = ev.entity_object.type
                self.current_request = ev.entity_object
                # set the current arrival rate for a fn
                self.worker.fn_request_rate[ev.entity_object.type] = ev.entity_object.arrival_rate
                return True
                # second parameter specifies if a request is a new one or a rescheduling request
                # self.worker.req_scheduler(ev.entity_object)
            elif ev_name == "RE_SCHEDULE_REQ":
                self.worker.req_scheduler(ev.entity_object)
            elif ev_name == "REQ_COMPLETE":
                # if ServEnv_base.clock == 162.73:
                # print("Debug")
                self.worker.req_completion(ev.entity_object)
            elif ev_name == "CREATE_CONT":
                # print(ev.entity_object.type)
                self.worker.container_creator(ev.entity_object)
            elif ev_name == "CREATE_VM":
                # print(ev.entity_object.type)
                self.worker.create_ec2_vms(ev.entity_object)
            elif ev_name == "KILL_CONT":
                # print(ev.entity_object.type)
                self.worker.container_terminator(ev.entity_object)
            elif ev_name == "KILL_VM":
                # print(ev.entity_object.type)
                self.worker.terminate_ec2_vm(ev.entity_object)
            # elif ev_name == "SCALE_POD":
            #     # print(ev.entity_object)
            #     self.worker.container_creator(ev.entity_object)
            # elif ev_name == "SCHEDULE_POD":
            #     self.worker.pod_scheduler(ev.entity_object)
            # elif ev_name == "STEP_SCALING":
            #     if self.worker.fn_iterator < (len(self.worker.fn_types) - 1):
            #         self.worker.fn_iterator += 1
            #     else:
            #         self.worker.fn_iterator = 0
            #     # if ServEnv_base.clock == 160:
            #     # print("Debug")
            #     return True

        self.simulation_running = False
        return True
        # for req in dropped_requests:
        #     req_wl.write(results_sheet_row_counter, 0, req.id)
        #     req_wl.write(results_sheet_row_counter, 1, req.type)
        #     req_wl.write(results_sheet_row_counter, 2, "None")
        #     req_wl.write(results_sheet_row_counter, 3, req.arrival_time)
        #     req_wl.write(results_sheet_row_counter, 4, req.start_time)
        #     req_wl.write(results_sheet_row_counter, 5, "None")
        #     req_wl.write(results_sheet_row_counter, 6, "Dropped")
        #
        #     results_sheet_row_counter += 1
        #
        # wb.save("D:\WL generation\Third_work\Results.xls")

    def calc_total_user_cost(self):
        vm_cost = 0
        vm_time = 0
        ser_fn_cost = 0
        for vm in self.worker.ec2_vm_up_time_dict:
            if self.worker.ec2_vm_up_time_dict[vm]['status'] == "ON":
                self.worker.ec2_vm_up_time_dict[vm]['total_time'] += self.worker.clock - \
                                                                     self.worker.ec2_vm_up_time_dict[vm]['time_now']
                self.worker.ec2_vm_up_time_dict[vm]['time_now'] = self.worker.clock
            vm_time += self.worker.ec2_vm_up_time_dict[vm]['total_time']
            vm_cost += vm.price * self.worker.ec2_vm_up_time_dict[vm]['total_time']

        # print(self.worker.serverless_cost_dict)
        for fn_type in self.worker.serverless_cost_dict:
            # print(int(self.worker.serverless_cost_dict[fn_type]['exec_time']))
            ser_fn_cost += self.worker.serverless_cost_dict[fn_type]['mem'] * self.worker.serverless_cost_dict[fn_type][
                'exec_time'] * constants.serverless_mbs_price
        ser_fn_cost += self.worker.serverless_request_no * constants.serverless_price_per_request
        return vm_cost/3600, vm_time, ser_fn_cost

    def execute_action(self, action1, action2):
        self.current_request.vm_id = action2
        self.current_request.act = [action1, action2]
        self.worker.action = [action1, action2]
        if action1 == 0:
            self.current_request.allocated_vm = self.worker.serverless_vms[action2]
            self.current_request.deployed_env = 's'

            if self.current_request.type in self.current_request.allocated_vm.idle_containers:
                self.current_request.allocated_cont = self.current_request.allocated_vm.idle_containers[self.current_request.type].pop(0)
                if len(self.current_request.allocated_vm.idle_containers[self.current_request.type]) == 0:
                    self.current_request.allocated_vm.idle_containers.pop(self.current_request.type)
                print("clock: {} Removing cont {} from idle list".format(self.worker.clock, self.current_request.allocated_cont.id))
                self.current_request.allocated_cont.running_request = self.current_request
                self.current_request.allocated_cont.idle_start_time = 0
            else:
                self.current_request.allocated_vm.cpu_allocated += int(
                    self.worker.fn_features[str(self.current_request.type) + "_cpu_req"])
                self.current_request.allocated_vm.mem_allocated += int(
                    self.worker.fn_features[str(self.current_request.type) + "_req_ram"])
        else:
            self.current_request.deployed_env = 'ns'
            self.worker.ec2_vms[action2].cpu_allocated += int(
                self.worker.fn_features[str(self.current_request.type) + "_cpu_req"])
            self.worker.ec2_vms[action2].mem_allocated += int(
                self.worker.fn_features[str(self.current_request.type) + "_req_ram"])
            if self.worker.ec2_vms[action2] in self.worker.ec2_vm_up_time_dict:
                if self.worker.ec2_vm_up_time_dict[self.worker.ec2_vms[action2]]['status'] == "ON":
                    self.current_request.allocated_vm = self.worker.ec2_vms[action2]

        self.worker.req_scheduler(self.current_request)

    def calculate_reward(self, step_c, ac1, ac2):
        req_info = {}
        latency = 0
        fn_failure_rate = 0
        failed_count = 0
        total_req_count_within_window = 0
        reward = 0
        vm_cost_step = 0

        logging.info("CLOCK: {} Calculating reward for previous step {}".format(self.worker.clock, step_c - 1))
        if self.worker.fn_type in self.worker.sorted_request_history_per_window:
            for req in reversed(self.worker.sorted_request_history_per_window[self.worker.fn_type]):
                if req.finish_time > self.worker.clock - constants.reward_window_size:
                    # print(req.id)
                    # self.logging.info(
                    #     "Req id: {}, arrival time: {}, finish time: {}".format(req.id, req.arrival_time, req.finish_time))
                    total_req_count_within_window += 1
                    # if req.status != "Dropped":
                    #     if req.type in req_info:
                    #         # self.logging.info("Req id: {}, arrival time: {}, finish time: {}".format(req.id, req.arrival_time,
                    #         #                                                                     req.finish_time))
                    #         req_info[req.type]['execution_time'] += req.finish_time - req.arrival_time
                    #         req_info[req.type]['req_count'] += 1
                    #     else:
                    #         req_info[req.type] = {}
                    #         # self.logging.info("Req id: {}, arrival time: {}, finish time: {}".format(req.id, req.arrival_time,
                    #         #                                                                     req.finish_time))
                    #         req_info[req.type]['execution_time'] = req.finish_time - req.arrival_time
                    #         req_info[req.type]['req_count'] = 1
                    # else:
                    #     failed_count += 1
                    if req.status == "Dropped":
                        failed_count += 1
                else:
                    break

        # for req_type, req_data in req_info.items():
        #     # self.logging.info("FN type: {}, Total exec time: {}, Req count: {}, MIPS for type: {}".format(req_type, req_data[
        #     #     'execution_time'], req_data['req_count'], ServEnv_base.fn_features[str(req_type) + "_req_MIPS"]))
        #     fn_latency += (req_data['execution_time'] / req_data['req_count']) / (
        #             int(self.worker.fn_features[str(req_type) + "_req_MIPS"]) / self.worker.fn_features[
        #         str(req_type) + "_cpu_req"])
        #
        # if len(req_info) != 0:
        #     fn_latency = fn_latency / len(req_info) / constants.max_step_latency_perfn
        # logging.info("CLOCK: {} step latency: {}".format(self.worker.clock, fn_latency))
        self.worker.ec2_vm_up_time_cost, self.worker.ec2_vm_up_time, self.worker.serverless_fn_cost = self.calc_total_user_cost()
        if total_req_count_within_window != 0:
            fn_failure_rate = failed_count / total_req_count_within_window
        logging.info("CLOCK: {} Cum ec2 vm_up_time_cost: {} Cum serverless_fn_cost: {}".format(self.worker.clock,
                                                                                               self.worker.ec2_vm_up_time_cost,
                                                                                               self.worker.serverless_fn_cost))
        logging.info("CLOCK: {} fn_failure_rate: {}".format(self.worker.clock, fn_failure_rate))

        self.worker.req_latency_prev = self.worker.req_latency
        env_available = False
        print("Actions are: {} and {}:".format(ac1, ac2))
        if ac1 == 0:
            if self.current_request.type in self.worker.serverless_vms[ac2].idle_containers:
                self.worker.req_latency = 0
                print("Cont avail so latency is: {} ".format(self.worker.req_latency))

            else:
                self.worker.req_latency = constants.container_creation_time/constants.vm_creation_time
                print("Cont not avail so latency is: {} ".format(self.worker.req_latency))
        else:
            if self.worker.ec2_vms[ac2] in self.worker.ec2_vm_up_time_dict:
                if self.worker.ec2_vm_up_time_dict[self.worker.ec2_vms[ac2]]['status'] == "ON":
                    env_available = True
                    self.worker.req_latency = 0
                    print("Ec2 vm is on so latency is: {} ".format(self.worker.req_latency))
            if not env_available:
                if self.worker.ec2_vms[ac2] in self.worker.pending_vms:
                    self.worker.req_latency = (constants.vm_creation_time - (self.worker.clock - self.worker.ec2_vms[ac2].launch_start_time))/constants.vm_creation_time
                    print("Ec2 vm is pending so latency is: {} ".format(self.worker.req_latency))
                else:
                    self.worker.req_latency = constants.vm_creation_time/constants.vm_creation_time
                    print("Ec2 vm is off so latency is: {} ".format(self.worker.req_latency))

        if step_c > 1:
            self.worker.fn_failures += fn_failure_rate
            self.worker.function_latency += self.worker.req_latency_prev

            # global total_vm_time_diff
            # total_vm_time_diff += reward_vm_up_time
            vm_cost_step = (self.worker.ec2_vm_up_time_cost + self.worker.serverless_fn_cost) / constants.max_step_vmcost - self.worker.vm_up_time_cost_prev
            self.worker.vm_up_time_cost_prev = (self.worker.ec2_vm_up_time_cost + self.worker.serverless_fn_cost) / constants.max_step_vmcost
            self.worker.total_vm_cost_diff += vm_cost_step
            logging.info("CLOCK: {}  vm_up_time_cost diff for step: {}".format(self.worker.clock, vm_cost_step))

            x = 0
            y = 1 - x
            # reward = - (reward_vm_up_time / 200 * x + reward_fn_latency / 30 * y)
            # vm cost multiplying by 100 to make that value as significant as the latency value
            reward = - (vm_cost_step * x + (self.worker.req_latency_prev + fn_failure_rate) * y)
            logging.info("CLOCK: {} Step reward: {}".format(self.worker.clock, reward))

            self.worker.episodic_reward += reward
        else:
            reward = 0

        # tuple = [fn_latency, vm_cost_step]
        return reward

    def render(self, mode="human"):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
        """
        # s = "state: {}  reward: {}  info: {}"
        # print(s.format(self.state, self.reward, self.info))

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

# a = ServerlessEnv()
