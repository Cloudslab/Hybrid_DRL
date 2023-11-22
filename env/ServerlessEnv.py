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

action_cpu = [-920, -690, -460, -230, -115, 0, 115, 230, 460, 690, 920]

action_mem = [-375, -300, -225, -150, -75, 0, 75, 150, 225, 300, 375]

# 43
action_rep = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

action_util = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
action_env = [0, 1]

wb_summary = Workbook()
ep_reward_sheet = wb_summary.add_sheet('Episodic_reward')
ep_lat_sheet = wb_summary.add_sheet('Episodic_latency')
ep_t_cost_sheet = wb_summary.add_sheet('Episodic_total_user_cost')
ep_serverless_cost_sheet = wb_summary.add_sheet('Episodic_serverless_fn_cost')
ep_ec2_cost_sheet = wb_summary.add_sheet('Episodic_ec2_vm_cost')
ep_ec2_time_sheet = wb_summary.add_sheet('Episodic_ec2_vm_uptime')
ep_env_action_sheet = wb_summary.add_sheet('Deploy_env')
ep_second_action_sheet = wb_summary.add_sheet('Second_action')
step_latency_sheet = wb_summary.add_sheet('Fn_Latency_step_based')
step_failure_rate_sheet = wb_summary.add_sheet('Fn_fail_rate_step_based')
step_t_costdiff_sheet = wb_summary.add_sheet('total_vm_cost_diff')

ep_reward_sheet_req = wb_summary.add_sheet('Episodic_reward_req')
ep_t_cost_sheet_req = wb_summary.add_sheet('Episodic_total_user_cost_req')
ep_ec2_cost_sheet_req = wb_summary.add_sheet('Episodic_ec2_vm_cost_req')
ep_serverless_cost_sheet_req = wb_summary.add_sheet('Episodic_serverless_fn_cost_req')
ep_env_action_sheet_req = wb_summary.add_sheet('Deploy_env_req')


episodic_reward_summary = 0
Episodic_latency_summary = 0
Episodic_failure_rate_summary = 0
ec2_vm_up_time_cost_summary = 0
ec2_vm_up_time_summary = 0
serverless_fn_cost_summary = 0
total_cost_summary = 0
deploy_env_action_total_summary = 0
second_action_total_summary = 0
function_latency_summary = 0
fn_failures_summary = 0
total_step_cost_summary = 0


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


class ServerlessEnv(gym.Env):
    """A serverless environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.fn_type = ""
        self.current_request = None
        self.episode_no = 0
        # self.env_base = ServEnv_base(self.fn_type, self.episode_no)
        # logging.basicConfig(filename="log/" + self.fn_type + "-logfile.log", filemode="w",
        #                     level=logging.DEBUG)

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
        self.done = False
        self.reward = 0
        self.clock = self.worker.clock
        self.episode_success = False
        self.current_count = 1
        self.info = {}

    def load(self, name):
        self.main_network = load_model(name)

    def summary_graphs(self, write_graph):
        total_req_count = 0
        failed_count = 0
        fn_failure_rate = 0
        req_info = {}
        fn_latency = 0

        global episodic_reward_summary
        global Episodic_latency_summary
        global Episodic_failure_rate_summary
        global ec2_vm_up_time_cost_summary
        global ec2_vm_up_time_summary
        global serverless_fn_cost_summary
        global total_cost_summary
        global deploy_env_action_total_summary
        global second_action_total_summary
        global function_latency_summary
        global fn_failures_summary
        global total_step_cost_summary

        for fn_type in self.worker.sorted_request_history_per_window:
            for req in self.worker.sorted_request_history_per_window[fn_type]:
                total_req_count += 1
                if req.status != "Dropped":
                    if req.type in req_info:
                        req_info[req.type]['execution_time'] += req.finish_time - req.arrival_time
                        req_info[req.type]['req_count'] += 1
                    else:
                        req_info[req.type] = {}
                        req_info[req.type]['execution_time'] = req.finish_time - req.arrival_time
                        req_info[req.type]['req_count'] = 1
                else:
                    failed_count += 1

        for req_type, req_data in req_info.items():
            logging.info("CLOCK: {} Avg execution_time: {}".format(self.worker.clock, round(
                (req_data['execution_time'] / req_data['req_count']), 4)))
            logging.info("CLOCK: {} std execution_time: {}".format(self.worker.clock, round((
                    int(self.worker.fn_features[str(req_type) + "_req_MIPS"]) / self.worker.fn_features[
                str(req_type) + "_cpu_req"]), 4)))
            fn_latency += round((req_data['execution_time'] / req_data['req_count']), 4) / round((
                    int(self.worker.fn_features[str(req_type) + "_req_MIPS"]) / self.worker.fn_features[
                str(req_type) + "_cpu_req"]), 4)

        if len(req_info) != 0:
            self.worker.Episodic_latency = fn_latency / len(req_info)
        logging.info("CLOCK: {} Overall latency: {}".format(self.worker.clock, fn_latency))
        self.worker.ec2_vm_up_time_cost, self.worker.ec2_vm_up_time, self.worker.serverless_fn_cost = self.calc_total_user_cost()
        logging.info(
            "CLOCK: {} Cum ec2 vm_up_time_cost {} serverless fn cost: {} at episode end".format(self.worker.clock,
                                                                                                self.worker.ec2_vm_up_time_cost,
                                                                                                self.worker.serverless_fn_cost))

        episodic_reward_summary += self.worker.episodic_reward
        Episodic_latency_summary += self.worker.Episodic_latency
        ec2_vm_up_time_cost_summary += self.worker.ec2_vm_up_time_cost
        ec2_vm_up_time_summary += self.worker.ec2_vm_up_time
        serverless_fn_cost_summary += self.worker.serverless_fn_cost
        total_cost_summary += self.worker.serverless_fn_cost + self.worker.ec2_vm_up_time_cost
        deploy_env_action_total_summary += int(self.worker.deploy_env_action_total)
        second_action_total_summary += int(self.worker.second_action_total)
        function_latency_summary += self.worker.function_latency
        fn_failures_summary += self.worker.fn_failures
        total_step_cost_summary += self.worker.total_step_cost

        ep = self.worker.summary_episode

        if write_graph:
            print("GRAPHS")
            ep_reward_sheet.write(ep + 1, 0, ep)
            ep_reward_sheet.write(ep + 1, 1, episodic_reward_summary)
            ep_lat_sheet.write(ep + 1, 0, ep)
            ep_lat_sheet.write(ep + 1, 1, Episodic_latency_summary)
            # ep_failure_rate_sheet.write(ep + 1, 0, ep)
            # ep_failure_rate_sheet.write(ep + 1, 1, self.worker.Episodic_failure_rate)
            ep_ec2_cost_sheet.write(ep + 1, 0, ep)
            ep_ec2_cost_sheet.write(ep + 1, 1, ec2_vm_up_time_cost_summary)
            ep_ec2_time_sheet.write(ep + 1, 0, ep)
            ep_ec2_time_sheet.write(ep + 1, 1,  ec2_vm_up_time_summary)
            ep_serverless_cost_sheet.write(ep + 1, 0, ep)
            ep_serverless_cost_sheet.write(ep + 1, 1, serverless_fn_cost_summary)
            ep_t_cost_sheet.write(ep + 1, 0, ep)
            ep_t_cost_sheet.write(ep + 1, 1, total_cost_summary)
            ep_env_action_sheet.write(ep + 1, 0, ep)
            ep_env_action_sheet.write(ep + 1, 1, int(deploy_env_action_total_summary))
            ep_second_action_sheet.write(ep + 1, 0, ep)
            ep_second_action_sheet.write(ep + 1, 1, int(second_action_total_summary))
            step_latency_sheet.write(ep + 1, 0, ep)
            step_latency_sheet.write(ep + 1, 1, function_latency_summary)
            step_failure_rate_sheet.write(ep + 1, 0, ep)
            step_failure_rate_sheet.write(ep + 1, 1, fn_failures_summary)
            step_t_costdiff_sheet.write(ep + 1, 0, ep)
            step_t_costdiff_sheet.write(ep + 1, 1, total_step_cost_summary)

            wb_summary.save("summary.xls")

            episodic_reward_summary = 0
            Episodic_latency_summary = 0
            Episodic_failure_rate_summary = 0
            ec2_vm_up_time_cost_summary = 0
            ec2_vm_up_time_summary = 0
            serverless_fn_cost_summary = 0
            total_cost_summary = 0
            deploy_env_action_total_summary = 0
            second_action_total_summary = 0
            function_latency_summary = 0
            fn_failures_summary = 0
            total_step_cost_summary = 0

    def graphs(self, write_graph, ep):
        total_req_count = 0
        failed_count = 0
        fn_failure_rate = 0
        req_info = {}
        fn_latency = 0

        for fn_type in self.worker.sorted_request_history_per_window:
            for req in self.worker.sorted_request_history_per_window[fn_type]:
                total_req_count += 1
                if req.status != "Dropped":
                    if req.type in req_info:
                        req_info[req.type]['execution_time'] += req.finish_time - req.arrival_time
                        req_info[req.type]['req_count'] += 1
                    else:
                        req_info[req.type] = {}
                        req_info[req.type]['execution_time'] = req.finish_time - req.arrival_time
                        req_info[req.type]['req_count'] = 1
                else:
                    failed_count += 1
        successful_req_count = 0
        for req_type, req_data in req_info.items():
            successful_req_count += req_data['req_count']
            fn_latency += round((req_data['execution_time'] / req_data['req_count']), 4) / round((
                    int(self.worker.fn_features[str(req_type) + "_req_MIPS"]) / self.worker.fn_features[
                str(req_type) + "_cpu_req"]), 4)

        if len(req_info) != 0:
            self.worker.Episodic_latency = fn_latency / len(req_info)
        self.worker.ec2_vm_up_time_cost_req, self.worker.ec2_vm_up_time_req, self.worker.serverless_fn_cost_req = self.calc_total_user_cost_per_req(successful_req_count)
        self.worker.ec2_vm_up_time_cost, self.worker.ec2_vm_up_time, self.worker.serverless_fn_cost = self.calc_total_user_cost()

        if total_req_count != 0:
            self.worker.Episodic_failure_rate = failed_count / total_req_count

        if write_graph:
            print("GRAPHS")
            ep_reward_sheet.write(ep + 1, 0, ep)
            ep_reward_sheet.write(ep + 1, 1, self.worker.episodic_reward)
            ep_lat_sheet.write(ep + 1, 0, ep)
            ep_lat_sheet.write(ep + 1, 1, self.worker.Episodic_latency)
            ep_t_cost_sheet.write(ep + 1, 0, ep)
            ep_t_cost_sheet.write(ep + 1, 1, (self.worker.serverless_fn_cost + self.worker.ec2_vm_up_time_cost))
            ep_ec2_cost_sheet.write(ep + 1, 0, ep)
            ep_ec2_cost_sheet.write(ep + 1, 1, self.worker.ec2_vm_up_time_cost)
            ep_serverless_cost_sheet.write(ep + 1, 0, ep)
            ep_serverless_cost_sheet.write(ep + 1, 1, self.worker.serverless_fn_cost)
            ep_env_action_sheet.write(ep + 1, 0, ep)
            ep_env_action_sheet.write(ep + 1, 1, float(self.worker.deploy_env_action_total))
            ep_second_action_sheet.write(ep + 1, 0, ep)
            ep_second_action_sheet.write(ep + 1, 1, float(self.worker.second_action_total))
            step_latency_sheet.write(ep + 1, 0, ep)
            step_latency_sheet.write(ep + 1, 1, self.worker.function_latency)
            step_t_costdiff_sheet.write(ep + 1, 0, ep)
            step_t_costdiff_sheet.write(ep + 1, 1, self.worker.total_step_cost/successful_req_count)

            ep_reward_sheet_req.write(ep + 1, 0, ep)
            ep_reward_sheet_req.write(ep + 1, 1, self.worker.episodic_reward / successful_req_count)
            ep_t_cost_sheet_req.write(ep + 1, 0, ep)
            ep_t_cost_sheet_req.write(ep + 1, 1, (
                        self.worker.serverless_fn_cost + self.worker.ec2_vm_up_time_cost) / successful_req_count)
            ep_ec2_cost_sheet_req.write(ep + 1, 0, ep)
            ep_ec2_cost_sheet_req.write(ep + 1, 1, self.worker.ec2_vm_up_time_cost_req)
            ep_serverless_cost_sheet_req.write(ep + 1, 0, ep)
            ep_serverless_cost_sheet_req.write(ep + 1, 1, self.worker.serverless_fn_cost_req)
            ep_env_action_sheet_req.write(ep + 1, 0, ep)
            ep_env_action_sheet_req.write(ep + 1, 1, float(self.worker.deploy_env_action_total / successful_req_count))

            wb_summary.save("summary.xls")


    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())


    def filtered_unavail_action_list(self, act_selected):
        unavail_action_list_serv = []
        unavail_action_list_nserv = []
        print("Action selected is {}".format(act_selected))
        if act_selected == 0:
            for vm in self.worker.serverless_vms:

                if self.fn_type not in vm.idle_containers:
                    if ((vm.cpu - vm.cpu_allocated) < self.worker.fn_features[str(self.fn_type) + "_cpu_req"]) or (
                            (vm.ram - vm.mem_allocated) < self.worker.fn_features[str(self.fn_type) + "_req_ram"]):
                        unavail_action_list_serv.append(vm.id)
            return unavail_action_list_serv
        else:
            for vm in self.worker.ec2_vms:
                if ((vm.cpu - vm.cpu_allocated) < self.worker.fn_features[str(self.fn_type) + "_cpu_req"]) or (
                        (vm.ram - vm.mem_allocated) < self.worker.fn_features[str(self.fn_type) + "_req_ram"]) or (
                        self.worker.ec2_vm_up_time_dict[vm]['status'] != "ON" and vm not in self.worker.pending_vms):
                    unavail_action_list_nserv.append(vm.id)
            return unavail_action_list_nserv


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
        return state_orig, self.state, self.worker.clock

    def select_action(self, step_c):
        state = self.worker.gen_serv_env_state()
        state = np.reshape(state, [1, self.state_size])
        self.action, self.act_type = self.act_test(step_c, state, self.fn_type)
        return state, self.act_type, self.action, self.worker.clock

    def select_action_test(self, step_c, ep):
        state = self.worker.gen_serv_env_state()
        state = np.reshape(state, [1, self.state_size])
        self.action, self.act_type = self.act_test(step_c, state, self.fn_type, ep)
        return state, self.act_type, self.action, self.worker.clock


    def execute_events(self):
        while self.worker.sorted_events:
            ev = self.worker.sorted_events.pop(0)
            prev_clock = self.worker.clock
            self.worker.clock = round(float(ev.received_time), 2)
            time_diff = self.worker.clock - prev_clock
            ev_name = ev.event_name
            if ev_name == "SCHEDULE_REQ":
                self.fn_type = ev.entity_object.type
                self.worker.fn_type = ev.entity_object.type
                self.current_request = ev.entity_object
                # set the current arrival rate for a fn
                if self.worker.fn_rate != ev.entity_object.arrival_rate:
                    self.worker.fn_request_rate[ev.entity_object.type] = []
                    pair = (self.worker.clock, ev.entity_object.arrival_rate)
                    self.worker.fn_request_rate[ev.entity_object.type].append(pair)
                    self.worker.fn_rate = ev.entity_object.arrival_rate
                return True
            elif ev_name == "RE_SCHEDULE_REQ":
                self.worker.req_scheduler(ev.entity_object)
            elif ev_name == "REQ_COMPLETE":
                self.worker.req_completion(ev.entity_object)
            elif ev_name == "CREATE_CONT":
                self.worker.container_creator(ev.entity_object)
            elif ev_name == "CREATE_VM":
                # print(ev.entity_object.type)
                self.worker.create_ec2_vms(ev.entity_object)
            elif ev_name == "KILL_CONT":
                self.worker.container_terminator(ev.entity_object)
            elif ev_name == "KILL_VM":
                self.worker.terminate_ec2_vm(ev.entity_object)
        self.simulation_running = False
        return True

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

        for fn_type in self.worker.serverless_cost_dict:
            ser_fn_cost += self.worker.serverless_cost_dict[fn_type]['mem'] * self.worker.serverless_cost_dict[fn_type][
                'exec_time'] * constants.serverless_mbs_price
        ser_fn_cost += self.worker.serverless_request_no * constants.serverless_price_per_request
        return vm_cost / 3600, vm_time, ser_fn_cost

    def calc_total_user_cost_per_req(self, req_count):
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

        for fn_type in self.worker.serverless_cost_dict:
            ser_fn_cost += self.worker.serverless_cost_dict[fn_type]['mem'] * self.worker.serverless_cost_dict[fn_type][
                'exec_time'] * constants.serverless_mbs_price
        ser_fn_cost += self.worker.serverless_request_no * constants.serverless_price_per_request
        if (req_count - self.worker.serverless_request_no) > 0:
            ec2_vm_up_time_cost_req =(vm_cost / 3600)/(req_count - self.worker.serverless_request_no)
            ec2_vm_up_time_req = vm_time / (req_count - self.worker.serverless_request_no)
        else:
            ec2_vm_up_time_cost_req = 0
            ec2_vm_up_time_req = 0
        if self.worker.serverless_request_no > 0:
            serverless_fn_cost_req = ser_fn_cost/self.worker.serverless_request_no
        else:
            serverless_fn_cost_req = 0

        return ec2_vm_up_time_cost_req, ec2_vm_up_time_req, serverless_fn_cost_req

    def execute_action(self, action1, action2):
        self.current_request.vm_id = action2
        self.current_request.act = [action1, action2]
        self.worker.action = [action1, action2]
        if action1 == 0:
            self.current_request.allocated_vm = self.worker.serverless_vms[action2]
            self.current_request.deployed_env = 's'

            if self.current_request.type in self.current_request.allocated_vm.idle_containers:
                self.current_request.allocated_cont = self.current_request.allocated_vm.idle_containers[
                    self.current_request.type].pop(0)
                if len(self.current_request.allocated_vm.idle_containers[self.current_request.type]) == 0:
                    self.current_request.allocated_vm.idle_containers.pop(self.current_request.type)
                print("clock: {} Removing cont {} from idle list".format(self.worker.clock,
                                                                         self.current_request.allocated_cont.id))
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
        self.worker.ec2_vm_autoscaler()

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
                    total_req_count_within_window += 1
                    if req.status == "Dropped":
                        failed_count += 1
                else:
                    break

        self.worker.ec2_vm_up_time_cost, self.worker.ec2_vm_up_time, self.worker.serverless_fn_cost = self.calc_total_user_cost()
        if total_req_count_within_window != 0:
            fn_failure_rate = failed_count / total_req_count_within_window
        logging.info("CLOCK: {} Cum ec2 vm_up_time_cost: {} Cum serverless_fn_cost: {}".format(self.worker.clock,
                                                                                               self.worker.ec2_vm_up_time_cost,
                                                                                               self.worker.serverless_fn_cost))
        logging.info("CLOCK: {} fn_failure_rate: {}".format(self.worker.clock, fn_failure_rate))

        self.worker.req_latency_prev = self.worker.req_latency

        self.worker.req_cost_prev = self.worker.req_cost
        env_available = False
        if ac1 == 0:
            if self.current_request.type in self.worker.serverless_vms[ac2].idle_containers:
                self.worker.req_latency = 0

            else:
                self.worker.req_latency = constants.container_creation_time / constants.vm_creation_time
        else:
            if self.worker.ec2_vms[ac2] in self.worker.ec2_vm_up_time_dict:
                if self.worker.ec2_vm_up_time_dict[self.worker.ec2_vms[ac2]]['status'] == "ON":
                    env_available = True
                    self.worker.req_latency = 0
            if not env_available:
                if self.worker.ec2_vms[ac2] in self.worker.pending_vms:
                    self.worker.req_latency = (constants.vm_creation_time - (self.worker.clock - self.worker.ec2_vms[
                        ac2].launch_start_time)) / constants.vm_creation_time
                else:
                    self.worker.req_latency = constants.vm_creation_time / constants.vm_creation_time

        if ac1 == 0:
            self.worker.req_cost = self.worker.serverless_cost_dict[self.current_request.type]['mem'] * (
                    self.worker.fn_features[
                        str(self.current_request.type) + "_req_MIPS"] / self.worker.fn_features[
                        str(
                            self.current_request.type) + "_cpu_req"]) * constants.serverless_mbs_price + constants.serverless_price_per_request
        else:

            self.worker.req_cost = (self.worker.ec2_vms[ac2].price / 3600) * (
                    self.worker.fn_features[str(self.current_request.type) + "_req_MIPS"] / self.worker.fn_features[
                str(self.current_request.type) + "_cpu_req"]) * (
                                           (self.worker.ec2_vms[ac2].cpu - self.worker.ec2_vms[ac2].cpu_allocated) /
                                           self.worker.ec2_vms[ac2].cpu)

        self.worker.req_cost = self.worker.req_cost / constants.max_step_vmcost

        if step_c > 1:
            self.worker.fn_failures += fn_failure_rate
            self.worker.function_latency += self.worker.req_latency_prev
            self.worker.total_step_cost += self.worker.req_cost_prev

            x = 0
            y = 1 - x
            reward = - (self.worker.req_cost_prev + (self.worker.req_latency_prev + fn_failure_rate))
            logging.info("CLOCK: {} Step reward: {}".format(self.worker.clock, reward))

            self.worker.episodic_reward += reward
        else:
            reward = 0

        return reward, self.worker.req_cost_prev, self.worker.req_latency_prev

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

