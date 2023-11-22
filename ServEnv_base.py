import glob
import logging
import math
import os
import threading
import uuid
from re import match
import xlwt
from xlwt import Workbook
from threading import Thread
import xlrd
import constants
import definitions as defs
import numpy as np
# import env.ServerlessEnv as se
from sys import maxsize
import cluster

WLS = {0: [0, 1, 2, 3, 4], 1: [0, 1, 2, 3], 2: [0, 1, 2, 3, 4, 5, 6], 3: [0, 1, 2, 3, 4], 4: [0, 1, 2, 3, 4],
       5: [0, 1, 2, 3, 4, 5], 6: [0, 1, 2, 3, 4], 7: [0, 1, 2, 3, 4], 8: [0, 1, 2, 3, 4], 9: [0, 1, 2, 3, 4]}
ep_iter = 0
wl_iter = 0
summary_ep = 0

vm_lock = threading.Lock()
history_lock = threading.Lock()


class Worker:
    def __init__(self, episode):
        self.sorted_request_history_per_window = {}
        self.cont_object_dict_by_type = {}
        self.pending_containers = {}
        self.required_pod_count = {}
        self.pod_counter = {}
        self.fn_request_rate = {}
        self.fn_rate = 0
        self.containers = []
        self.sorted_events = []
        self.dropped_requests = []
        self.clock = 0
        self.simulation_running = True
        self.ec2_vm_up_time_dict = {}
        self.vm_up_time_cost_prev = 0
        self.req_latency = 0
        self.req_latency_prev = 0
        self.req_cost = 0
        self.req_cost_prev = 0
        self.total_step_cost = 0
        self.serverless_cost_dict = {}
        self.serverless_request_no = 0
        self.episodic_reward = 0
        self.function_latency = 0
        self.fn_failures = 0
        self.user_cost = 0
        self.serverless_fn_cost = 0
        self.ec2_vm_up_time_cost = 0
        self.ec2_vm_up_time = 0
        self.ec2_vm_idle_list = []
        self.deploy_env_action_total = 0
        self.second_action_total = 0
        self.ver_cpu_action_total = 0
        self.ver_mem_action_total = 0
        self.hor_action_total = 0
        self.Episodic_failure_rate = 0
        self.Episodic_latency = 0
        self.Episodic_user_cost = 0
        # total_vm_time_diff = 0
        self.total_vm_cost_diff = 0
        self.fn_type = ""
        # Used to iterate over pods in RR manner
        self.features = 20
        self.container_id = 1
        self.episode_no = episode
        self.pending_vms = {}

        self.serverless_vms = []
        self.ec2_vms = []
        self.serverless_vms = cluster.init_serverless_vms(self.clock)
        self.ec2_vms = cluster.init_ec2_vms()

        for vm in self.ec2_vms:
            self.ec2_vm_up_time_dict[vm] = {}
            self.ec2_vm_up_time_dict[vm]['status'] = "OFF"
            self.ec2_vm_up_time_dict[vm]['total_time'] = 0
        for vm in self.ec2_vms:
            self.ec2_vm_up_time_dict[vm]['status'] = "ON"
            self.ec2_vm_up_time_dict[vm]['time_now'] = self.clock
            vm.start_time = self.clock
            self.ec2_vm_idle_list.append(vm)
            vm.idle_start_time = round(self.clock, 2)
            self.sorted_events.append(
                defs.EVENT(self.clock + constants.ec2_vm_idle_time, constants.terminate_ec2_vm,
                           vm))
            break
        self.idle_containers = {}

        self.action = [0, 0]
        self.end_episode = False
        self.summary_episode = 0
        self.local_ep = 0
        self.local_wl = 0
        # self.create_ec2_vms()

        self.serv_env_state_init = cluster.gen_serv_env_state_init()
        self.serv_env_state_min = cluster.gen_serv_env_state_min()
        self.serv_env_state_max = cluster.gen_serv_env_state_max()

        logging.info(
            "ServEnv_Base to be initialized for episode: {}".format(episode))

        self.fn_features = {'fn0_name': "fn0", 'fn0_req_MIPS': 1500, 'fn0_cpu_req': 700, 'fn0_req_ram': 315,
                            'fn1_name': "fn1", 'fn1_req_MIPS': 800, 'fn1_cpu_req': 440, 'fn1_req_ram': 200,
                            'fn2_name': "fn2", 'fn2_req_MIPS': 175, 'fn2_cpu_req': 110, 'fn2_req_ram': 50,
                            'fn3_name': "fn3", 'fn3_req_MIPS': 165, 'fn3_cpu_req': 220, 'fn3_req_ram': 100,
                            'fn4_name': "fn4", 'fn4_req_MIPS': 800, 'fn4_cpu_req': 570, 'fn4_req_ram': 260,
                            'fn5_name': "fn5", 'fn5_req_MIPS': 600, 'fn5_cpu_req': 135, 'fn5_req_ram': 60,
                            'fn6_name': "fn6", 'fn6_req_MIPS': 540, 'fn6_cpu_req': 270, 'fn6_req_ram': 120,
                            'fn7_name': "fn7", 'fn7_req_MIPS': 50, 'fn7_cpu_req': 390, 'fn7_req_ram': 175,
                            'fn8_name': "fn8", 'fn8_req_MIPS': 152, 'fn8_cpu_req': 350, 'fn8_req_ram': 160,
                            'fn9_name': "fn9", 'fn9_req_MIPS': 630, 'fn9_cpu_req': 415, 'fn9_req_ram': 190
                            }

        self.wb = Workbook()
        self.req_wl = self.wb.add_sheet('Results')
        self.req_wl.write(0, 0, 'Fn ID')
        self.req_wl.write(0, 1, 'Fn type')
        self.req_wl.write(0, 2, 'Allocated pod')
        self.req_wl.write(0, 3, 'Arrival time')
        self.req_wl.write(0, 4, 'Start/Dropped time')
        self.req_wl.write(0, 5, 'Finish time')
        self.req_wl.write(0, 6, 'Result')
        self.req_wl.write(0, 7, 'Allocated vm')

        self.pod_data_sheet = self.wb.add_sheet('Pods')
        self.pod_data_sheet.write(0, 0, 'Pod ID')
        self.pod_data_sheet.write(0, 1, 'Pod type')
        self.pod_data_sheet.write(0, 2, 'Start time')
        self.pod_data_sheet.write(0, 3, 'Finish time')

        self.ec2_vm_data_sheet = self.wb.add_sheet('EC2_VMs')
        self.ec2_vm_data_sheet.write(0, 0, 'VM ID')
        self.ec2_vm_data_sheet.write(0, 1, 'type')
        self.ec2_vm_data_sheet.write(0, 2, 'Start time')
        self.ec2_vm_data_sheet.write(0, 3, 'Finish time')

        self.results_sheet_row_counter = 1
        self.pod_sheet_row_counter = 1
        self.vm_sheet_row_counter = 1

        self.fn_types = []
        self.fn_iterator = -1

        self.init_workload()

    def sorter_events(self, item):
        time = float(item.received_time)
        return time

    def pod_sorter(self, item):
        util = float(item.cpu_util)
        return util

    # EC2 VMs are scaled up based on the cpu utilisation and scaled down after some idle time
    def create_ec2_vms(self, vm):
        logging.info(
            "CLOCK: {} vm: {} is getting launched ".format(self.clock, vm.id))
        vm.launch_start_time = 0
        if vm in self.ec2_vm_up_time_dict:
            if self.ec2_vm_up_time_dict[vm]['status'] != "ON":
                self.ec2_vm_up_time_dict[vm]['status'] = "ON"
                self.ec2_vm_up_time_dict[vm]['time_now'] = self.clock
                vm.start_time = self.clock

        else:
            self.ec2_vm_up_time_dict[vm] = {}
            self.ec2_vm_up_time_dict[vm]['status'] = "ON"
            self.ec2_vm_up_time_dict[vm]['time_now'] = self.clock
            self.ec2_vm_up_time_dict[vm]['total_time'] = 0
            vm.start_time = self.clock

        if len(self.pending_vms[vm]) == 0:
            self.ec2_vm_idle_list.append(vm)
            vm.idle_start_time = round(self.clock, 2)
            self.sorted_events.append(
                defs.EVENT(self.clock + constants.ec2_vm_idle_time, constants.terminate_ec2_vm,
                           vm))
        else:
            for req in self.pending_vms[vm]:
                req.allocated_vm = vm
                self.req_scheduler(req)

        self.pending_vms.pop(vm, None)

    def container_creator(self, request):
        self.containers.append(
            defs.CONTAINER(self.container_id, request.allocated_vm, self.fn_features[str(request.type) + "_name"],
                           self.clock, 0,
                           int(self.fn_features[str(request.type) + "_cpu_req"]),
                           int(self.fn_features[str(request.type) + "_req_ram"]), 0, 0,
                           self.container_id, request, [], 0))

        if self.container_id == 1:
            print("CONTAINER 1")

        self.container_id += 1
        request.allocated_cont = self.containers[len(self.containers) - 1]

        if request.type in request.allocated_vm.running_list:
            request.allocated_vm.running_list[request.type].append(request.allocated_cont)
        else:
            request.allocated_vm.running_list[request.type] = []
            request.allocated_vm.running_list[request.type].append(request.allocated_cont)

        # print(PODS[len(PODS) - 1])
        if request.type in self.cont_object_dict_by_type:
            self.cont_object_dict_by_type[request.type].append(request.allocated_cont)
        else:
            self.cont_object_dict_by_type[request.type] = []
            self.cont_object_dict_by_type[request.type].append(request.allocated_cont)

        self.req_scheduler(request)

    # Containers are terminated after some idle time
    def container_terminator(self, cont):
        print("clock: {} checking cont {} for destroying".format(self.clock, cont.id))
        if cont.type in cont.allocated_vm.idle_containers:
            if cont in cont.allocated_vm.idle_containers[cont.type]:
                print("clock: {} cont {} in vm idle list".format(self.clock, cont.id))
                print("the spent time idling : {}".format(self.clock - round(cont.idle_start_time, 2)))
                if round(self.clock - round(cont.idle_start_time, 2), 2) == constants.container_idle_time:
                    cont.allocated_vm.running_list[cont.type].remove(cont)
                    print("clock: {} Destroying cont {} ".format(self.clock, cont.id))
                    # cont.allocated_vm.term_list[cont.type].append(cont)
                    if cont.type in cont.allocated_vm.term_list:
                        cont.allocated_vm.term_list[cont.type].append(cont)
                    else:
                        cont.allocated_vm.term_list[cont.type] = []
                        cont.allocated_vm.term_list[cont.type].append(cont)

                    cont.allocated_vm.cpu_allocated -= cont.cpu_req
                    cont.allocated_vm.mem_allocated -= cont.ram_req
                    cont.term_time = self.clock
                    self.pod_data_sheet.write(self.pod_sheet_row_counter, 0, cont.id)
                    self.pod_data_sheet.write(self.pod_sheet_row_counter, 1, cont.type)
                    self.pod_data_sheet.write(self.pod_sheet_row_counter, 2, cont.start_time)
                    self.pod_data_sheet.write(self.pod_sheet_row_counter, 3, cont.term_time)
                    self.pod_sheet_row_counter += 1

                    self.cont_object_dict_by_type[cont.type].remove(cont)
                    self.containers.remove(cont)
                    cont.allocated_vm.idle_containers[cont.type].remove(cont)
                    if len(cont.allocated_vm.idle_containers[cont.type]) == 0:
                        cont.allocated_vm.idle_containers.pop(cont.type)

                    self.wb.save(
                        "results/Results_Episode_" + str(self.episode_no) + "_ep" + str(self.local_ep) + "_wl" + str(
                            self.local_wl) + ".xls")

    def terminate_ec2_vm(self, vm):
        print("clock {} checking the spent time idling vm : {}".format(self.clock,
                                                                       self.clock - round(vm.idle_start_time, 2)))
        if vm in self.ec2_vm_idle_list:
            if (self.clock - round(vm.idle_start_time, 2)) == constants.ec2_vm_idle_time:
                self.ec2_vm_idle_list.remove(vm)
                self.ec2_vm_up_time_dict[vm]['status'] = "OFF"
                self.ec2_vm_up_time_dict[vm]['total_time'] += self.clock - \
                                                              self.ec2_vm_up_time_dict[
                                                                  vm][
                                                                  'time_now']
                logging.info(
                    "clock: {} ec2 vm {} ec2 up time so far: {} latest segment: {}".format(self.clock,
                                                                                           vm.id,
                                                                                           self.ec2_vm_up_time_dict[
                                                                                               vm][
                                                                                               'total_time'],
                                                                                           self.clock - \
                                                                                           self.ec2_vm_up_time_dict[
                                                                                               vm][
                                                                                               'time_now']))

                self.ec2_vm_up_time_dict[vm]['time_now'] = self.clock
                vm.idle_start_time = 0
                vm.finish_time = self.clock
                self.ec2_vm_data_sheet.write(self.vm_sheet_row_counter, 0, vm.id)
                self.ec2_vm_data_sheet.write(self.vm_sheet_row_counter, 1, vm.type)
                self.ec2_vm_data_sheet.write(self.vm_sheet_row_counter, 2, vm.start_time)
                self.ec2_vm_data_sheet.write(self.vm_sheet_row_counter, 3, vm.finish_time)
                self.vm_sheet_row_counter += 1
                self.wb.save(
                    "results/Results_Episode_" + str(self.episode_no) + "_ep" + str(self.local_ep) + "_wl" + str(
                        self.local_wl) + ".xls")

    def req_scheduler(self, request):
        if request.act[0] == 0:
            if request.allocated_cont is not None:
                request.allocated_vm.cpu_used += self.fn_features[str(request.type) + "_cpu_req"]
                request.allocated_vm.ram_used += self.fn_features[str(request.type) + "_req_ram"]
                request.allocated_cont.cpu_util += self.fn_features[str(request.type) + "_cpu_req"]
                request.allocated_cont.ram_util += self.fn_features[str(request.type) + "_req_ram"]

                request.start_time = self.clock
                request.finish_time = self.clock + self.fn_features[
                    str(request.type) + "_req_MIPS"] / self.fn_features[
                                          str(request.type) + "_cpu_req"]
                self.sorted_events.append(
                    defs.EVENT(request.finish_time, constants.finish_request, request))

                self.serverless_request_no += 1

            else:
                self.sorted_events.append(
                    defs.EVENT(self.clock + constants.container_creation_time, constants.create_new_cont, request))

        else:
            if request.allocated_vm is not None:
                request.allocated_vm.cpu_used += self.fn_features[str(request.type) + "_cpu_req"]
                request.allocated_vm.ram_used += self.fn_features[str(request.type) + "_req_ram"]

                if request.allocated_vm in self.ec2_vm_idle_list:
                    self.ec2_vm_idle_list.remove(request.allocated_vm)
                    request.allocated_vm.idle_start_time = 0

                if request.allocated_vm in self.ec2_vm_up_time_dict:
                    if self.ec2_vm_up_time_dict[request.allocated_vm]['status'] != "ON":
                        self.ec2_vm_up_time_dict[request.allocated_vm]['status'] = "ON"
                        self.ec2_vm_up_time_dict[request.allocated_vm]['time_now'] = self.clock
                else:
                    self.ec2_vm_up_time_dict[request.allocated_vm] = {}
                    self.ec2_vm_up_time_dict[request.allocated_vm]['status'] = "ON"
                    self.ec2_vm_up_time_dict[request.allocated_vm]['time_now'] = self.clock
                    self.ec2_vm_up_time_dict[request.allocated_vm]['total_time'] = 0

                if request.type in request.allocated_vm.running_list:
                    request.allocated_vm.running_list[request.type].append(request)
                else:
                    request.allocated_vm.running_list[request.type] = []
                    request.allocated_vm.running_list[request.type].append(request)
                request.start_time = self.clock
                request.finish_time = self.clock + self.fn_features[
                    str(request.type) + "_req_MIPS"] / self.fn_features[str(request.type) + "_cpu_req"]
                self.sorted_events.append(
                    defs.EVENT(request.finish_time, constants.finish_request, request))
            else:
                if self.ec2_vms[request.vm_id] in self.pending_vms:
                    self.pending_vms[self.ec2_vms[request.vm_id]].append(request)
                else:
                    self.sorted_events.append(
                        defs.EVENT(self.clock + constants.vm_creation_time, constants.create_new_vm,
                                   self.ec2_vms[request.vm_id]))
                    self.ec2_vms[request.vm_id].launch_start_time = self.clock
                    self.pending_vms[self.ec2_vms[request.vm_id]] = []
                    self.pending_vms[self.ec2_vms[request.vm_id]].append(request)

        self.sorted_events = sorted(self.sorted_events, key=self.sorter_events)

    def ec2_vm_autoscaler(self):
        cpu_util = 0
        vm_count = 0
        vm_req_sent = 0
        for vm in self.ec2_vms:
            if self.ec2_vm_up_time_dict[vm]['status'] == "ON" or vm in self.pending_vms:
                cpu_util += vm.cpu_allocated / vm.cpu
                vm_count += 1
        if vm_count != 0:
            cpu_util = cpu_util / vm_count
            desired_replicas = math.ceil(vm_count * cpu_util / constants.ec2_vm_scale_cpu_util)

            if desired_replicas > 0:
                logging.info("CLOCK: {} Desired vms: {} now running vm count: {} ".format(self.clock, desired_replicas,
                                                                                          vm_count))
                new_vm_count = min(desired_replicas, int(constants.max_num_ec2_vms))
            else:
                new_vm_count = 1

            if new_vm_count > vm_count:
                new_vms_to_create = new_vm_count - vm_count

                for vm in self.ec2_vm_up_time_dict:
                    if self.ec2_vm_up_time_dict[vm]['status'] != "ON" and vm not in self.pending_vms:
                        logging.info(
                            "CLOCK: {} Request sent by ec2 auto scaler to create new vm ".format(self.clock))
                        self.sorted_events.append(
                            defs.EVENT(self.clock + constants.vm_creation_time, constants.create_new_vm, vm))
                        vm_req_sent += 1
                        vm.launch_start_time = self.clock
                        self.pending_vms[vm] = []

                    if vm_req_sent == new_vms_to_create:
                        break

    def req_completion(self, request):

        request.finish_time = self.clock
        request.status = "Ok"
        self.req_wl.write(self.results_sheet_row_counter, 0, request.id)
        self.req_wl.write(self.results_sheet_row_counter, 1, request.type)
        if request.allocated_cont is not None:
            self.req_wl.write(self.results_sheet_row_counter, 2, int(request.allocated_cont.id))
        else:
            self.req_wl.write(self.results_sheet_row_counter, 7, int(request.allocated_vm.id))
        self.req_wl.write(self.results_sheet_row_counter, 3, request.arrival_time)
        self.req_wl.write(self.results_sheet_row_counter, 4, request.start_time)
        self.req_wl.write(self.results_sheet_row_counter, 5, request.finish_time)
        self.req_wl.write(self.results_sheet_row_counter, 6, "Ok")

        if request.type in self.sorted_request_history_per_window:
            self.sorted_request_history_per_window[str(request.type)].append(request)
        else:
            self.sorted_request_history_per_window[str(request.type)] = []
            self.sorted_request_history_per_window[str(request.type)].append(request)

        self.results_sheet_row_counter += 1
        self.wb.save("results/Results_Episode_" + str(self.episode_no) + "_ep" + str(self.local_ep) + "_wl" + str(
            self.local_wl) + ".xls")

        if request.deployed_env == "s":
            request.allocated_cont.running_req = None
            request.allocated_cont.completed_req_list.append(request)
            request.allocated_cont.cpu_util = 0
            request.allocated_cont.ram_util = 0
            request.allocated_cont.idle_start_time = round(self.clock, 2)
            if request.type in request.allocated_vm.idle_containers:
                request.allocated_vm.idle_containers[request.type].append(request.allocated_cont)
            else:
                request.allocated_vm.idle_containers[request.type] = []
                request.allocated_vm.idle_containers[request.type].append(request.allocated_cont)

            self.sorted_events.append(
                defs.EVENT(self.clock + constants.container_idle_time, constants.terminate_container,
                           request.allocated_cont))

            self.sorted_events = sorted(self.sorted_events, key=self.sorter_events)
            request.allocated_cont.allocated_vm.cpu_used -= self.fn_features[str(request.type) + "_cpu_req"]
            request.allocated_cont.allocated_vm.ram_used -= self.fn_features[str(request.type) + "_req_ram"]

            self.serverless_cost_dict[request.type]['exec_time'] += request.finish_time - request.start_time


        else:
            request.allocated_vm.cpu_used -= self.fn_features[str(request.type) + "_cpu_req"]
            request.allocated_vm.ram_used -= self.fn_features[str(request.type) + "_req_ram"]
            request.allocated_vm.cpu_allocated -= self.fn_features[str(request.type) + "_cpu_req"]
            request.allocated_vm.mem_allocated -= self.fn_features[str(request.type) + "_req_ram"]

            request.allocated_vm.running_list[request.type].remove(request)
            if request.type in request.allocated_vm.term_list:
                request.allocated_vm.term_list[request.type].append(request)
            else:
                request.allocated_vm.term_list[request.type] = []
                request.allocated_vm.term_list[request.type].append(request)

            if len(request.allocated_vm.running_list[request.type]) == 0:
                request.allocated_vm.running_list.pop(request.type)

            if len(request.allocated_vm.running_list) == 0:
                self.ec2_vm_idle_list.append(request.allocated_vm)
                request.allocated_vm.idle_start_time = round(self.clock, 2)
                self.sorted_events.append(
                    defs.EVENT(self.clock + constants.ec2_vm_idle_time, constants.terminate_ec2_vm,
                               request.allocated_vm))

    def calculate_average_arrival_rate(self):
        total_req_count_within_window = 0
        total_rate = 0
        average_rate = 0
        if self.fn_type in self.sorted_request_history_per_window:
            for req in reversed(self.sorted_request_history_per_window[self.fn_type]):
                if req.arrival_time > self.clock - constants.request_rate_window:
                    total_req_count_within_window += 1
                    total_rate += req.arrival_rate
                else:
                    break
        if total_req_count_within_window != 0:
            average_rate = round(total_rate / total_req_count_within_window, 2)
        return average_rate

    def gen_serv_env_state(self):
        pod_info = {}
        env_state = np.zeros(124)
        function_latency = 0

        i = 0
        x = 0
        while i < len(self.serverless_vms):
            env_state[x] = (self.serverless_vms[i].cpu - self.serverless_vms[i].cpu_allocated) / self.serverless_vms[
                i].cpu
            env_state[x + 1] = (self.serverless_vms[i].ram - self.serverless_vms[i].mem_allocated) / \
                               self.serverless_vms[
                                   i].ram
            if self.fn_type in self.serverless_vms[i].idle_containers:
                env_state[x + 2] = len(
                    self.serverless_vms[i].idle_containers[self.fn_type]) / constants.max_containers_in_vm
            else:
                env_state[x + 2] = 0
            x = x + 3
            i += 1

        # the current requested cpu of a container
        env_state[x] = self.fn_features[str(self.fn_type) + "_cpu_req"] / constants.max_cont_cpu_req
        # the current requested mem of a container
        env_state[x + 1] = self.fn_features[str(self.fn_type) + "_req_ram"] / constants.max_cont_mem_req
        rate = 0
        mark = self.clock
        if (self.clock - self.fn_request_rate[self.fn_type][-1][0]) >= (
                constants.vm_creation_time + constants.ec2_vm_idle_time) or self.clock == \
                self.fn_request_rate[self.fn_type][-1][0]:
            rate = self.fn_request_rate[self.fn_type][-1][1]

        else:
            for index, y in enumerate(reversed(self.fn_request_rate[self.fn_type])):
                if (self.clock - y[0]) < (constants.vm_creation_time + constants.ec2_vm_idle_time):
                    rate += int(y[1]) * (mark - y[0])
                    mark = y[0]
                else:
                    rate += int(y[1]) * (mark - y[0])
                    break
            rate = rate / (self.clock - y[0])
        env_state[x + 2] = round((rate / constants.max_request_rate), 2)
        env_state[x + 3] = self.action[0]
        x = x + 4

        j = 0
        while j < len(self.ec2_vms):
            env_state[x] = ((self.ec2_vms[j].cpu - self.ec2_vms[j].cpu_allocated) / self.ec2_vms[j].cpu)
            env_state[x + 1] = ((self.ec2_vms[j].ram - self.ec2_vms[j].mem_allocated) / self.ec2_vms[j].ram)
            # request scheduling latency if allocated to this vm
            if self.ec2_vm_up_time_dict[self.ec2_vms[j]]['status'] == "ON":
                env_state[x + 2] = 0
            elif self.ec2_vms[j] in self.pending_vms:
                env_state[x + 2] = (constants.vm_creation_time - (self.clock - self.ec2_vms[
                    j].launch_start_time)) / constants.vm_creation_time
            else:
                env_state[x + 2] = 1
            x = x + 3
            j += 1

        return env_state

    def init_workload(self):
        print("New workload created for new episode {}".format(self.episode_no % constants.num_wls))

        wbr = xlrd.open_workbook("wl/wl2.xls")
        sheet = wbr.sheet_by_index(0)
        for i in range(sheet.nrows - 1):
            fn_name = int(sheet.cell_value(i + 1, 0))
            arr_time = sheet.cell_value(i + 1, 1)
            arr_rate = sheet.cell_value(i + 1, 2)
            idr = sheet.cell_value(i + 1, 3)

            fn_type = "fn" + str(fn_name)
            if not (fn_type in self.fn_types):
                self.fn_types.append(fn_type)
            if not (fn_type in self.fn_request_rate):
                self.fn_request_rate[fn_type] = {}

            if fn_type not in self.serverless_cost_dict:
                self.serverless_cost_dict[fn_type] = {}
                self.serverless_cost_dict[fn_type]['mem'] = self.fn_features[str(fn_type) + "_req_ram"]
                self.serverless_cost_dict[fn_type]['exec_time'] = 0

            req_obj = defs.REQUEST(idr, "fn" + str(fn_name), None, None, arr_time, 0, 0, arr_rate, "initial", None, 's',
                                   [])

            # Event list (unsorted list) consists of the event received time, event name and the object associated with the event
            self.sorted_events.append(defs.EVENT(arr_time, constants.schedule_request, req_obj))

        self.sorted_events = sorted(self.sorted_events, key=self.sorter_events)
