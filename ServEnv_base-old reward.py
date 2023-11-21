import logging
import math
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

# wb = Workbook()
# req_wl = wb.add_sheet('Results')
# req_wl.write(0, 0, 'Fn ID')
# req_wl.write(0, 1, 'Fn type')
# req_wl.write(0, 2, 'Allocated pod')
# req_wl.write(0, 3, 'Arrival time')
# req_wl.write(0, 4, 'Start/Dropped time')
# req_wl.write(0, 5, 'Finish time')
# req_wl.write(0, 6, 'Result')
#
# pod_data_sheet = wb.add_sheet('Pods')
# pod_data_sheet.write(0, 0, 'Pod ID')
# pod_data_sheet.write(0, 1, 'Pod type')
# pod_data_sheet.write(0, 2, 'Start time')
# pod_data_sheet.write(0, 3, 'Finish time')
#
# results_sheet_row_counter = 1
# pod_sheet_row_counter = 1
#
serverless_vm_features = {'vm0_cpu_total_speed': 23200, 'vm0_mem': 8000, 'vm0_cpu_used': 0, 'vm0_mem_used': 0,
                          'vm0_cpu_allocated': 0, 'vm0_mem_allocated': 0, 'vm0_price': 0.043, 'vm0_nw_bandwidth': 4000,
                          'vm0_diskio_bandwidth': 4000, 'vm1_cpu_total_speed': 44800, 'vm1_mem': 16000,
                          'vm1_cpu_used': 0,
                          'vm1_mem_used': 0, 'vm1_cpu_allocated': 0, 'vm1_mem_allocated': 0, 'vm1_price': 0.086,
                          'vm1_nw_bandwidth': 4000, 'vm1_diskio_bandwidth': 4000, 'vm2_cpu_total_speed': 76800,
                          'vm2_mem': 32000,
                          'vm2_cpu_used': 0, 'vm2_mem_used': 0, 'vm2_cpu_allocated': 0, 'vm2_mem_allocated': 0,
                          'vm2_price': 0.172,
                          'vm2_nw_bandwidth': 4000, 'vm2_diskio_bandwidth': 4000, 'vm3_cpu_total_speed': 147200,
                          'vm3_mem': 64000,
                          'vm3_cpu_used': 0, 'vm3_mem_used': 0, 'vm3_cpu_allocated': 0, 'vm3_mem_allocated': 0,
                          'vm3_price': 0.344,
                          'vm3_nw_bandwidth': 4000, 'vm3_diskio_bandwidth': 4000}

ec2_vm_features = {'vm0_cpu_total_speed': 23600, 'vm0_mem': 8000, 'vm0_cpu_used': 0, 'vm0_mem_used': 0,
                   'vm0_cpu_allocated': 0, 'vm0_mem_allocated': 0,
                   'vm0_price': 0.108, 'vm1_cpu_total_speed': 40000, 'vm1_mem': 16000, 'vm1_cpu_used': 0,
                   'vm1_mem_used': 0, 'vm1_cpu_allocated': 0, 'vm1_mem_allocated': 0, 'vm1_price': 0.1696,
                   'vm2_cpu_total_speed': 99200, 'vm2_mem': 32000,
                   'vm2_cpu_used': 0, 'vm2_mem_used': 0, 'vm2_cpu_allocated': 0, 'vm2_mem_allocated': 0,
                   'vm2_price': 0.48, 'vm3_cpu_total_speed': 160000, 'vm3_mem': 64000,
                   'vm3_cpu_used': 0, 'vm3_mem_used': 0, 'vm3_cpu_allocated': 0, 'vm3_mem_allocated': 0,
                   'vm3_price': 0.864}

# from peeking behind paper
# vm0 > 2.9 GHz X 2 cores
# vm1 > 2.8 X 2
# vm2 > 2.4 X 2
# vm3 > 2.3 X 2

# Serverless vms
# intel
# vm0 > (2.9 GHz X 2 core) x 8 GB memory= 23200 m, 8000 MB
# vm1 > (2.8 X 4 cores) x 16 GB = 44800, 16000 MB
# vm2 > (2.4 X 8) x 32 GB = 76800, 32000
# vm3 > (2.3 X 16) x 64 GB = 147200, 64000

# EC2 instances Most similar to the serverless vms above
# vm0 > m6a.large > (2.95 GHz X 2 core) x 8 GB memory= 23600 m, 8000 MB > 0.108
# vm2 > t4g.xlarge > (2.5 X 4) x 16 GB = 40000, 16000 > $0.1696
# vm1 > m5.2xlarge > (3.1 X 8 cores) x 32 GB = 99200, 32000 MB > 0.48
# vm3 > m5a.4xlarge > (2.5 X 16) x 64 GB = 160000, 64000 > $0.864
# ********************
# vm3 > a1.4xlarge > (2.3 X 16) x 32 GB = 147200, 32000 > $0.408


# CPU >>> 460, 920, 1380, 1840, 2300, 2760, up to 9200>>> 41
# mem >>> 150, 300, up to 3000 >> 41
# replicas >>> 1, 2 up to 10 replicas >> 21

# fn0 > 10% of a core (take 2.3 as clock speed), 300 MB = 920 m, 300 MB
# keepin ginitial concurrency at 4 > fn0 request MIPS > 276 m ( to give 1.2 exec time when allocated with 230 ( 1/4th of pod cpu)

# VM type 1
# vm0_latency = 0.001  # second
# vm0_power_cpu = 0.5  # WATT
# vm0_power_transmission = 0.2
# vm0_power_idle = 0.002

# VM type 2

# VM type 3

# VM type 4

# fn_features = {'fn0_name': "fn0", 'fn0_pod_cpu_req': 920, 'fn0_cpu_req': 276, 'fn0_pod_ram_req': 300,
#                'fn0_pod_cpu_util': 0, 'fn0_pod_ram_util': 0, 'fn0_req_per_sec': 10, 'fn0_num_current_replicas': 1,
#                'fn0_num_max_replicas': 2, 'fn0_inflight_requests': 0, 'fn1_name': "fn1", 'fn1_pod_cpu_req': 100,
#                'fn1_cpu_req': 20, 'fn1_pod_ram_req': 300, 'fn1_pod_cpu_util': 0, 'fn1_pod_ram_util': 0,
#                'fn1_req_per_sec': 10,
#                'fn1_num_current_replicas': 1, 'fn1_num_max_replicas': 1, 'fn1_inflight_requests': 0, 'fn2_name': "fn2",
#                'fn2_pod_cpu_req': 100, 'fn2_cpu_req': 20, 'fn2_pod_ram_req': 300, 'fn2_pod_cpu_util': 0,
#                'fn2_pod_ram_util': 0,
#                'fn2_req_per_sec': 10, 'fn2_num_current_replicas': 1, 'fn2_num_max_replicas': 1,
#                'fn2_inflight_requests': 0}
# FN 1

# FN 2

# FN 3

# serverless_vms = []
vm_lock = threading.Lock()
# pods = []
# cont_object_dict_by_type = {}
# ec2_vm_up_time_dict = {}
# ec2_vm_up_time_cost_prev = 0
# episodic_reward = 0
# function_latency = 0
# fn_failures = 0
# total_vm_time_diff = 0
# total_vm_cost_diff = 0
# ver_action_total = 0
# hor_action_total = 0
# Episodic_latency = 0
# Episodic_failure_rate = 0
# events = []
# sorted_events = []
# dropped_requests = []
# pending_containers = {}
# sorted_request_history_per_window = {}
history_lock = threading.Lock()


# clock = 0
# simulation_running = True
# required_pod_count = {}
# fn_request_rate = {'fn0': 0, 'fn1': 0, 'fn2': 0}

# Used to iterate over pods in RR manner
# pod_counters = {}
# serv_env_state_init = []
# serv_env_state_max = []
# serv_env_state_min = []


# features = 20
#
# container_id = 1
#
# episode_no = 0


class Worker:
    def __init__(self, episode):
        # global sorted_request_history_per_window
        # global serv_env_state_init
        # global serv_env_state_min
        # global serv_env_state_max
        self.sorted_request_history_per_window = {}
        self.cont_object_dict_by_type = {}
        self.pending_containers = {}
        self.required_pod_count = {}
        self.pod_counter = {}
        self.fn_request_rate = {}
        self.containers = []
        self.sorted_events = []
        self.dropped_requests = []
        self.clock = 0
        self.simulation_running = True
        self.ec2_vm_up_time_dict = {}
        self.vm_up_time_cost_prev = 0
        self.req_latency = 0
        self.req_latency_prev = 0
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
        # self.fn_type = fn
        self.pending_vms = {}

        self.serverless_vms = []
        self.ec2_vms = []
        self.init_serverless_vms()
        self.init_ec2_vms()
        self.idle_containers = {}

        self.action = [0, 0]
        # self.create_ec2_vms()

        self.serv_env_state_init = self.gen_serv_env_state_init()
        self.serv_env_state_min = self.gen_serv_env_state_min()
        self.serv_env_state_max = self.gen_serv_env_state_max()

        # with vm_lock:
        #     for vm in serverless_vms:
        #         if fn in vm.running_list:
        #             for pod in vm.running_list[fn]:
        #                 vm.cpu_allocated -= pod.cpu_req
        #                 vm.mem_allocated -= pod.ram_req
        #
        #             vm.running_list.pop(fn)

        logging.info(
            "ServEnv_Base to be initialized for episode: {}".format(episode))

        # cpu for cont calculated as per ec2 vm0's config
        self.fn_features = {'fn0_name': "fn0", 'fn0_req_MIPS': 120, 'fn0_cpu_req': 155, 'fn0_req_ram': 70,
                            'fn1_name': "fn1", 'fn1_req_MIPS': 966, 'fn1_cpu_req': 180, 'fn1_req_ram': 80,
                            'fn2_name': "fn2", 'fn2_req_MIPS': 874, 'fn2_cpu_req': 110, 'fn2_req_ram': 50,
                            'fn3_name': "fn3", 'fn3_req_MIPS': 828, 'fn3_cpu_req': 220, 'fn3_req_ram': 100,
                            'fn4_name': "fn4", 'fn4_req_MIPS': 920, 'fn4_cpu_req': 210, 'fn4_req_ram': 95,
                            'fn5_name': "fn5", 'fn5_req_MIPS': 644, 'fn5_cpu_req': 135, 'fn5_req_ram': 60,
                            'fn6_name': "fn6", 'fn6_req_MIPS': 713, 'fn6_cpu_req': 270, 'fn6_req_ram': 120,
                            'fn7_name': "fn7", 'fn7_req_MIPS': 250, 'fn7_cpu_req': 390, 'fn7_req_ram': 175,
                            'fn8_name': "fn8", 'fn8_req_MIPS': 759, 'fn8_cpu_req': 350, 'fn8_req_ram': 160,
                            'fn9_name': "fn9", 'fn9_req_MIPS': 390, 'fn9_cpu_req': 415, 'fn9_req_ram': 190
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

        self.fn_types = []
        self.fn_iterator = -1
        # self.fn_type = None

        self.init_workload()

    def gen_serv_env_state_init(self):
        env_state = np.zeros(124)
        return env_state

    def gen_serv_env_state_min(self):
        env_state = np.zeros(124)
        # global serverless_vms

        i = 0
        # x = 0
        # while i < len(self.serverless_vms):
        #     # env_state.append(0)  # cpu
        #     # env_state.append(0)  # ram
        #     env_state[x] = 0  # price
        #     env_state[x + 1] = 0  # bandwidth
        #     env_state[x + 2] = 0  # latency
        #     env_state[x + 3] = 0  # utilization
        #     env_state[x + 4] = 0
        #     env_state[x + 5] = 0
        #     env_state[x + 6] = 0
        #
        #     i += 1
        #     x = x + 7

        return env_state

    def gen_serv_env_state_max(self):
        # env_state = np.zeros(90)
        env_state = [maxsize] * 124

        return env_state

    def sorter_events(self, item):
        time = float(item.received_time)
        return time

    def pod_sorter(self, item):
        util = float(item.cpu_util)
        return util

    def init_serverless_vms(self):
        for i in range(5):  # IOT DEVICE
            self.serverless_vms.append(
                defs.VM(0, len(self.serverless_vms), int(serverless_vm_features['vm0_cpu_total_speed']),
                        int(serverless_vm_features['vm0_mem']),
                        int(serverless_vm_features['vm0_cpu_used']), int(serverless_vm_features['vm0_mem_used']),
                        int(serverless_vm_features['vm0_cpu_allocated']),
                        int(serverless_vm_features['vm0_mem_allocated']), float(serverless_vm_features['vm0_price']),
                        int(serverless_vm_features['vm0_nw_bandwidth']),
                        int(serverless_vm_features['vm0_diskio_bandwidth']), {}, {}, {},
                        "s", 0, 0))

            self.serverless_vms.append(
                defs.VM(1, len(self.serverless_vms), int(serverless_vm_features['vm1_cpu_total_speed']),
                        int(serverless_vm_features['vm1_mem']),
                        int(serverless_vm_features['vm1_cpu_used']), int(serverless_vm_features['vm1_mem_used']),
                        int(serverless_vm_features['vm1_cpu_allocated']),
                        int(serverless_vm_features['vm1_mem_allocated']), float(serverless_vm_features['vm1_price']),
                        int(serverless_vm_features['vm1_nw_bandwidth']),
                        int(serverless_vm_features['vm1_diskio_bandwidth']), {}, {}, {},
                        "s", 0, 0))

            self.serverless_vms.append(
                defs.VM(2, len(self.serverless_vms), int(serverless_vm_features['vm2_cpu_total_speed']),
                        int(serverless_vm_features['vm2_mem']),
                        int(serverless_vm_features['vm2_cpu_used']), int(serverless_vm_features['vm2_mem_used']),
                        int(serverless_vm_features['vm2_cpu_allocated']),
                        int(serverless_vm_features['vm2_mem_allocated']), float(serverless_vm_features['vm2_price']),
                        int(serverless_vm_features['vm2_nw_bandwidth']),
                        int(serverless_vm_features['vm2_diskio_bandwidth']), {}, {}, {},
                        "s", 0, 0))

            self.serverless_vms.append(
                defs.VM(3, len(self.serverless_vms), int(serverless_vm_features['vm3_cpu_total_speed']),
                        int(serverless_vm_features['vm3_mem']),
                        int(serverless_vm_features['vm3_cpu_used']), int(serverless_vm_features['vm3_mem_used']),
                        int(serverless_vm_features['vm3_cpu_allocated']),
                        int(serverless_vm_features['vm3_mem_allocated']), float(serverless_vm_features['vm3_price']),
                        int(serverless_vm_features['vm3_nw_bandwidth']),
                        int(serverless_vm_features['vm3_diskio_bandwidth']), {}, {}, {},
                        "s", 0, 0))

    def init_ec2_vms(self):
        for i in range(5):
            self.ec2_vms.append(
                defs.VM(0, len(self.ec2_vms), int(ec2_vm_features['vm0_cpu_total_speed']),
                        int(ec2_vm_features['vm0_mem']),
                        int(ec2_vm_features['vm0_cpu_used']), int(ec2_vm_features['vm0_mem_used']),
                        int(ec2_vm_features['vm0_cpu_allocated']),
                        int(ec2_vm_features['vm0_mem_allocated']), float(ec2_vm_features['vm0_price']), 0, 0, {}, {},
                        {}, "ns", 0, 0))

            self.ec2_vms.append(
                defs.VM(1, len(self.ec2_vms), int(ec2_vm_features['vm1_cpu_total_speed']),
                        int(ec2_vm_features['vm1_mem']),
                        int(ec2_vm_features['vm1_cpu_used']), int(ec2_vm_features['vm1_mem_used']),
                        int(ec2_vm_features['vm1_cpu_allocated']),
                        int(ec2_vm_features['vm1_mem_allocated']), float(ec2_vm_features['vm1_price']), 0, 0, {}, {},
                        {}, "ns", 0, 0))

            self.ec2_vms.append(
                defs.VM(2, len(self.ec2_vms), int(ec2_vm_features['vm2_cpu_total_speed']),
                        int(ec2_vm_features['vm2_mem']),
                        int(ec2_vm_features['vm2_cpu_used']), int(ec2_vm_features['vm2_mem_used']),
                        int(ec2_vm_features['vm2_cpu_allocated']),
                        int(ec2_vm_features['vm2_mem_allocated']), float(ec2_vm_features['vm2_price']), 0, 0, {}, {},
                        {}, "ns", 0, 0))

            self.ec2_vms.append(
                defs.VM(3, len(self.ec2_vms), int(ec2_vm_features['vm3_cpu_total_speed']),
                        int(ec2_vm_features['vm3_mem']),
                        int(ec2_vm_features['vm3_cpu_used']), int(ec2_vm_features['vm3_mem_used']),
                        int(ec2_vm_features['vm3_cpu_allocated']),
                        int(ec2_vm_features['vm3_mem_allocated']), float(ec2_vm_features['vm3_price']), 0, 0, {}, {},
                        {}, "ns", 0, 0))

        for vm in self.ec2_vms:
            self.ec2_vm_up_time_dict[vm] = {}
            self.ec2_vm_up_time_dict[vm]['status'] = "OFF"
            self.ec2_vm_up_time_dict[vm]['total_time'] = 0
        for vm in self.ec2_vms:
            self.ec2_vm_up_time_dict[vm]['status'] = "ON"
            self.ec2_vm_up_time_dict[vm]['time_now'] = self.clock
            break

    # EC2 VMs are scaled up based on the cpu utilisation and scaled down after some idle time
    def create_ec2_vms(self, vm):
        vm.launch_start_time = 0
        if vm in self.ec2_vm_up_time_dict:
            if self.ec2_vm_up_time_dict[vm]['status'] != "ON":
                # logging.info(
                #     "VM {} is already in ON status so not making changes to dict".format(vm.id))
                # else:
                self.ec2_vm_up_time_dict[vm]['status'] = "ON"
                self.ec2_vm_up_time_dict[vm]['time_now'] = self.clock
        else:
            self.ec2_vm_up_time_dict[vm] = {}
            self.ec2_vm_up_time_dict[vm]['status'] = "ON"
            self.ec2_vm_up_time_dict[vm]['time_now'] = self.clock
            self.ec2_vm_up_time_dict[vm]['total_time'] = 0

        if len(self.pending_vms[vm]) == 0:
            self.ec2_vm_idle_list.append(vm)
            vm.idle_start_time = self.clock
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

        # request.allocated_vm.cpu_allocated += int(self.fn_features[str(request.type) + "_cpu_req"])
        # request.allocated_vm.mem_allocated += int(self.fn_features[str(request.type) + "_req_ram"])

        if self.container_id == 1:
            print("CONTAINER 1")

        self.container_id += 1
        request.allocated_cont = self.containers[len(self.containers) - 1]

        # for serverless, include the containers for the running list
        if request.type in request.allocated_vm.running_list:
            request.allocated_vm.running_list[request.type].append(request.allocated_cont)
        else:
            request.allocated_vm.running_list[request.type] = []
            request.allocated_vm.running_list[request.type].append(request.allocated_cont)
            # for pod in vm.running_list:
            #     # logging.info(
            #     #     "Pod {} is in running list of vm {}".format(pod.id, vm.id))

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
                print("clock: {} cont {} in vm idle list".format(self.clock, self.clock - cont.idle_start_time))
                print("the spent time idling : {}".format(self.clock, cont.id))
                if (self.clock - cont.idle_start_time) == constants.container_idle_time:
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
                        "results/Results_Episode_" + str(self.episode_no) + ".xls")

    def terminate_ec2_vm(self, vm):
        if vm in self.ec2_vm_idle_list:
            if (self.clock - vm.idle_start_time) == constants.ec2_vm_idle_time:
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

    # def pod_scaler(self):
    #     pod_info = {}
    #     for pod in self.containers:
    #         if pod.type in pod_info:
    #             if pod in self.cont_object_dict_by_type[pod.type]:
    #                 # pod.cpu_util is in no of cores (not %)
    #                 pod_info[pod.type]['pod_cpu_util_total'] += pod.cpu_util
    #                 pod_info[pod.type]['pod_count'] += 1
    #         else:
    #             if pod.type in self.cont_object_dict_by_type:
    #                 if pod in self.cont_object_dict_by_type[pod.type]:
    #                     pod_info[pod.type] = {}
    #                     pod_info[pod.type]['pod_cpu_util_total'] = pod.cpu_util
    #                     pod_info[pod.type]['pod_count'] = 1
    #     # logging.info(
    #     #     "CLOCK: {} worker: {} Now in pod scaler. now running pod count: {} ".format(self.clock, self.worker_id,
    #     #                                                                                 pod_info['pod_count']))
    #
    #     # print("Pod list  details : " + str(PODS))
    #     # print("Pod object by type details : " + str(cont_object_dict_by_type))
    #     # print("Pod info details : "+str(pod_info))
    #
    #     # logging.info(
    #     #     "CLOCK: {} worker: {} CPU util total: {} Pod count: {} Pod cpu req: {} Pod ram req: {}".format(
    #     #         self.clock, self.worker_id, pod_info['pod_cpu_util_total'], pod_info['pod_count'],
    #     #         self.fn_features[str(self.fn_type) + "_pod_cpu_req"], self.fn_features[str(self.fn_type) + "_pod_ram_req"]))
    #     # if pod_data['pod_cpu_util_total'] / pod_data['pod_count'] > constants.pod_scale_cpu_util:
    #     for pod_type, pod_data in pod_info.items():
    #         desired_replicas = math.ceil(pod_data['pod_count'] * (pod_data['pod_cpu_util_total'] / int(
    #             self.fn_features[str(pod_type) + "_pod_cpu_req"])) / pod_data['pod_count'] / self.fn_features[
    #                                          str(pod_type) + "_scale_cpu_threshold"])
    #         logging.info("CLOCK: {} Desired replicas: {} now running pod count: {} Pod type: {}".format(self.clock,
    #                                                                                                     desired_replicas,
    #                                                                                                     pod_data[
    #                                                                                                         'pod_count'],
    #                                                                                                     pod_type))
    #         if desired_replicas > 0:
    #             new_pod_count = min(desired_replicas, int(constants.max_num_replicas))
    #         else:
    #             new_pod_count = 1
    #         # else:
    #         #     new_pod_count = 1
    #
    #         # logging.info("CLOCK: {} worker: {} new pod count: {} _num_max_replicas: {}".format(self.clock, self.worker_id, new_pod_count, int(
    #         #     self.fn_features[str(self.fn_type) + "_num_max_replicas"])))
    #         self.required_pod_count[pod_type] = new_pod_count
    #         if pod_type in self.pending_containers:
    #             pending = self.pending_containers[pod_type]
    #         else:
    #             pending = 0
    #         # logging.info("CLOCK: {} worker: {} No of pending pods of this type: {}".format(self.clock, self.worker_id, pending))
    #         if new_pod_count > pod_data['pod_count'] + pending:
    #             new_pods_to_create = new_pod_count - pod_data['pod_count'] - pending
    #             logging.info(
    #                 "CLOCK: {} fn: {} now running pod count: {} pending pods: {} Scaling up, new pods to create: {}".format(
    #                     self.clock, pod_type, pod_data['pod_count'], self.pending_containers[pod_type],
    #                     new_pods_to_create))
    #             for x in range(new_pods_to_create):
    #                 self.sorted_events.append(
    #                     defs.EVENT(self.clock + constants.container_creation_time, constants.scale_pod, pod_type))
    #                 self.pending_containers[pod_type] += 1
    #
    #         # scaling down of pods if pod has no running requests
    #         if new_pod_count < pod_data['pod_count'] + pending:
    #             pods_to_scale_down = pending + pod_data['pod_count'] - new_pod_count
    #             logging.info(
    #                 "CLOCK: {} Scaling down, num pods to remove: {}".format(self.clock,
    #                                                                                    pods_to_scale_down))
    #             pod_list_length = len(self.cont_object_dict_by_type[pod_type])
    #             removed_pods = 0
    #             pods_to_remove = []
    #             for i in range(pod_list_length):
    #                 pod = self.cont_object_dict_by_type[pod_type][i]
    #                 if pod.num_inflight_requests == 0:
    #                     # with vm_lock:
    #                     # logging.info(
    #                     #     "CLOCK: {} worker: {} Pod Id: {} to be removed, Allocated VM id: {}, vm running pod list: {}".format(self.clock, self.worker_id, pod.id,
    #                     #                                                                                     pod.allocated_vm.id,
    #                     #                                                                                     pod.allocated_vm.running_list))
    #                     # print(PODS)
    #                     self.containers.remove(pod)
    #                     pods_to_remove.append(pod)
    #                     # print(pod.id)
    #                     # print(pod.allocated_vm.id)
    #                     # print(pod.allocated_vm.running_list)
    #                     pod.allocated_vm.running_list[pod.type].remove(pod)
    #
    #                     if len(pod.allocated_vm.running_list[pod.type]) == 0:
    #                         pod.allocated_vm.running_list.pop(pod.type)
    #
    #                     if len(pod.allocated_vm.running_list) == 0:
    #                         self.ec2_vm_up_time_dict[pod.allocated_vm]['status'] = "OFF"
    #                         self.ec2_vm_up_time_dict[pod.allocated_vm]['total_time'] += self.clock - \
    #                                                                                     self.ec2_vm_up_time_dict[
    #                                                                                         pod.allocated_vm][
    #                                                                                         'time_now']
    #                         logging.info(
    #                             "clock: {} vm {} up time so far: {} latest segment: {}".format(self.clock,
    #                                                                                                       pod.allocated_vm.id,
    #                                                                                                       self.ec2_vm_up_time_dict[
    #                                                                                                           pod.allocated_vm][
    #                                                                                                           'total_time'],
    #                                                                                                       self.clock - \
    #                                                                                                       self.ec2_vm_up_time_dict[
    #                                                                                                           pod.allocated_vm][
    #                                                                                                           'time_now']))
    #
    #                         self.ec2_vm_up_time_dict[pod.allocated_vm]['time_now'] = self.clock
    #
    #                     pod.allocated_vm.cpu_allocated -= pod.cpu_req
    #                     pod.allocated_vm.mem_allocated -= pod.ram_req
    #                     pod.term_time = self.clock
    #                     removed_pods += 1
    #
    #                     self.pod_data_sheet.write(self.pod_sheet_row_counter, 0, pod.id)
    #                     self.pod_data_sheet.write(self.pod_sheet_row_counter, 1, pod.type)
    #                     self.pod_data_sheet.write(self.pod_sheet_row_counter, 2, pod.start_time)
    #                     self.pod_data_sheet.write(self.pod_sheet_row_counter, 3, pod.term_time)
    #                     self.pod_sheet_row_counter += 1
    #
    #                     if removed_pods == pods_to_scale_down:
    #                         break
    #             self.wb.save(
    #                 "results/Results_Episode_" + str(self.episode_no) + ".xls")
    #
    #             for p in pods_to_remove:
    #                 logging.info(
    #                     "Clock {} Pod {} of type {} removed from list".format(self.clock,
    #                                                                                      p.id,
    #                                                                                      p.type))
    #                 self.cont_object_dict_by_type[p.type].remove(p)
    #
    #     self.sorted_events = sorted(self.sorted_events, key=self.sorter_events)
    #
    #     for pod in self.containers:
    #         if pod.type in self.cont_object_dict_by_type:
    #             pod.num_current_replicas = len(self.cont_object_dict_by_type[pod.type])
    #
    #     # if pod_data['pod_cpu_util_total'] / pod_data['pod_count'] < constants.pod_scale_cpu_util:
    #     #     desired_replicas = math.ceil((pod_data['pod_cpu_util_total']) / constants.pod_scale_cpu_util)
    #     #     new_pod_count = min(desired_replicas, int(fn_features[str(pod_type) + "_num_max_replicas"]))
    #     #     required_pod_count[pod_type] = new_pod_count
    #     #     if new_pod_count > pod_data['pod_count']:
    #     #         new_pods_to_create = new_pod_count - pod_data['pod_count']
    #     #         for x in range(new_pods_to_create):
    #     #             sorted_events.append(
    #     #                 defs.EVENT(clock + constants.container_creation_time, constants.scale_pod, pod_type))

    # c but vm dict i so
    # def pod_scheduler(self, pod):
    #     vm_allocated = False
    #     if self.episode_no == 4:
    #         print("debug at pod scheduler ")
    #     # with vm_lock:
    #     for vm in self.serverless_vms:
    #         # print("CPU: " + str(vm.cpu) + " CPU allocated: " + str(vm.cpu_allocated))
    #         # print("ram: " + str(vm.ram) + " ram allocated: " + str(vm.mem_allocated))
    #         if ((vm.cpu - vm.cpu_allocated) >= pod.cpu_req) & ((vm.ram - vm.mem_allocated) >= pod.ram_req):
    #             pod.allocated_vm = vm
    #             logging.info(
    #                 " Pod {} type: {} is allocated to VM {}".format( pod.id, pod.type, vm.id))
    #             vm.cpu_allocated += pod.cpu_req
    #             vm.mem_allocated += pod.ram_req
    #             if vm in self.ec2_vm_up_time_dict:
    #                 if self.ec2_vm_up_time_dict[vm]['status'] != "ON":
    #                     # logging.info(
    #                     #     "VM {} is already in ON status so not making changes to dict".format(vm.id))
    #                     # else:
    #                     self.ec2_vm_up_time_dict[vm]['status'] = "ON"
    #                     self.ec2_vm_up_time_dict[vm]['time_now'] = self.clock
    #             else:
    #                 self.ec2_vm_up_time_dict[vm] = {}
    #                 self.ec2_vm_up_time_dict[vm]['status'] = "ON"
    #                 self.ec2_vm_up_time_dict[vm]['time_now'] = self.clock
    #                 self.ec2_vm_up_time_dict[vm]['total_time'] = 0
    #
    #             if pod.type in vm.running_list:
    #                 vm.running_list[pod.type].append(pod)
    #             else:
    #                 vm.running_list[pod.type] = []
    #                 vm.running_list[pod.type].append(pod)
    #             # for pod in vm.running_list:
    #             #     # logging.info(
    #             #     #     "Pod {} is in running list of vm {}".format(pod.id, vm.id))
    #             pod.start_time = self.clock
    #
    #             # print(PODS[len(PODS) - 1])
    #             if pod.type in self.cont_object_dict_by_type:
    #                 self.cont_object_dict_by_type[pod.type].append(pod)
    #             else:
    #                 self.cont_object_dict_by_type[pod.type] = []
    #                 self.cont_object_dict_by_type[pod.type].append(pod)
    #             pod.num_current_replicas = len(self.cont_object_dict_by_type[pod.type])
    #
    #             # if str(pod.type) + "_pod_counter" in pod_counters:
    #             #     pod_counters[str(pod.type) + "_pod_counter"] += 1
    #             # else:
    #             #     pod_counters[str(pod.type) + "_pod_counter"] = 0
    #
    #             if pod.type not in self.pod_counter:
    #                 self.pod_counter[pod.type] = 0
    #
    #             if self.pending_containers[pod.type] > 0:
    #                 self.pending_containers[pod.type] -= 1
    #             vm_allocated = True
    #             break

    # c but event q i so
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
                # request.reschedule = True
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
                        # logging.info(
                        #     "VM {} is already in ON status so not making changes to dict".format(vm.id))
                        # else:
                        self.ec2_vm_up_time_dict[request.allocated_vm]['status'] = "ON"
                        self.ec2_vm_up_time_dict[request.allocated_vm]['time_now'] = self.clock
                else:
                    self.ec2_vm_up_time_dict[request.allocated_vm] = {}
                    self.ec2_vm_up_time_dict[request.allocated_vm]['status'] = "ON"
                    self.ec2_vm_up_time_dict[request.allocated_vm]['time_now'] = self.clock
                    self.ec2_vm_up_time_dict[request.allocated_vm]['total_time'] = 0

                # for serverful include the requests for running list
                if request.type in request.allocated_vm.running_list:
                    request.allocated_vm.running_list[request.type].append(request)
                    print("Adding request {} to running list".format(request.id))
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
                    # self.ec2_vms[request.vm_id].cpu_allocated += int(self.fn_features[str(request.type) + "_cpu_req"])
                    # self.ec2_vms[request.vm_id].mem_allocated += int(self.fn_features[str(request.type) + "_req_ram"])
                else:
                    self.sorted_events.append(
                        defs.EVENT(self.clock + constants.vm_creation_time, constants.create_new_vm,
                                   self.ec2_vms[request.vm_id]))
                    self.ec2_vms[request.vm_id].launch_start_time = self.clock
                    self.pending_vms[self.ec2_vms[request.vm_id]] = []
                    self.pending_vms[self.ec2_vms[request.vm_id]].append(request)
                    # self.ec2_vms[request.vm_id].cpu_allocated += int(self.fn_features[str(request.type) + "_cpu_req"])
                    # self.ec2_vms[request.vm_id].mem_allocated += int(self.fn_features[str(request.type) + "_req_ram"])

            # self.ec2_vm_autoscaler()

        self.sorted_events = sorted(self.sorted_events, key=self.sorter_events)

    def ec2_vm_autoscaler(self):
        cpu_util = 0
        vm_count = 0
        vm_req_sent = 0
        for vm in self.ec2_vms:
            if vm.cpu_allocated != 0:
                cpu_util += vm.cpu_allocated / vm.cpu
                vm_count += 1

        cpu_util = cpu_util / vm_count
        desired_replicas = math.ceil(vm_count * cpu_util / constants.ec2_vm_scale_cpu_util)
        #         logging.info("CLOCK: {} Desired replicas: {} now running pod count: {} Pod type: {}".format(self.clock,
        #                                                                                                     desired_replicas,
        #                                                                                                     pod_data[
        #                                                                                                         'pod_count'],
        #                                                                                                     pod_type))
        if desired_replicas > 0:
            new_vm_count = min(desired_replicas, int(constants.max_num_ec2_vms))
        else:
            new_vm_count = 1

        if new_vm_count > vm_count:
            new_vms_to_create = new_vm_count - vm_count

            for vm in self.ec2_vm_up_time_dict:
                if self.ec2_vm_up_time_dict[vm]['status'] != "ON":
                    self.sorted_events.append(
                        defs.EVENT(self.clock + constants.vm_creation_time, constants.create_new_vm, vm))
                    vm_req_sent += 1
                    vm.launch_start_time = self.clock
                    self.pending_vms[vm] = []

                if vm_req_sent == new_vms_to_create:
                    break
        # elif new_vm_count < vm_count:
        #     vms_to_destroy = vm_count - new_vm_count
        #     for vm in self.ec2_vm_up_time_dict:
        #         if self.ec2_vm_up_time_dict[vm]['status'] == "ON":
        #             if vm.cpu_allocated == 0:

    def req_completion(self, request):

        # logging.info("CLOCK: {} worker: {} Req completion > Request ID: {}".format(self.clock, self.worker_id, request.id))
        request.finish_time = self.clock
        request.status = "Ok"
        self.req_wl.write(self.results_sheet_row_counter, 0, request.id)
        self.req_wl.write(self.results_sheet_row_counter, 1, request.type)
        if request.allocated_cont is not None:
            self.req_wl.write(self.results_sheet_row_counter, 2, int(request.allocated_cont.id))
        self.req_wl.write(self.results_sheet_row_counter, 3, request.arrival_time)
        self.req_wl.write(self.results_sheet_row_counter, 4, request.start_time)
        self.req_wl.write(self.results_sheet_row_counter, 5, request.finish_time)
        self.req_wl.write(self.results_sheet_row_counter, 6, "Ok")

        # with history_lock:
        if request.type in self.sorted_request_history_per_window:
            self.sorted_request_history_per_window[str(request.type)].append(request)
        else:
            self.sorted_request_history_per_window[str(request.type)] = []
            self.sorted_request_history_per_window[str(request.type)].append(request)

        self.results_sheet_row_counter += 1
        self.wb.save("results/Results_Episode_" + str(self.episode_no) + ".xls")

        if request.deployed_env == "s":
            request.allocated_cont.running_req = None
            request.allocated_cont.completed_req_list.append(request)
            request.allocated_cont.cpu_util = 0
            request.allocated_cont.ram_util = 0
            request.allocated_cont.idle_start_time = self.clock
            if request.type in request.allocated_vm.idle_containers:
                request.allocated_vm.idle_containers[request.type].append(request.allocated_cont)
            else:
                request.allocated_vm.idle_containers[request.type] = []
                request.allocated_vm.idle_containers[request.type].append(request.allocated_cont)
            # request.allocated_cont.allocated_vm.idle_containers.append(request.allocated_cont)

            self.sorted_events.append(
                defs.EVENT(self.clock + constants.container_idle_time, constants.terminate_container,
                           request.allocated_cont))
            print("clock: {} Adding cont {} to idle list".format(self.clock, request.allocated_cont.id))
            # with vm_lock:
            self.sorted_events = sorted(self.sorted_events, key=self.sorter_events)
            request.allocated_cont.allocated_vm.cpu_used -= self.fn_features[str(request.type) + "_cpu_req"]
            request.allocated_cont.allocated_vm.ram_used -= self.fn_features[str(request.type) + "_req_ram"]

            # if request.type in self.serverless_cost_dict:
            self.serverless_cost_dict[request.type]['exec_time'] += request.finish_time - request.start_time

            # else:
            #     self.serverless_cost_dict[request.type] = {}
            #     self.serverless_cost_dict[request.type]['mem'] = 0
            #     self.serverless_cost_dict[request.type]['exec_time'] = request.finish_time - request.start_time

        else:
            request.allocated_vm.cpu_used -= self.fn_features[str(request.type) + "_cpu_req"]
            request.allocated_vm.ram_used -= self.fn_features[str(request.type) + "_req_ram"]
            request.allocated_vm.cpu_allocated -= self.fn_features[str(request.type) + "_cpu_req"]
            request.allocated_vm.mem_allocated -= self.fn_features[str(request.type) + "_req_ram"]

            print("Removing request {} from running list".format(request.id))
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
                request.allocated_vm.idle_start_time = self.clock
                self.sorted_events.append(
                    defs.EVENT(self.clock + constants.ec2_vm_idle_time, constants.terminate_ec2_vm,
                               request.allocated_vm))

    #    create objects for each request with type, pod, start time, finish time, dleay in starting

    #                 add pods to a list so that we can use RR

    # need to add an event to wait for xx sec and check again if a suitable pod appears, if not drop the request

    # def gen_serv_env_state(fn_type):

    def calculate_average_arrival_rate(self):
        total_req_count_within_window = 0
        total_rate = 0
        average_rate = 0
        if self.fn_type in self.sorted_request_history_per_window:
            for req in reversed(self.sorted_request_history_per_window[self.fn_type]):
                if req.arrival_time > self.clock - constants.request_rate_window:
                    # print(req.id)
                    # self.logging.info(
                    #     "Req id: {}, arrival time: {}, finish time: {}".format(req.id, req.arrival_time, req.finish_time))
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
        env_state[x + 2] = round(self.fn_request_rate[self.fn_type] / constants.max_request_rate, 2)
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

    # def gen_serv_env_state_min(fn_type):

    # def gen_serv_env_state_max(fn_type):
    # c

    # i
    def init_workload(self):
        print("New workload created for new episode {}".format(self.episode_no % constants.num_wls))
        wbr = xlrd.open_workbook("wl/" + "wl" + str(
            self.episode_no % constants.num_wls) + ".xls")
        sheet = wbr.sheet_by_index(0)
        for i in range(sheet.nrows - 1):
            # for i in range(500):
            fn_name = int(sheet.cell_value(i + 1, 0))
            arr_time = sheet.cell_value(i + 1, 1)
            arr_rate = sheet.cell_value(i + 1, 2)
            idr = sheet.cell_value(i + 1, 3)

            fn_type = "fn" + str(fn_name)
            if not (fn_type in self.fn_types):
                self.fn_types.append(fn_type)
            if not (fn_type in self.fn_request_rate):
                self.fn_request_rate[fn_type] = arr_rate

            if fn_type not in self.serverless_cost_dict:
                self.serverless_cost_dict[fn_type] = {}
                self.serverless_cost_dict[fn_type]['mem'] = self.fn_features[str(fn_type) + "_req_ram"]
                self.serverless_cost_dict[fn_type]['exec_time'] = 0
                # print(self.serverless_cost_dict[fn_type]['exec_time'])

            # self.serverless_cost_dict['mem'] = self.fn_features[str(self.fn_type) + "_req_ram"]

            running_list = {}
            term_list = []
            term_req_list = []
            idle_list = []

            # creating a bogus VM and pod object so that a request object can be initialized
            # vm = defs.VM(0, len(self.serverless_vms), serverless_vm_features['vm0_cpu_total_speed'],
            #              serverless_vm_features['vm0_mem'],
            #              serverless_vm_features['vm0_cpu_used'], serverless_vm_features['vm0_mem_used'],
            #              serverless_vm_features['vm0_cpu_allocated'],
            #              serverless_vm_features['vm0_mem_allocated'], serverless_vm_features['vm0_price'],
            #              serverless_vm_features['vm0_nw_bandwidth'],
            #              serverless_vm_features['vm0_diskio_bandwidth'], running_list, term_list, idle_list, 's')
            # cont = defs.CONTAINER(self.container_id, vm, self.fn_features["fn0" + "_name"], 0, 0,
            #                       self.fn_features["fn0" + "_cpu_req"],
            #                       self.fn_features["fn0" + "_ram_req"],
            #                       0, 0, 0, None, term_req_list, 0)
            # Create a request object
            req_obj = defs.REQUEST(idr, "fn" + str(fn_name), None, None, arr_time, 0, 0, arr_rate, "initial", None, 's',
                                   [])

            # Event list (unsorted list) consists of the event received time, event name and the object associated with the event
            self.sorted_events.append(defs.EVENT(arr_time, constants.schedule_request, req_obj))

        self.sorted_events = sorted(self.sorted_events, key=self.sorter_events)
        # for x in sorted_events:
        #     print(x.event_name)
        print("Request event creation done")
