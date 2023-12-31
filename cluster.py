from sys import maxsize
import numpy as np
import definitions as defs
import constants
import ServEnv_base

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


def init_fn_features():
    fn_features = {'fn0_name': "fn0", 'fn0_pod_cpu_req': 2070, 'fn0_req_MIPS': 1012, 'fn0_pod_ram_req': 300,
                   'fn0_req_ram': 75, 'fn0_req_exec_time': 4.4,
                   'fn0_pod_cpu_util': 0, 'fn0_pod_ram_util': 0, 'fn0_req_per_sec': 10,
                   'fn0_num_current_replicas': 1,
                   'fn0_num_max_replicas': 2, 'fn0_inflight_requests': 0, 'fn0_scale_cpu_threshold': 0.8,
                   'fn1_name': "fn1",
                   'fn1_pod_cpu_req': 2070,
                   'fn1_req_MIPS': 966, 'fn1_pod_ram_req': 300, 'fn1_req_ram': 60, 'fn1_req_exec_time': 4.2,
                   'fn1_pod_cpu_util': 0, 'fn1_pod_ram_util': 0,
                   'fn1_req_per_sec': 10,
                   'fn1_num_current_replicas': 1, 'fn1_num_max_replicas': 2, 'fn1_inflight_requests': 0,
                   'fn1_scale_cpu_threshold': 0.9,
                   'fn2_name': "fn2",
                   'fn2_pod_cpu_req': 1840, 'fn2_req_MIPS': 874, 'fn2_pod_ram_req': 300, 'fn2_req_ram': 80,
                   'fn2_req_exec_time': 3.8,
                   'fn2_pod_cpu_util': 0,
                   'fn2_pod_ram_util': 0,
                   'fn2_req_per_sec': 10, 'fn2_num_current_replicas': 1, 'fn2_num_max_replicas': 2,
                   'fn2_inflight_requests': 0, 'fn2_scale_cpu_threshold': 0.8,
                   'fn3_name': "fn3", 'fn3_pod_cpu_req': 1610, 'fn3_req_MIPS': 828, 'fn3_pod_ram_req': 300,
                   'fn3_req_ram': 75, 'fn3_req_exec_time': 3.6,
                   'fn3_pod_cpu_util': 0, 'fn3_pod_ram_util': 0, 'fn3_req_per_sec': 10,
                   'fn3_num_current_replicas': 1,
                   'fn3_num_max_replicas': 2, 'fn3_inflight_requests': 0, 'fn3_scale_cpu_threshold': 0.8,
                   'fn4_name': "fn4",
                   'fn4_pod_cpu_req': 1840,
                   'fn4_req_MIPS': 920, 'fn4_pod_ram_req': 300, 'fn4_req_ram': 65, 'fn4_req_exec_time': 4,
                   'fn4_pod_cpu_util': 0, 'fn4_pod_ram_util': 0,
                   'fn4_req_per_sec': 10,
                   'fn4_num_current_replicas': 1, 'fn4_num_max_replicas': 2, 'fn4_inflight_requests': 0,
                   'fn4_scale_cpu_threshold': 0.8,
                   'fn5_name': "fn5",
                   'fn5_pod_cpu_req': 460, 'fn5_req_MIPS': 184, 'fn5_pod_ram_req': 300, 'fn5_req_ram': 90,
                   'fn5_req_exec_time': 0.8,
                   'fn5_pod_cpu_util': 0,
                   'fn5_pod_ram_util': 0,
                   'fn5_req_per_sec': 10, 'fn5_num_current_replicas': 1, 'fn5_num_max_replicas': 2,
                   'fn5_inflight_requests': 0, 'fn5_scale_cpu_threshold': 0.8,
                   'fn6_name': "fn6", 'fn6_pod_cpu_req': 2070, 'fn6_req_MIPS': 1035, 'fn6_pod_ram_req': 300,
                   'fn6_req_ram': 85, 'fn6_req_exec_time': 4.5,
                   'fn6_pod_cpu_util': 0, 'fn6_pod_ram_util': 0, 'fn6_req_per_sec': 10,
                   'fn6_num_current_replicas': 1,
                   'fn6_num_max_replicas': 2, 'fn6_inflight_requests': 0, 'fn6_scale_cpu_threshold': 0.8,
                   'fn7_name': "fn7",
                   'fn7_pod_cpu_req': 1840,
                   'fn7_req_MIPS': 897, 'fn7_pod_ram_req': 300, 'fn7_req_ram': 70, 'fn7_req_exec_time': 3.9,
                   'fn7_pod_cpu_util': 0, 'fn7_pod_ram_util': 0,
                   'fn7_req_per_sec': 10,
                   'fn7_num_current_replicas': 1, 'fn7_num_max_replicas': 2, 'fn7_inflight_requests': 0,
                   'fn7_scale_cpu_threshold': 0.8,
                   'fn8_name': "fn8",
                   'fn8_pod_cpu_req': 1610, 'fn8_req_MIPS': 759, 'fn8_pod_ram_req': 300, 'fn8_req_ram': 70,
                   'fn8_req_exec_time': 3.3,
                   'fn8_pod_cpu_util': 0,
                   'fn8_pod_ram_util': 0,
                   'fn8_req_per_sec': 10, 'fn8_num_current_replicas': 1, 'fn8_num_max_replicas': 2,
                   'fn8_inflight_requests': 0, 'fn8_scale_cpu_threshold': 0.8, 'fn9_name': "fn9",
                   'fn9_pod_cpu_req': 690, 'fn9_req_MIPS': 345, 'fn9_pod_ram_req': 300, 'fn9_req_ram': 80,
                   'fn9_req_exec_time': 1.5,
                   'fn9_pod_cpu_util': 0,
                   'fn9_pod_ram_util': 0,
                   'fn9_req_per_sec': 10, 'fn9_num_current_replicas': 1, 'fn9_num_max_replicas': 2,
                   'fn9_inflight_requests': 0, 'fn9_scale_cpu_threshold': 0.8, 'fn10_name': "fn10",
                   'fn10_pod_cpu_req': 2760, 'fn10_req_MIPS': 1288, 'fn10_pod_ram_req': 300,
                   'fn10_pod_cpu_util': 0,
                   'fn10_pod_ram_util': 0,
                   'fn10_req_per_sec': 10, 'fn10_num_current_replicas': 1, 'fn10_num_max_replicas': 2,
                   'fn10_inflight_requests': 0, 'fn10_scale_cpu_threshold': 0.8, 'fn11_name': "fn11",
                   'fn11_pod_cpu_req': 2300, 'fn11_req_MIPS': 1127, 'fn11_pod_ram_req': 300, 'fn11_req_ram': 100,
                   'fn11_req_exec_time': 4.9,
                   'fn11_pod_cpu_util': 0,
                   'fn11_pod_ram_util': 0,
                   'fn11_req_per_sec': 10, 'fn11_num_current_replicas': 1, 'fn11_num_max_replicas': 2,
                   'fn11_inflight_requests': 0, 'fn11_scale_cpu_threshold': 0.8
                   }
    return fn_features


def init_serverless_vms(clock):
    serverless_vms = []
    for i in range(5):  # IOT DEVICE
        serverless_vms.append(defs.VM(0, len(serverless_vms), int(serverless_vm_features['vm0_cpu_total_speed']),
                                  int(serverless_vm_features['vm0_mem']),
                                  int(serverless_vm_features['vm0_cpu_used']),
                                  int(serverless_vm_features['vm0_mem_used']),
                                  int(serverless_vm_features['vm0_cpu_allocated']),
                                  int(serverless_vm_features['vm0_mem_allocated']),
                                  float(serverless_vm_features['vm0_price']),
                                  int(serverless_vm_features['vm0_nw_bandwidth']),
                                  int(serverless_vm_features['vm0_diskio_bandwidth']), {}, {}, {},
                                  "s", 0, 0, clock, 0))
        serverless_vms.append(defs.VM(1, len(serverless_vms), int(serverless_vm_features['vm1_cpu_total_speed']),
                  int(serverless_vm_features['vm1_mem']),
                  int(serverless_vm_features['vm1_cpu_used']),
                  int(serverless_vm_features['vm1_mem_used']),
                  int(serverless_vm_features['vm1_cpu_allocated']),
                  int(serverless_vm_features['vm1_mem_allocated']),
                  float(serverless_vm_features['vm1_price']),
                  int(serverless_vm_features['vm1_nw_bandwidth']),
                  int(serverless_vm_features['vm1_diskio_bandwidth']), {}, {}, {},
                  "s", 0, 0, clock, 0))
        serverless_vms.append(defs.VM(2, len(serverless_vms), int(serverless_vm_features['vm2_cpu_total_speed']),
                  int(serverless_vm_features['vm2_mem']),
                  int(serverless_vm_features['vm2_cpu_used']),
                  int(serverless_vm_features['vm2_mem_used']),
                  int(serverless_vm_features['vm2_cpu_allocated']),
                  int(serverless_vm_features['vm2_mem_allocated']),
                  float(serverless_vm_features['vm2_price']),
                  int(serverless_vm_features['vm2_nw_bandwidth']),
                  int(serverless_vm_features['vm2_diskio_bandwidth']), {}, {}, {},
                  "s", 0, 0, clock, 0))
        serverless_vms.append(defs.VM(3, len(serverless_vms), int(serverless_vm_features['vm3_cpu_total_speed']),
                  int(serverless_vm_features['vm3_mem']),
                  int(serverless_vm_features['vm3_cpu_used']),
                  int(serverless_vm_features['vm3_mem_used']),
                  int(serverless_vm_features['vm3_cpu_allocated']),
                  int(serverless_vm_features['vm3_mem_allocated']),
                  float(serverless_vm_features['vm3_price']),
                  int(serverless_vm_features['vm3_nw_bandwidth']),
                  int(serverless_vm_features['vm3_diskio_bandwidth']), {}, {}, {},
                  "s", 0, 0, clock, 0))

        return serverless_vms


def init_ec2_vms():
    ec2_vms = []
    for i in range(5):
        ec2_vms.append(
            defs.VM(0, len(ec2_vms), int(ec2_vm_features['vm0_cpu_total_speed']),
                    int(ec2_vm_features['vm0_mem']),
                    int(ec2_vm_features['vm0_cpu_used']), int(ec2_vm_features['vm0_mem_used']),
                    int(ec2_vm_features['vm0_cpu_allocated']),
                    int(ec2_vm_features['vm0_mem_allocated']), float(ec2_vm_features['vm0_price']), 0, 0, {}, {},
                    {}, "ns", 0, 0, 0, 0))

        ec2_vms.append(
            defs.VM(1, len(ec2_vms), int(ec2_vm_features['vm1_cpu_total_speed']),
                    int(ec2_vm_features['vm1_mem']),
                    int(ec2_vm_features['vm1_cpu_used']), int(ec2_vm_features['vm1_mem_used']),
                    int(ec2_vm_features['vm1_cpu_allocated']),
                    int(ec2_vm_features['vm1_mem_allocated']), float(ec2_vm_features['vm1_price']), 0, 0, {}, {},
                    {}, "ns", 0, 0, 0, 0))

        ec2_vms.append(
            defs.VM(2, len(ec2_vms), int(ec2_vm_features['vm2_cpu_total_speed']),
                    int(ec2_vm_features['vm2_mem']),
                    int(ec2_vm_features['vm2_cpu_used']), int(ec2_vm_features['vm2_mem_used']),
                    int(ec2_vm_features['vm2_cpu_allocated']),
                    int(ec2_vm_features['vm2_mem_allocated']), float(ec2_vm_features['vm2_price']), 0, 0, {}, {},
                    {}, "ns", 0, 0, 0, 0))

        ec2_vms.append(
            defs.VM(3, len(ec2_vms), int(ec2_vm_features['vm3_cpu_total_speed']),
                    int(ec2_vm_features['vm3_mem']),
                    int(ec2_vm_features['vm3_cpu_used']), int(ec2_vm_features['vm3_mem_used']),
                    int(ec2_vm_features['vm3_cpu_allocated']),
                    int(ec2_vm_features['vm3_mem_allocated']), float(ec2_vm_features['vm3_price']), 0, 0, {}, {},
                    {}, "ns", 0, 0, 0, 0))

    return ec2_vms


def gen_serv_env_state_init():
    env_state = np.zeros(124)
    return env_state


def gen_serv_env_state_min():
    env_state = np.zeros(124)
    return env_state


def gen_serv_env_state_max():
    env_state = [maxsize] * 124
    return env_state
