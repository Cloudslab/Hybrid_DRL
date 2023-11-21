max_num_serverless_vms = 20
max_num_ec2_vms = 20

max_num_replicas = 40
max_total_pod_cpu = 8
max_cont_mem_req = 3000
min_pod_mem_req = 100
max_cont_cpu_req = 920  # (1 core of type 2.3 GHz)
min_cont_cpu_req = 460  # ( 5% of a core)

max_vm_mem = 32000
max_vm_cpu = 73600

max_request_rate = 100
max_step_latency_perfn = 10
# 1 earlier
max_step_vmcost = 0.01
max_containers_in_vm = 1000
# desired util
pod_scale_cpu_util = 0.5
ec2_vm_scale_cpu_util = 0.3
pod_scale_cpu_util_low = 0.2
pod_scale_cpu_util_high = 1
# no of requests that can be served concurrently in a pod. We assume this to be the value when a pod requests 920 m cpu. Thus when it claims more, the concurrency is scaled proportionately
pod_size_concurrency = 4
# did this based on above statement
MIPS_for_one_request = 230
# in the same way
ram_for_one_request = 300 / 4
max_reschedule_tries = 4

# if a pod is not available, the wait time for a request to retry scheduling, in seconds
req_wait_time_to_schedule = 0.2

# changed from 2
# lambda cold start delay 500 ms
# cold start time around 500ms normally in average
# 1 sec changed to 1/6 > roughly 0.2
container_creation_time = 0.2
# was 2 sec
container_idle_time = 0.3
# vm creation 60-75 sec
# 60 sec chnaged to 60/6 > 10
# accordingly in the paper, the total wl duration, fn exec times and container vm creation times need to be multiplied byt his number. keep fn rates the same
vm_creation_time = 5
# was 50 sec
ec2_vm_idle_time = 10
serverless_mbs_price = 0.0000000166667
serverless_price_per_request = 0.0000002
pod_scheduling_time = 0
WL_duration = 50
step_interval = 4

# changed from 3
reward_window_size = 1
state_latency_window = 1
request_rate_window = 4

# Events tags

schedule_request = "SCHEDULE_REQ"
re_schedule_request = "RE_SCHEDULE_REQ"
finish_request = "REQ_COMPLETE"
create_new_cont = "CREATE_CONT"
create_new_vm = "CREATE_VM"
scale_pod = "SCALE_POD"
schedule_pod = "SCHEDULE_POD"
invoke_step_scaling = "STEP_SCALING"
terminate_container = "KILL_CONT"
terminate_ec2_vm = "KILL_VM"
num_episodes = 2
num_wls = 1
