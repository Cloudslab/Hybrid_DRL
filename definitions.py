class VM:
    def __init__(self, type, vm_id, cpu, mem, cpu_u, mem_u, cpu_allo, mem_allo, price, nw_bandwidth, io_bandwidth, run_list, term_list, idle_list, env, l_s_time, idle_t, st_t, f_t):
        self.type = type
        self.id = vm_id
        self.cpu = cpu
        self.ram = mem
        self.cpu_used = cpu_u
        self.ram_used = mem_u
        self.cpu_allocated = cpu_allo
        self.mem_allocated = mem_allo
        self.price = price
        self.bandwidth = nw_bandwidth
        self.diskio_bw = io_bandwidth
        self.running_list = run_list
        self.term_list = term_list
        self.idle_containers = idle_list
        self.mode = env
        self.launch_start_time = l_s_time
        self.idle_start_time = idle_t
        self.start_time = st_t
        self.finish_time = f_t


class CONTAINER:
    def __init__(self, id, vm_allo, type, start_t, term_t, cpu, mem, cpu_u, ram_u, cur_replicas, run_req, com_list, idle_t):
        self.id = id
        self.allocated_vm = vm_allo
        self.type = type
        self.start_time = start_t
        self.term_time = term_t
        self.cpu_req = cpu
        self.ram_req = mem
        self.cpu_util = cpu_u
        self.ram_util = ram_u
        self.num_current_replicas = cur_replicas
        self.running_req = run_req
        self.completed_req_list = com_list
        self.idle_start_time = idle_t



class REQUEST:
    def __init__(self, id, type, cont_allo, vm, arr_t, st_t, fin_t, rate, stat, vmid, env, action):
        self.id = id
        self.type = type
        self.allocated_cont = cont_allo
        self.allocated_vm = vm
        self.arrival_time = arr_t
        self.start_time = st_t
        self.finish_time = fin_t
        self.arrival_rate = rate
        # self.reschedule_tries = tries
        self.status = stat
        self.vm_id = vmid
        self.deployed_env = env
        self.act = action



class EVENT:
    def __init__(self, time, ev_name, object):
        self.received_time = time
        self.event_name = ev_name
        self.entity_object = object
