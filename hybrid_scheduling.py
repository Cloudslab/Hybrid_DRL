import logging
import sys
import warnings
from collections import deque
from queue import Queue
import random
import gym
import multiprocessing
import threading
import numpy as np
import os
import shutil
# import matplotlib.pyplot as plt
from xlwt import Workbook
from env.ServerlessEnv import ServerlessEnv
# from env.ServerlessEnv import ActorCriticModel
import constants
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Dense, Input, Concatenate, LSTM, Flatten, Reshape
from tensorflow.python.keras.models import Sequential, Model, load_model
# from tensorflow.python.keras.utils import to_categorical
# from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.optimizer_v2.adam import Adam
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow as tf
# from ServEnv_base
import ServEnv_base
from ServEnv_base import Worker
#
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution

tf.compat.v1.disable_eager_execution()
# disable_eager_execution()
# tf.disable_v2_behavior()
# tf.enable_eager_execution()
# warnings.filterwarnings('ignore')

MODEL_NAME = "hybrid_Scheduling"

# state_shape = env.observation_space.shape[0]
# action_shape = env.action_space.shape[0]
# action_bound = [env.action_space.low, env.action_space.high]
num_workers = multiprocessing.cpu_count()
num_episodes = 4000
num_timesteps = 200
global_net_scope = 'Global_Net'
update_global = 100
# gamma = 0.90
beta = 0.01
log_dir = 'logs2'
# global_episode = 0
# best_score = 0
fns = ["fn0", "fn1", "fn2", "fn3"]

reward_q = Queue()
state_q = Queue()
action_q = Queue()
done_q = Queue()
threadID_q = Queue()
next_state_q = Queue()
memory = deque(maxlen=2000)

total_steps = 0
model_save_freq = 50
mode = "vm_only"
run_mode = "comparison"


class Agent:
    def __init__(self, st_size):
        self.state_size = st_size
        self.input_dims = state_size
        self.memory = []
        self.n_tiers = 2
        self.gamma = 0.99
        self.fc1_dims = 150
        self.fc2_dims = 150
        self.s_actor_action_size = 40
        self.first_action_size = 2
        self.first_action_space = [i for i in range(self.first_action_size)]
        self.ENTROPY_LOSS = 5e-3
        self.LOSS_CLIPPING = 0.2  # Only implemented clipping for the surrogate loss, paper said it was best
        self.second_action_sizes = [20, 20]
        self.batch_size = 128
        self.epochs = 50
        # self.fn_type = "fn" + str(fn)

        try:
            f = open('entropy.txt', 'r')
            self.entropy = float(f.readline())
            f.close()
        except:
            self.entropy = 1.0  # exploration rate

        self.entropy_min = 0.01
        self.entropy_decay = 0.99
        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.actor_lr = 0.000001
        self.second_actor_lr = 0.00001
        self.lr_decay = 0.97
        self.critic_lr = 0.000005
        self.episode_no = 0
        self.summary_ep = 0
        self.env = ServerlessEnv()
        if run_mode == "s_actor":
            self.actor, self.policy = self.build_actor()
            self.critic = self.build_critic()
        elif run_mode == "train" or run_mode == "test":
            self.first_actor, self.first_policy = self.build_first_actor()
            self.actor_serverless, self.actor_ec2, self.second_policy = self.build_second_actor()
            self.critic = self.build_critic()
        if run_mode == "test":
            # self.save_dir = "model/model_0_138156.h5"
            # model_path = os.path.join(self.save_dir)
            # print('Loading model from: {}'.format(model_path))
            # self.global_model(tf.convert_to_tensor(self.current_state[None, :], dtype=tf.float32))
            self.first_actor.load_weights(os.path.join("model/first-actor-model.h5"))
            self.first_policy.load_weights(os.path.join("model/first-policy-model.h5"))
            self.actor_serverless.load_weights(os.path.join("model/actor-serverless-model.h5"))
            self.actor_ec2.load_weights(os.path.join("model/actor-ec2-model.h5"))
            self.second_policy.load_weights(os.path.join("model/second-policy-model.h5"))
        # disable_eager_execution()

    def proximal_policy_optimization_loss(self, advantage, old_prediction):
        def loss(y_true, y_pred):
            prob = K.sum(y_true * y_pred, axis=-1)
            old_prob = K.sum(y_true * old_prediction, axis=-1)
            r = prob / (old_prob + 1e-10)
            return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - self.LOSS_CLIPPING,
                                                           max_value=1 + self.LOSS_CLIPPING) * advantage))

        return loss

    # def ppo_loss(y_true, y_pred, oldpolicy_probs, advantages, rewards, values):
    #     newpolicy_probs = y_pred
    #     ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
    #     p1 = ratio * advantages
    #     p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
    #     actor_loss = -K.mean(K.minimum(p1, p2))
    #     critic_loss = K.mean(K.square(rewards - values))
    #     total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(
    #         -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
    #     return total_loss

    def build_second_actor(self):
        state_input = Input(shape=(self.input_dims,))
        advantage = Input(shape=(1,))

        actions_serverless = Input(shape=(self.second_action_sizes[0],))
        actions_ec2 = Input(shape=(self.second_action_sizes[1],))

        # should we use the same input and hidden layers for the branches as mentioned in the paper?
        dense1 = Dense(self.fc1_dims, activation='relu')(state_input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        probs_serverless = Dense(self.second_action_sizes[0], activation='softmax')(dense2)

        dense3 = Dense(self.fc1_dims, activation='relu')(state_input)
        dense4 = Dense(self.fc2_dims, activation='relu')(dense3)
        probs_ec2 = Dense(self.second_action_sizes[1], activation='softmax')(dense4)

        actor_serverless = Model(inputs=[state_input, advantage, actions_serverless], outputs=[probs_serverless])
        # actor_serverless.add_loss(self.proximal_policy_optimization_loss(advantage=advantage, old_prediction=actions_serverless))
        # actor_serverless.compile(optimizer=Adam(lr=self.second_actor_lr))

        actor_serverless.compile(optimizer=Adam(lr=self.second_actor_lr), loss=[self.proximal_policy_optimization_loss(
            advantage=advantage,
            old_prediction=actions_serverless)], experimental_run_tf_function=False)

        actor_ec2 = Model(inputs=[state_input, advantage, actions_ec2], outputs=[probs_ec2])
        # actor_ec2.add_loss(
        #     self.proximal_policy_optimization_loss(advantage=advantage, old_prediction=actions_ec2))
        # actor_ec2.compile(optimizer=Adam(lr=self.second_actor_lr))

        actor_ec2.compile(optimizer=Adam(lr=self.second_actor_lr), loss=[self.proximal_policy_optimization_loss(
            advantage=advantage,
            old_prediction=actions_ec2)], experimental_run_tf_function=False)

        policy = Model(inputs=[state_input], outputs=[probs_serverless, probs_ec2])

        return actor_serverless, actor_ec2, policy

    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_first_actor(self):
        state_input = Input(shape=(self.input_dims,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(self.first_action_size,))
        dense1 = Dense(self.fc1_dims, activation='relu')(state_input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        probs = Dense(self.first_action_size, activation='softmax')(dense2)
        actor = Model(inputs=[state_input, advantage, old_prediction], outputs=[probs])

        # actor.add_loss(self.proximal_policy_optimization_loss(advantage=advantage, old_prediction=old_prediction))
        # actor.compile(optimizer=Adam(lr=self.actor_lr))

        actor.compile(optimizer=Adam(lr=self.actor_lr), loss=[self.proximal_policy_optimization_loss(
            advantage=advantage,
            old_prediction=old_prediction)], experimental_run_tf_function=False)

        policy = Model(inputs=[state_input], outputs=[probs])

        return actor, policy

    def build_actor(self):
        state_input = Input(shape=(self.input_dims,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(self.s_actor_action_size,))
        dense1 = Dense(self.fc1_dims, activation='relu')(state_input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        probs = Dense(self.s_actor_action_size, activation='softmax')(dense2)
        actor = Model(inputs=[state_input, advantage, old_prediction], outputs=[probs])

        # actor.add_loss(self.proximal_policy_optimization_loss(advantage=advantage, old_prediction=old_prediction))
        # actor.compile(optimizer=Adam(lr=self.actor_lr))

        actor.compile(optimizer=Adam(lr=self.actor_lr), loss=[self.proximal_policy_optimization_loss(
            advantage=advantage,
            old_prediction=old_prediction)], experimental_run_tf_function=False)

        policy = Model(inputs=[state_input], outputs=[probs])

        return actor, policy

    # critic: state is input and value of state is output of model
    def build_critic(self):
        state_input = Input(shape=(self.state_size,))
        dense1 = Dense(self.fc1_dims, activation='relu', kernel_initializer='he_uniform')(state_input)
        dense2 = Dense(self.fc1_dims, activation='relu', kernel_initializer='he_uniform')(dense1)
        value = Dense(1, activation='linear', kernel_initializer='he_uniform')(dense2)

        critic = Model(inputs=[state_input], outputs=[value])
        critic.compile(optimizer=Adam(lr=self.critic_lr), loss='mse', experimental_run_tf_function=False)
        return critic

    def get_action_test(self, state):
        action_1 = 0
        action_2 = 0
        p_action_1 = self.first_policy.predict(state)[0]
        action_1 = np.argmax(p_action_1)

        p_action_2 = self.second_policy.predict(state, batch_size=1)
        discarded_action_list = self.env.filtered_unavail_action_list(action_1)

        for a in discarded_action_list:
            p_action_2[action_1][0][a] = 0

        p_action_2[action_1][0] /= np.array(p_action_2[action_1][0]).sum()

        if action_1 == 0:
            action_2 = np.argmax(p_action_2[action_1][0])
            self.env.worker.second_action_total += action_2
            # serverless_actions[np.arange(1), action_2] = 1
            # serverless_actions[action_2] = 1
        else:
            action_2 = np.argmax(p_action_2[action_1][0])
            self.env.worker.second_action_total += action_2
            # ec2_actions[np.arange(1), action_2] = 1


        # advantage = Input(shape=(1,))
        # old_prediction = Input(shape=(self.first_action_size,))
        # actions_serverless = Input(shape=(self.second_action_sizes[0],))
        # actions_ec2 = Input(shape=(self.second_action_sizes[1],))
        # action_1 = np.argmax(self.first_actor.predict([state, advantage, old_prediction]))
        # if action_1 == 0:
        #     action_2 = np.argmax(self.actor_serverless.predict([state, advantage, actions_serverless]))
        # else:
        #     action_2 = np.argmax(self.actor_ec2.predict([state, advantage, actions_ec2]))
        return action_1, action_2

    def get_action_comparison(self):
        if mode == "s_only":
            action_1 = 0
            action_2 = 0
            eligible_vm_list = []
            action2_selected = False
            for vm in self.env.worker.serverless_vms:
                if self.env.worker.fn_type in vm.idle_containers:
                    action_2 = vm.id
                    action2_selected = True
                    break
                elif ((vm.cpu - vm.cpu_allocated) >= self.env.worker.fn_features[
                    str(self.env.worker.fn_type) + "_cpu_req"]) and (
                        (vm.ram - vm.mem_allocated) >= self.env.worker.fn_features[
                    str(self.env.worker.fn_type) + "_req_ram"]):
                    eligible_vm_list.append(vm.id)
            if not action2_selected:
                action_2 = eligible_vm_list[0]

        elif mode == "vm_only":
            action_1 = 1
            lowest_cpu_remaining = 0
            action_2 = 0
            for vm in self.env.worker.ec2_vms:
                print(
                    "serverful: cpu left after allocation in vm {} is {}".format(vm.id, str(vm.cpu - vm.cpu_allocated)))
                print("serverful: mem left after allocation in vm {} is {}".format(vm.id,
                                                                                   str(vm.ram - vm.mem_allocated)))
                if ((vm.cpu - vm.cpu_allocated) >= self.env.worker.fn_features[
                    str(self.env.worker.fn_type) + "_cpu_req"]) and (
                        (vm.ram - vm.mem_allocated) >= self.env.worker.fn_features[
                    str(self.env.worker.fn_type) + "_req_ram"]) and (
                        self.env.worker.ec2_vm_up_time_dict[vm]['status'] == "ON" or vm in self.env.worker.pending_vms):
                    if lowest_cpu_remaining > vm.cpu - vm.cpu_allocated:
                        lowest_cpu_remaining = vm.cpu - vm.cpu_allocated
                        action_2 = vm.id

        return action_1, action_2

    def get_action(self, state):
        p_action_1 = self.first_policy.predict(state)[0]
        action_1 = np.random.choice(self.first_action_space, p=p_action_1)
        self.env.worker.deploy_env_action_total += action_1
        action_1_matrix = np.zeros(self.first_action_size)
        action_1_matrix[action_1] = 1

        # serverless_actions = np.zeros(self.second_action_sizes[0])
        serverless_actions = np.zeros([1, self.second_action_sizes[0]])
        # ec2_actions = np.zeros(self.second_action_sizes[1])
        ec2_actions = np.zeros([1, self.second_action_sizes[1]])
        p_action_2 = self.second_policy.predict(state, batch_size=1)
        discarded_action_list = self.env.filtered_unavail_action_list(action_1)

        for a in discarded_action_list:
            p_action_2[action_1][0][a] = 0

        p_action_2[action_1][0] /= np.array(p_action_2[action_1][0]).sum()

        if action_1 == 0:
            action_2 = np.random.choice(self.second_action_sizes[0], 1, p=p_action_2[action_1][0])[0]
            self.env.worker.second_action_total += action_2
            serverless_actions[np.arange(1), action_2] = 1
            # serverless_actions[action_2] = 1
        else:
            action_2 = np.random.choice(self.second_action_sizes[1], 1, p=p_action_2[action_1][0])[0]
            self.env.worker.second_action_total += action_2
            ec2_actions[np.arange(1), action_2] = 1
            # ec2_actions[action_2] = 1

        return action_1, action_1_matrix, p_action_1, action_2, serverless_actions, ec2_actions, p_action_2

    def checkpoint_models(self):
        self.first_actor.save_weights('model/first-actor-model.h5')
        self.first_policy.save_weights('model/first-policy-model.h5')
        self.critic.save_weights('model/critic-model.h5')
        self.second_policy.save_weights('model/second-policy-model.h5')
        self.actor_serverless.save_weights('model/actor-serverless-model.h5')
        self.actor_ec2.save_weights('model/actor-ec2-model.h5')

    def run_episode_t_c(self, wbook, sheet, episode):
        global total_steps
        states = []
        next_states = []
        action1_matrices = []
        serv_action_matrices = []
        ec2_action_matrices = []
        action1_probs = []
        action2_probs = []
        rewards = []
        step_count = 1

        episode_steps_sheet_row_counter = 1
        done = False

        while self.env.simulation_running:
            if self.env.execute_events():
                write_log = False
                if not self.env.simulation_running:
                    continue
                print("CLOCK: {} Starting step {}:".format(self.env.worker.clock, step_count))
                state_original, current_state, clock = self.env.get_state(step_count)
                if run_mode == "comparison":
                    act1, act2 = self.get_action_comparison()
                elif run_mode == "test":
                    act1, act2 = self.get_action_test(current_state)
                action = [act1, act2]
                logging.info("CLOCK: {}: Action selected: {} and {}".format(self.env.worker.clock, act1, act2))
                # if verbose:
                #     print("action:", action)
                reward_e, s_cost, s_lat = self.env.calculate_reward(step_count, act1, act2)
                self.env.execute_action(act1, act2)
                # x = 0
                # y = 1 - x
                # reward = reward_tuple[0] * x + reward_tuple[1] * (1 - x)
                # sum_reward += reward

                if step_count != 1:
                    rewards.append(reward_e)
                    next_states.append(state_original)

                states.append(state_original)

                print("updating current state and action")

                if step_count == 1:
                    sheet.write(episode_steps_sheet_row_counter, 0, self.env.worker.clock)
                    sheet.write(episode_steps_sheet_row_counter, 1, self.episode_no)
                    sheet.write(episode_steps_sheet_row_counter, 2, step_count)
                    sheet.write(episode_steps_sheet_row_counter, 3, np.array_str(current_state))
                    sheet.write(episode_steps_sheet_row_counter, 4, str(action))
                    sheet.write(episode_steps_sheet_row_counter, 9, done)

                else:
                    sheet.write(episode_steps_sheet_row_counter, 0, self.env.worker.clock)
                    sheet.write(episode_steps_sheet_row_counter, 1, episode)
                    sheet.write(episode_steps_sheet_row_counter, 2, step_count)
                    sheet.write(episode_steps_sheet_row_counter, 3, np.array_str(current_state))
                    sheet.write(episode_steps_sheet_row_counter, 4, str(action))
                    sheet.write(episode_steps_sheet_row_counter - 1, 5, reward_e)
                    sheet.write(episode_steps_sheet_row_counter - 1, 6, s_cost)
                    sheet.write(episode_steps_sheet_row_counter - 1, 7, s_lat)
                    sheet.write(episode_steps_sheet_row_counter - 1, 8, np.array_str(current_state))
                    sheet.write(episode_steps_sheet_row_counter, 9, done)

                wbook.save(
                    "drl_steps/DRL_Steps_Episode" + str(episode) + "_ep" + str(self.env.worker.local_ep) + "_wl" + str(
                        self.env.worker.local_wl) + ".xls")

                episode_steps_sheet_row_counter += 1
                step_count += 1
                total_steps += 1

            # this represents the done stage

        write_log = True
        ############

        # self.env.summary_graphs(self.env.worker.end_episode)
        ###############
        self.env.graphs(write_log, self.episode_no)
        # states.pop()
        # action1_matrices.pop()
        # serv_action_matrices.pop()
        # ec2_action_matrices.pop()
        # action1_probs.pop()
        # action2_probs.pop()
        # # states, act1_mat, s_act_mat, e_act_mat, act1_probs, act2_probs, rewards, next_states = np.array(
        # #     states), np.array(action1_matrices), np.array(serv_action_matrices), np.array(
        # #     ec2_action_matrices), np.array(action1_probs), np.array(action2_probs), np.array(rewards), np.array(
        # #     next_states)
        # states, act1_mat, s_act_mat, e_act_mat, act1_probs, act2_probs, rewards, next_states = states, action1_matrices, serv_action_matrices, ec2_action_matrices, action1_probs, action2_probs, rewards, next_states
        # return states, act1_mat, s_act_mat, e_act_mat, act1_probs, act2_probs, rewards, next_states

    def run_episode(self, wbook, sheet, episode):
        global total_steps
        states = []
        next_states = []
        action1_matrices = []
        serv_action_matrices = []
        ec2_action_matrices = []
        action1_probs = []
        action2_probs = []
        rewards = []
        step_count = 1

        episode_steps_sheet_row_counter = 1
        done = False

        while self.env.simulation_running:
            if self.env.execute_events():
                write_log = False
                if not self.env.simulation_running:
                    continue
                print("CLOCK: {} Starting step {}:".format(self.env.worker.clock, step_count))
                state_original, current_state, clock = self.env.get_state(step_count)
                act1, act1_matrix, act1_p, act2, serv_actions, ec2_actions, act2_p = self.get_action(current_state)
                action = [act1, act2]
                logging.info("CLOCK: {}: Action selected: {} and {}".format(self.env.worker.clock, act1, act2))
                # if verbose:
                #     print("action:", action)
                reward_e, s_cost, s_lat = self.env.calculate_reward(step_count, act1, act2)
                self.env.execute_action(act1, act2)
                # x = 0
                # y = 1 - x
                # reward = reward_tuple[0] * x + reward_tuple[1] * (1 - x)
                # sum_reward += reward

                if step_count != 1:
                    rewards.append(reward_e)
                    next_states.append(state_original)

                states.append(state_original)
                action1_matrices.append(act1_matrix)
                serv_action_matrices.append(serv_actions)
                ec2_action_matrices.append(ec2_actions)
                action1_probs.append(act1_p)
                action2_probs.append(act2_p)
                print("updating current state and action")

                if step_count == 1:
                    sheet.write(episode_steps_sheet_row_counter, 0, self.env.worker.clock)
                    sheet.write(episode_steps_sheet_row_counter, 1, self.episode_no)
                    sheet.write(episode_steps_sheet_row_counter, 2, step_count)
                    sheet.write(episode_steps_sheet_row_counter, 3, np.array_str(current_state))
                    sheet.write(episode_steps_sheet_row_counter, 4, str(action))
                    sheet.write(episode_steps_sheet_row_counter, 9, done)

                else:
                    sheet.write(episode_steps_sheet_row_counter, 0, self.env.worker.clock)
                    sheet.write(episode_steps_sheet_row_counter, 1, episode)
                    sheet.write(episode_steps_sheet_row_counter, 2, step_count)
                    sheet.write(episode_steps_sheet_row_counter, 3, np.array_str(current_state))
                    sheet.write(episode_steps_sheet_row_counter, 4, str(action))
                    sheet.write(episode_steps_sheet_row_counter - 1, 5, reward_e)
                    sheet.write(episode_steps_sheet_row_counter - 1, 6, s_cost)
                    sheet.write(episode_steps_sheet_row_counter - 1, 7, s_lat)
                    sheet.write(episode_steps_sheet_row_counter - 1, 8, np.array_str(current_state))
                    sheet.write(episode_steps_sheet_row_counter, 9, done)

                wbook.save(
                    "drl_steps/DRL_Steps_Episode" + str(episode) + "_ep" + str(self.env.worker.local_ep) + "_wl" + str(
                        self.env.worker.local_wl) + ".xls")

                episode_steps_sheet_row_counter += 1
                step_count += 1
                total_steps += 1

            # this represents the done stage

        write_log = True
        ############

        # self.env.summary_graphs(self.env.worker.end_episode)
        ###############
        self.env.graphs(write_log, self.episode_no)
        states.pop()
        action1_matrices.pop()
        serv_action_matrices.pop()
        ec2_action_matrices.pop()
        action1_probs.pop()
        action2_probs.pop()
        # states, act1_mat, s_act_mat, e_act_mat, act1_probs, act2_probs, rewards, next_states = np.array(
        #     states), np.array(action1_matrices), np.array(serv_action_matrices), np.array(
        #     ec2_action_matrices), np.array(action1_probs), np.array(action2_probs), np.array(rewards), np.array(
        #     next_states)
        states, act1_mat, s_act_mat, e_act_mat, act1_probs, act2_probs, rewards, next_states = states, action1_matrices, serv_action_matrices, ec2_action_matrices, action1_probs, action2_probs, rewards, next_states
        return states, act1_mat, s_act_mat, e_act_mat, act1_probs, act2_probs, rewards, next_states

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        minibatch = np.asarray(minibatch)
        states = minibatch[:, 0]
        actions = minibatch[:, 1]
        probs = minibatch[:, 2]
        rewards = minibatch[:, 3]
        action_arr = minibatch[:, 4]
        next_states = minibatch[:, 5]
        nodes = minibatch[:, 6]
        tiers = minibatch[:, 7]
        policies_second_actor = minibatch[:, 8]
        edge_actions = minibatch[:, 9]
        cloud_actions = minibatch[:, 10]

        states = np.vstack(states)
        actions = np.vstack(actions)
        probs = np.vstack(probs)
        rewards = np.vstack(rewards)
        action_arr = np.vstack(action_arr)
        next_states = np.vstack(next_states)
        edge_actions = np.vstack(edge_actions)
        cloud_actions = np.vstack(cloud_actions)

        critic_value = self.critic.predict(states)
        critic_value_ = self.critic.predict(next_states)

        target = rewards + self.gamma * critic_value_
        advantages = target - critic_value

        cloud_policies = []
        edge_policies = []

        for i in range(0, batch_size):
            cloud_policies.append(np.array(policies_second_actor[0][0]))
            edge_policies.append(np.array(policies_second_actor[0][1]))

        cloud_policies = np.vstack(cloud_policies)
        edge_policies = np.vstack(edge_policies)

        self.critic.train_on_batch(states, target)
        self.first_actor.train_on_batch([states, advantages, probs], action_arr)
        self.actor_edge.train_on_batch([states, advantages, edge_policies], edge_actions)
        self.actor_cloud.train_on_batch([states, advantages, cloud_policies], cloud_actions)

    def run(self):
        # first, create the custom environment and run it for one episode
        # self.env.tensorboard.step = 0

        for ep in range(constants.num_episodes):
            logging.info("Starting episode: {}".format(ep))
            # current_episode = ep
            self.episode_no = ep
            wb = Workbook()
            drl_steps = wb.add_sheet('Episode_steps')
            drl_steps.write(0, 0, 'Time')
            drl_steps.write(0, 1, 'Episode')
            drl_steps.write(0, 2, 'Step')
            drl_steps.write(0, 3, 'State')
            drl_steps.write(0, 4, 'Action')
            drl_steps.write(0, 5, 'Reward')
            drl_steps.write(0, 6, 'Step cost')
            drl_steps.write(0, 7, 'Step latency')
            drl_steps.write(0, 8, 'Next State')
            drl_steps.write(0, 9, 'Done')

            ep_data = wb.add_sheet('Episodes')
            ep_data.write(0, 0, 'Time')
            ep_data.write(0, 1, 'Episode')
            ep_data.write(0, 2, 'Ep_reward')
            ep_data.write(0, 3, 'Avg_nodes')

            if run_mode == "comparison" or run_mode == "test":
                self.run_episode_t_c(wb, drl_steps, ep)

            # current_episode = ep + 1
            else:
                states, action1_matrices, serv_action_matrices, ec2_action_matrices, action1_probs, action2_probs, rewards, next_states = self.run_episode(
                    wb, drl_steps, ep)

                # for x in range(self.epochs):
                #     states = random.sample(states, self.batch_size)
                #     action1_probs = random.sample(action1_probs, self.batch_size)
                #     rewards = random.sample(rewards, self.batch_size)
                #     action1_matrices = random.sample(action1_matrices, self.batch_size)
                #     next_states = random.sample(next_states, self.batch_size)
                #     action2_probs = random.sample(action2_probs, self.batch_size)
                #     serv_action_matrices = random.sample(serv_action_matrices, self.batch_size)
                #     ec2_action_matrices = random.sample(ec2_action_matrices, self.batch_size)
                #
                #     states = np.vstack(states)
                #     action1_probs = np.vstack(action1_probs)
                #     rewards = np.vstack(rewards)
                #     action1_matrices = np.vstack(action1_matrices)
                #     next_states = np.vstack(next_states)
                #     # action2_probs = np.vstack(action2_probs)
                #     serv_action_matrices = np.vstack(serv_action_matrices)
                #     ec2_action_matrices = np.vstack(ec2_action_matrices)
                #
                #     critic_value = self.critic.predict(states)
                #     critic_value_ = self.critic.predict(next_states)
                #
                #     target = rewards + self.gamma * np.transpose(critic_value_)
                #     advantages = target - critic_value
                #
                #     serv_policies = []
                #     ec2_policies = []
                #
                #     for i in range(0, self.batch_size):
                #         serv_policies.append(np.array(action2_probs[0][0]))
                #         ec2_policies.append(np.array(action2_probs[0][1]))
                #
                #     self.critic.train_on_batch(states, target)
                #     self.first_actor.train_on_batch([states, np.transpose(advantages), action1_probs], action1_matrices)
                #     self.actor_serverless.train_on_batch([states, np.transpose(advantages), np.squeeze(np.array(serv_policies))], np.squeeze(serv_action_matrices))
                #     self.actor_ec2.train_on_batch([states, np.transpose(advantages), np.squeeze(np.array(ec2_policies))], np.squeeze(ec2_action_matrices))

                # ***************************************
                # print(np.shape(states))
                critic_value = self.critic.predict(np.vstack(states))
                critic_value_ = self.critic.predict(np.vstack(next_states))

                target = rewards + self.gamma * np.transpose(critic_value_)
                advantages = target - np.transpose(critic_value)
                # advantages = np.reshape(advantages, [1, len(advantages)])
                # print(np.shape(advantages))
                # print(np.shape(action1_probs))
                serv_policies = []
                ec2_policies = []

                for i in range(0, len(states)):
                    serv_policies.append(np.array(action2_probs[0][0]))
                    ec2_policies.append(np.array(action2_probs[0][1]))

                actor1_loss = self.first_actor.fit([states, np.transpose(advantages), action1_probs],
                                                   [action1_matrices],
                                                   batch_size=self.batch_size, shuffle=True,
                                                   epochs=self.epochs, verbose=False)
                # actor1_loss = self.first_actor.fit([states, np.transpose(advantages), action1_probs], [action1_matrices],
                #                                    batch_size=self.batch_size, shuffle=True,
                #                                    callbacks=[self.env.tensorboard],
                #                                    epochs=self.epochs, verbose=False)
                serverless_actor_loss = self.actor_serverless.fit(
                    [states, np.transpose(advantages), np.squeeze(np.array(serv_policies))],
                    np.squeeze(serv_action_matrices), batch_size=self.batch_size,
                    shuffle=True,
                    epochs=self.epochs, verbose=False)
                ec2_actor_loss = self.actor_ec2.fit(
                    [states, np.transpose(advantages), np.squeeze(np.array(ec2_policies))],
                    np.squeeze(ec2_action_matrices), batch_size=self.batch_size,
                    shuffle=True,
                    epochs=self.epochs, verbose=False)
                critic_loss = self.critic.fit([states], [rewards], batch_size=self.batch_size, shuffle=True,
                                              epochs=self.epochs,
                                              verbose=False)

                # ******************************************************
                # enable_eager_execution()

            print(
                "episode: {}/{}, episodic reward: {}".format(ep, constants.num_episodes,
                                                             self.env.worker.episodic_reward))

            if (ep + 1) % model_save_freq == 0:
                self.checkpoint_models()

            # try:
            #     ep_data.write(ep + 1, 0, self.env.worker.clock)
            #     ep_data.write(ep + 1, 1, ep)
            #     ep_data.write(ep + 1, 2, self.env.worker.episodic_reward)
            #     wb.save("drl_steps/DRL_Steps_Episode" + str(ep) + ".xls")
            #     # print("Saved to Episodic data3")
            #
            # except Exception as inst:
            #     # print(type(inst))  # the exception instance
            #     print(inst.args)  # arguments stored in .args

            # ServEnv_base.episode_no = ep + 1
            self.env.reset()
            if self.second_actor_lr > self.actor_lr:
                self.second_actor_lr = self.second_actor_lr * self.lr_decay

            # *******************************************************************

        logging.info("CLOCK: {} now training ended".format(self.env.worker.clock))
        print("Saving trained model")
        self.checkpoint_models()


if __name__ == "__main__":
    state_size = 124
    action_size = 2
    fn_id = 0
    agent = Agent(state_size)
    agent.run()
    # test()
