import logging
from collections import deque
from queue import Queue

from xlwt import Workbook

from env.ServerlessEnv import ServerlessEnv
import constants
import numpy as np
import definitions as defs
import ServEnv_base

MODEL_NAME = "Serverless_Scheduling"

current_episode = 0
total_steps = 0

reward_q = Queue()
state_q = Queue()
action_q = Queue()
done_q = Queue()
threadID_q = Queue()
next_state_q = Queue()
memory = deque(maxlen=2000)
n_step = 5
n_step_buffer = deque(maxlen=n_step)
gamma_nstep = 0.95

epsilon = 1  # exploration rate
epsilon_min = 0.04
epsilon_decay = 0.999
train_start = 100
update_rate = 100


def get_n_step_info(n_s_buffer, gam):
    """Return n step reward, next state, and done."""
    # info of the last transition
    reward, next_state, done = n_s_buffer[-1][-3:]

    for transition in reversed(list(n_s_buffer)[:-1]):
        r, n_s, d = transition[-3:]

        reward = r + gam * reward * (1 - d)
        next_state, done = (n_s, d) if d else (next_state, done)

    return reward, next_state, done


def add_to_buffer(total_s):
    while state_q.qsize() != 0 and reward_q.qsize() != 0 and action_q.qsize() != 0 and next_state_q.qsize() != 0 and done_q.qsize() != 0:
        print("Appending to buffer")
        # memory.append(
        #     (state_q.get(), action_q.get(), reward_q.get(), next_state_q.get(), done_q.get()))

        n_step_buffer.append(
            (state_q.get(), action_q.get(), reward_q.get(), next_state_q.get(), done_q.get()))
        print("n-step buffer length: ", len(n_step_buffer))

        if len(n_step_buffer) == n_step:  # fill the n-step buffer for the first translation
            # add a multi step transition
            reward, next_obs, done = get_n_step_info(n_step_buffer, gamma_nstep)
            obs, action = n_step_buffer[0][:2]

            memory.append((obs, action, reward, next_obs, done))

        # print("memory size : ", len(memory))
        logging.info("CLOCK: {}: memory size : {}".format(ServEnv_base.clock, len(memory)))

    global epsilon
    if len(memory) > train_start:
        if total_s % 2 == 0:
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
                print("changed Epsilon: " + str(epsilon))


def run_episode(env, wbook, sheet, verbose):
    global total_steps

    reward_q.queue.clear()
    state_q.queue.clear()
    action_q.queue.clear()
    done_q.queue.clear()
    next_state_q.queue.clear()
    # .resete()
    sum_reward = 0
    remaining_scaling_steps = True
    env.create_scaling_events()
    step_count = 1

    episode_steps_sheet_row_counter = 1

    while env.simulation_running:
        if env.execute_events():
            write_log = False
            if not env.simulation_running:
                continue
            print("CLOCK: {} Starting step {}:".format(ServEnv_base.clock, step_count))
            current_state, act_t, action, clock = env.select_action(step_count, epsilon)
            logging.info("CLOCK: {}: Action selected: {}".format(ServEnv_base.clock, action))
            # if verbose:
            #     print("action:", action)
            reward_e, done, info = env.take_step(act_t, action, step_count)
            # x = 0
            # y = 1 - x
            # reward = reward_tuple[0] * x + reward_tuple[1] * (1 - x)
            # sum_reward += reward
            if step_count != 1:
                next_state_q.put(current_state)
                next_state = current_state
                reward_q.put(reward_e)
                done_q.put(done)

            print("updating next state")

            state_q.put(current_state)
            action_q.put(action)
            print("updating current state and action")

            add_to_buffer(total_steps)

            if step_count == 1:
                sheet.write(episode_steps_sheet_row_counter, 0, ServEnv_base.clock)
                sheet.write(episode_steps_sheet_row_counter, 1, current_episode)
                sheet.write(episode_steps_sheet_row_counter, 2, step_count)
                sheet.write(episode_steps_sheet_row_counter, 3, np.array_str(current_state))
                sheet.write(episode_steps_sheet_row_counter, 4, str(action))
                sheet.write(episode_steps_sheet_row_counter, 7, done)

            else:
                sheet.write(episode_steps_sheet_row_counter, 0, ServEnv_base.clock)
                sheet.write(episode_steps_sheet_row_counter, 1, current_episode)
                sheet.write(episode_steps_sheet_row_counter, 2, step_count)
                sheet.write(episode_steps_sheet_row_counter, 3, np.array_str(current_state))
                sheet.write(episode_steps_sheet_row_counter, 4, str(action))
                sheet.write(episode_steps_sheet_row_counter - 1, 5, reward_e)
                sheet.write(episode_steps_sheet_row_counter - 1, 6, np.array_str(next_state))
                sheet.write(episode_steps_sheet_row_counter, 7, done)

            wbook.save("drl_steps/DRL_Steps_Episode" + str(current_episode) + ".xls")

            if done:
                print("done @ step {}".format(step_count))
                write_log = True
                env.replay(write_log, memory, train_start)
                continue

            env.replay(write_log, memory, train_start)
            if total_steps % update_rate == 0 and len(memory) > train_start:
                # print("Updating target network")
                logging.info("Updating target network")
                env.update_target_network()

            # print("cumulative reward", sum_reward)

            episode_steps_sheet_row_counter += 1
            step_count += 1
            total_steps += 1

    return True


def main():
    global current_episode
    # first, create the custom environment and run it for one episode
    env = ServerlessEnv()
    env.tensorboard.step = 0

    for ep in range(constants.num_episodes):
        logging.info("Starting episode: {}".format(ep))
        current_episode = ep
        ServEnv_base.episode_no = ep
        wb = Workbook()
        drl_steps = wb.add_sheet('Episode_steps')
        drl_steps.write(0, 0, 'Time')
        drl_steps.write(0, 1, 'Episode')
        drl_steps.write(0, 2, 'Step')
        drl_steps.write(0, 3, 'State')
        drl_steps.write(0, 4, 'Action')
        drl_steps.write(0, 5, 'Reward')
        drl_steps.write(0, 6, 'Next State')
        drl_steps.write(0, 7, 'Done')

        ep_data = wb.add_sheet('Episodes')
        ep_data.write(0, 0, 'Time')
        ep_data.write(0, 1, 'Episode')
        ep_data.write(0, 2, 'Epsilon')
        ep_data.write(0, 3, 'Ep_reward')
        ep_data.write(0, 4, 'Avg_nodes')

        # current_episode = ep + 1
        result = run_episode(env, wb, drl_steps, verbose=True)

        print(
            "episode: {}/{}, e: {:.2}, episodic reward: {}".format(ep, constants.num_episodes, float(epsilon),
                                                                   ServEnv_base.episodic_reward))

        try:
            ep_data.write(ep + 1, 0, ServEnv_base.clock)
            ep_data.write(ep + 1, 1, ep)
            ep_data.write(ep + 1, 2, epsilon)
            ep_data.write(ep + 1, 3, ServEnv_base.episodic_reward)
            wb.save("drl_steps/DRL_Steps_Episode" + str(ep) + ".xls")
            # print("Saved to Episodic data3")

        except Exception as inst:
            # print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args

        # ServEnv_base.episode_no = ep + 1
        env.reset()

        # *******************************************************************

    logging.info("CLOCK: {} now training ended".format(ServEnv_base.clock))
    print("Saving trained model as Serverless_Scheduling %s.h5" % str(ServEnv_base.clock))
    env.save(f"model/{MODEL_NAME}-{ServEnv_base.clock}.h5")



def run_episode_test(env, wbook, sheet, verbose):
    global total_steps

    reward_q.queue.clear()
    state_q.queue.clear()
    action_q.queue.clear()
    done_q.queue.clear()
    next_state_q.queue.clear()
    # .resete()
    sum_reward = 0
    remaining_scaling_steps = True
    env.create_scaling_events()
    step_count = 1

    episode_steps_sheet_row_counter = 1

    while env.simulation_running:
        if env.execute_events():
            write_log = False
            if not env.simulation_running:
                continue
            print("CLOCK: {} Starting step {}:".format(ServEnv_base.clock, step_count))
            current_state, act_t, action, clock = env.select_action_test(step_count, epsilon)
            logging.info("CLOCK: {}: Action selected: {}".format(ServEnv_base.clock, action))
            # if verbose:
            #     print("action:", action)
            reward_e, done, info = env.take_step(act_t, action, step_count)
            # x = 0
            # y = 1 - x
            # reward = reward_tuple[0] * x + reward_tuple[1] * (1 - x)
            # sum_reward += reward
            if step_count != 1:
                next_state_q.put(current_state)
                next_state = current_state
                reward_q.put(reward_e)
                done_q.put(done)

            print("updating next state")

            state_q.put(current_state)
            action_q.put(action)
            print("updating current state and action")

            add_to_buffer(total_steps)

            if step_count == 1:
                sheet.write(episode_steps_sheet_row_counter, 0, ServEnv_base.clock)
                sheet.write(episode_steps_sheet_row_counter, 1, current_episode)
                sheet.write(episode_steps_sheet_row_counter, 2, step_count)
                sheet.write(episode_steps_sheet_row_counter, 3, np.array_str(current_state))
                sheet.write(episode_steps_sheet_row_counter, 4, str(action))
                sheet.write(episode_steps_sheet_row_counter, 7, done)

            else:
                sheet.write(episode_steps_sheet_row_counter, 0, ServEnv_base.clock)
                sheet.write(episode_steps_sheet_row_counter, 1, current_episode)
                sheet.write(episode_steps_sheet_row_counter, 2, step_count)
                sheet.write(episode_steps_sheet_row_counter, 3, np.array_str(current_state))
                sheet.write(episode_steps_sheet_row_counter, 4, str(action))
                sheet.write(episode_steps_sheet_row_counter - 1, 5, reward_e)
                sheet.write(episode_steps_sheet_row_counter - 1, 6, np.array_str(next_state))
                sheet.write(episode_steps_sheet_row_counter, 7, done)

            wbook.save("drl_steps/DRL_Steps_Episode" + str(current_episode) + ".xls")

            if done:
                print("done @ step {}".format(step_count))
                write_log = True
                # env.graphs(write_log)
                # env.replay(write_log, memory, train_start)
                continue

            # env.replay(write_log, memory, train_start)
            # if total_steps % update_rate == 0 and len(memory) > train_start:
            #     # print("Updating target network")
            #     logging.info("Updating target network")
            #     env.update_target_network()

            # print("cumulative reward", sum_reward)

            episode_steps_sheet_row_counter += 1
            step_count += 1
            total_steps += 1

    return True


def test():
    global current_episode
    env = ServerlessEnv()
    env.load("model\Serverless_Scheduling-0.h5")
    env.tensorboard.step = 0
    for ep in range(constants.num_episodes):
        if ep == 1:
            print("debug")
        ServEnv_base.episode_no = ep
        current_episode = ep
        wb = Workbook()
        drl_steps = wb.add_sheet('Episode_steps')
        drl_steps.write(0, 0, 'Time')
        drl_steps.write(0, 1, 'Episode')
        drl_steps.write(0, 2, 'Step')
        drl_steps.write(0, 3, 'State')
        drl_steps.write(0, 4, 'Action')
        drl_steps.write(0, 5, 'Reward')
        drl_steps.write(0, 6, 'Next State')
        drl_steps.write(0, 7, 'Done')

        ep_data = wb.add_sheet('Episodes')
        ep_data.write(0, 0, 'Time')
        ep_data.write(0, 1, 'Episode')
        ep_data.write(0, 2, 'Epsilon')
        ep_data.write(0, 3, 'Ep_reward')
        ep_data.write(0, 4, 'Avg_nodes')
        result = run_episode_test(env, wb, drl_steps, verbose=True)

        print(
            "episode: {}/{}, e: {:.2}, episodic reward: {}".format(ep, constants.num_episodes, float(epsilon),
                                                                   ServEnv_base.episodic_reward))

        try:
            ep_data.write(ep + 1, 0, ServEnv_base.clock)
            ep_data.write(ep + 1, 1, ep)
            ep_data.write(ep + 1, 2, epsilon)
            ep_data.write(ep + 1, 3, ServEnv_base.episodic_reward)
            wb.save("drl_steps/DRL_Steps_Episode" + str(ep) + ".xls")
            # print("Saved to Episodic data3")

        except Exception as inst:
            # print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args

        env.reset()

        # *******************************************************************

    logging.info("CLOCK: {} now training ended".format(ServEnv_base.clock))
        # state = self.env.reset()
        # state = np.reshape(state, [1, self.state_size])
        # done = False
        # i = 0
        # while not done:
        #     self.env.render()
        #     action = np.argmax(self.model.predict(state))
        #     next_state, reward, done, _ = self.env.step(action)
        #     state = np.reshape(next_state, [1, self.state_size])
        #     i += 1
        #     if done:
        #         print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
        #         break


if __name__ == "__main__":
    main()
    # test()
