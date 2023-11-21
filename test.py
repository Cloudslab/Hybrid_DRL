import sys

import random

import numpy as np

# from ipython_genutils.py3compat import xrange
# from pygments import formatter
#
# import definitions as df
#
# # # Creating an empty dictionary
# # myDict = {}
# #
# # # Adding list as value
# # myDict["key1"] = [1, 2]
# #
# # # # creating a list
# # lst = ['Geeks', 'For', 'Geeks']
# # print(lst[0])
# # #
# # # # Adding this list as sublist in myDict
# # myDict = {}
# # myDict["key1"] = []
# # myDict["key1"].append('Geeks')
# # myDict["key1"].append('For')
# # print(myDict["key1"][0])
#
#
# # x = 3.3
# # print(round(x))
#
#
# # myDict["key1"] = [1]
# # # myDict["key1"].append(100)
# # # print(myDict)
# # #
# # # f1 = 100
# # # i = 1
# # # print("f" + str(i))
# #
# # for i in range(1,5):
# #     print(i)
# #
# # from sortedcontainers import SortedList, SortedSet, SortedDict
# #
# # # initializing a sorted list with parameters
# # # it takes an iterable as a parameter.
# # sorted_list = SortedList([1, 2, 3, 4])
# #
# # # initializing a sorted list using default constructor
# # sorted_list = SortedList()
# #
# # # inserting values one by one using add()
# # for i in range(5, 0, -1):
# #     sorted_list.add(i)
# #
# # # prints the elements in sorted order
# # print('list after adding 5 elements: ', sorted_list)
# #
# # global events
# #
# # events= []
# # events.append(df.EVENT(50, "fn_exec"))
# # events.append(df.EVENT(25, "scale"))
# # events.append(df.EVENT(70, "fn_exec"))
# #
# # def sorter(item):
# #     time = item.time
# #     return time
# #
# # sorted_list = sorted(events, key=sorter)
# #
# # for ob in sorted_list:
# #     print(ob.event_name + " " + str(ob.time))
#
# # Nested list of student's info in a Science Olympiad
# # List elements: (Student's Name, Marks out of 100 , Age)
# # participant_list = [
# #     ('Alison', 50, 18),
# #     ('Terence', 75, 12),
# #     ('David', 75, 20),
# #     ('Jimmy', 90, 22),
# #     ('John', 45, 12)
# # ]
# #
# #
# # def sorter(item):
# #     # Since highest marks first, least error = most marks
# #     error = 100 - item[1]
# #     age = item[2]
# #     return (error, age)
# #
# #
# # sorted_list = sorted(participant_list, key=sorter)
# # print(sorted_list)
#
# # pod_info = {}
# # PODS = []
# #
# # PODS.append('a')
# # PODS.append('b')
# # PODS.append('c')
# #
# # PODS.pop(0)
# # print(PODS)
# # for x in range(3):
# #     if 'float' in pod_info:
# #         pod_info['float']['pod_cpu_util_total'] += 2
# #         pod_info['float']['pod_count'] += 1
# #
# #     else:
# #         pod_info['float'] = {}
# #         pod_info['float']['pod_cpu_util_total'] = 2
# #         pod_info['float']['pod_count'] = 1
# #
# #
# # for pod_type, pod_data in pod_info.items():
# #     print(pod_data['pod_cpu_util_total'])
# #     print(pod_data['pod_count'])
# #     print(pod_type)
#
#
# # for x in range(3):
# #     pod_info['float'] = 2
# #     pod_info['load'] = 2
# #
# #
# # print(pod_info)
#
#
# # people = {}
# #
# # people[3] = {}
# #
# # people[3]['name'] = 'Luna'
# # people[3]['age'] = '24'
# # people[3]['sex'] = 'Female'
# # people[3]['married'] = 'No'
# #
# # print(people[3])
#
# # for i in range(40):
# #     print(str(i % 20))
#
# # x = {}
# # ty = "fn"
# # x["fn"] = []
# # x[ty].append("cd")
# #
# # if ty in x:
# #     print("Y")
#
#
# import threading
# import logging
# import logging.config
#
#
# class ThreadLogFilter(logging.Filter):
#     """
#     This filter only show log entries for specified thread name
#     """
#
#     def __init__(self, thread_name, *args, **kwargs):
#         logging.Filter.__init__(self, *args, **kwargs)
#         self.thread_name = thread_name
#
#     def filter(self, record):
#         return record.threadName == self.thread_name
#
#
# def start_thread_logging():
#     """
#     Add a log handler to separate file for current thread
#     """
#     thread_name = threading.Thread.getName(threading.current_thread())
#     log_file = 'perThreadLogging-{}.log'.format(thread_name)
#     log_handler = logging.FileHandler(log_file)
#
#     log_handler.setLevel(logging.DEBUG)
#
#     formatter = logging.Formatter(
#         "%(asctime)-15s"
#         "| %(threadName)-11s"
#         "| %(levelname)-5s"
#         "| %(message)s")
#     log_handler.setFormatter(formatter)
#
#     log_filter = ThreadLogFilter(thread_name)
#     log_handler.addFilter(log_filter)
#
#     logger = logging.getLogger()
#     logger.addHandler(log_handler)
#
#     return log_handler
#
#
# def stop_thread_logging(log_handler):
#     # Remove thread log handler from root logger
#     logging.getLogger().removeHandler(log_handler)
#
#     # Close the thread log handler so that the lock on log file can be released
#     log_handler.close()
#
#
# def worker():
#     thread_log_handler = start_thread_logging()
#     logging.info('Info log entry in sub thread.')
#     logging.debug('Debug log entry in sub thread.')
#     stop_thread_logging(thread_log_handler)
#
#
# # def config_root_logger():
# #     log_file = '/tmp/perThreadLogging.log'
# #
# #     formatter = "%(asctime)-15s"
# #                 "| %(threadName)-11s"
# #                 "| %(levelname)-5s"
# #                 "| %(message)s"
# #
# #     logging.config.dictConfig({
# #         'version': 1,
# #         'formatters': {
# #             'root_formatter': {
# #                 'format': formatter
# #             }
# #         },
# #         'handlers': {
# #             'console': {
# #                 'level': 'INFO',
# #                 'class': 'logging.StreamHandler',
# #                 'formatter': 'root_formatter'
# #             },
# #             'log_file': {
# #                 'class': 'logging.FileHandler',
# #                 'level': 'DEBUG',
# #                 'filename': '/tmp/perThreadLogging.log',
# #                 'formatter': 'root_formatter',
# #             }
# #         },
# #         'loggers': {
# #             '': {
# #                 'handlers': [
# #                     'console',
# #                     'log_file',
# #                 ],
# #                 'level': 'DEBUG',
# #                 'propagate': True
# #             }
# #         }
# #     })
#
#
# if __name__ == '__main__':
#     # config_root_logger()
#
#     logging.info('Info log entry in main thread.')
#     logging.debug('Debug log entry in main thread.')
#
#     for i in xrange(3):
#         t = threading.Thread(target=worker,
#                              name='Thread-{}'.format(i),
#                              args=[])
#         t.start()


# p = {"a": 0.2, "b": 0.1, "c": 0.4, "d": [0, 2]}
# # p.pop("b")
#
# if "b" in p:
#     print(p)
#
# print(len(p))
# p = np.array(p)
#
# p /= p.sum()
# print(p)
# print(p.sum())


# print(sys.maxsize)

# print(str(random.randint(1,899999)+100000))

#
# import numpy as np
# logits_action = [[1, 2,3, 4, 5, 6, 7, 8, 9, 10]]
# print(np.shape(logits_action))
# print(np.shape(np.transpose(logits_action)))
# action_size = [2, 5, 3]
# logits_c = []
# logits_m = []
# logits_r = []
# for x in range(0, action_size[0]):
#     logits_c.append(logits_action[x])
#
# for x in range(action_size[0], (action_size[0]+action_size[1])):
#     logits_m.append(logits_action[x])
#
# for x in range((action_size[0]+action_size[1]), (action_size[0]+action_size[1] + action_size[2])):
#     logits_r.append(logits_action[x])
#
# print(logits_c)
# print(logits_m)
# print(logits_r)
#
# fn_types = [0, 1, 2]
#
# if not (3 in fn_types):
#     print("yes")

# fn_array = {0: [{1: 10}, {2:15}], 1: [{4: 15}, {6:9}]}
# print(list(fn_array[0])[-1])
#
# fn_array[3] = [7, 8]
# print(fn_array)

# candles = [3, 2, 1, 3]
#
# candles = sorted(candles)
# count = 0
# for x in range(len(candles)-1,-1, -1):
#     if candles[x]==candles[len(candles) - 1]:
#         count = count + 1
#
# print(count)

# !/bin/python3

import math
import os
import random
import re
import sys


#
# Complete the 'jumpingRooks' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER k
#  2. STRING_ARRAY board
#
#

arr = [2, 2, 4, 4, 5, 8]
temp_arr = arr
lengths = []
while len(arr) > 0:
    lengths.append(len(arr))
    smallest = arr[0]
    for x in range(0, len(arr)):
        print(x)
        print(arr)
        arr[x] = arr[x] - smallest
        temp_arr = arr
        print(arr)
        if arr[x] == 0:
            temp_arr.remove(0)
        print(arr)
    print(temp_arr)
    arr = temp_arr

# def permutation(lst):
#     print("Now permutation of " + str(lst))
#     # If lst is empty then there are no permutations
#     if len(lst) == 0:
#         return []
#
#     # If there is only one element in lst then, only
#     # one permutation is possible
#     if len(lst) == 1:
#         return [lst]
#
#     # Find the permutations for lst if there are
#     # more than 1 characters
#
#     l = []  # empty list that will store current permutation
#
#     # Iterate the input(lst) and calculate the permutation
#     for i in range(len(lst)):
#         m = lst[i]
#
#         # Extract lst[i] or m from the list.  remLst is
#         # remaining list
#         remLst = lst[:i] + lst[i + 1:]
#
#         # Generating all permutations where m is first
#         # element
#         for p in permutation(remLst):
#             l.append([m] + p)
#             print("Now l: " + str(l))
#     return l
#
#
# # Driver program to test above function
# data = list('123')
# for p in permutation(data):
#     print(p)
# k=10
# board = [[".", ".", "#",".","."], [".", ".", "#",".","."], ["#", "#", "#","#","#"] , [".", ".", "#",".","."] , [".", ".", "#",".","."]]
# count = 0
# pair = 0
# for row in range(len(board)):
#     for column in range(len(board[row])):
#         if not board[row][column] == "#":
#             if (row - 1) < 0:
#                 if (column - 1) < 0:
#                     board[row][column] = "R"
#                     count += 1
#                     if count == k:
#                         break
#                 else:
#                     for col in range(column - 1, -1, -1):
#                         if board[row][col] == "R":
#                             break
#                         elif board[row][col] == "#":
#                             board[row][column] = "R"
#                             count += 1
#                             if count == k:
#                                 break
#             elif (column - 1) < 0:
#
#                 for r in range(row - 1, -1, -1):
#                     if board[r][column] == "R":
#                         break
#                     elif board[r][column] == "#":
#                         board[row][column] = "R"
#                         count += 1
#                         if count == k:
#                             break
#             else:
#                 cannot = False
#                 for col in range(column - 1, -1, -1):
#                     if board[row][col] == "R":
#                         cannot = True
#                         break
#                     elif board[row][col] == "#":
#                         break
#                 if not cannot:
#                     for r in range(row - 1, -1, -1):
#                         if board[r][column] == "R":
#                             cannot = True
#                             break
#                         elif board[r][column] == "#":
#                             break
#                     if not cannot:
#                         board[row][column] = "R"
#                         count += 1
#                         if count == k:
#                             break
#         print("row " + str(row))
#         print("column " + str(column))
#         print(board)
#
#     if count == k:
#         break
#
# if count == k:
#     print(pair)
# elif count < k:
#     for row in range(len(board)):
#         for column in range(len(board[row])):
#             if board[row][column] != "#" and board[row][column] != "R":
#
#                 board[row][column] = "R"
#                 count += 1
#                 pair += 2
#                 if count == k:
#                     break
#         if count == k:
#             break
#     print(pair)

    # Write your code here


# def jumpingRooks(k, board):
#     count = 0
#     pair = 0
#     for row in range(len(board)):
#         for column in range(len(board[row])):
#             if not board[row][column] == "#":
#                 if (row - 1) < 0:
#                     if (column - 1) < 0:
#                         board[row]= board[row][:column] + "R" + board[row][column+1:]
#                         count += 1
#                         if count == k:
#                             break
#                     else:
#                         for col in range(column - 1, -1, -1):
#                             if board[row][col] == "R":
#                                 break
#                             elif board[row][col] == "#":
#                                 board[row]= board[row][:column] + "R" + board[row][column+1:]
#                                 count += 1
#                                 if count == k:
#                                     break
#                 elif (column - 1) < 0:
#
#                     for r in range(row - 1, -1, -1):
#                         if board[r][column] == "R":
#                             break
#                         elif board[r][column] == "#":
#                             board[row]= board[row][:column] + "R" + board[row][column+1:]
#                             count += 1
#                             if count == k:
#                                 break
#                 else:
#                     cannot = False
#                     for col in range(column - 1, -1, -1):
#                         if board[row][col] == "R":
#                             cannot = True
#                             break
#                         elif board[row][col] == "#":
#                             break
#                     if not cannot:
#                         for r in range(row - 1, -1, -1):
#                             if board[r][column] == "R":
#                                 cannot = True
#                                 break
#                             elif board[r][column] == "#":
#                                 break
#                         if not cannot:
#                             board[row]= board[row][:column] + "R" + board[row][column+1:]
#                             count += 1
#                             if count == k:
#                                 break
#
#         if count == k:
#             break
#
#     if count == k:
#         print(pair)
#     elif count < k:
#         for row in range(len(board)):
#             for column in range(len(board[row])):
#                 if board[row][column] != "#" and board[row][column] != "R":
#
#                     board[row][column] = "R"
#                     count += 1
#                     pair += 2
#                     if count == k:
#                         break
#             if count == k:
#                 break
#         print(pair)
#
#     # Write your code here
#
#
# if __name__ == '__main__':
#     fptr = open(os.environ['OUTPUT_PATH'], 'w')
#
#     first_multiple_input = input().rstrip().split()
#
#     n = int(first_multiple_input[0])
#
#     k = int(first_multiple_input[1])
#
#     board = []
#
#     for _ in range(n):
#         board_item = input()
#         board.append(board_item)
#
#     result = jumpingRooks(k, board)
#
#     fptr.write(str(result) + '\n')
#
#     fptr.close()
