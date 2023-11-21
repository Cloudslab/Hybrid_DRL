import math
import random
import csv

import xlrd
import xlwt
from xlwt import Workbook

fn_array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# arrivalTime = 1
interArrivalTime = 0
noRateChanges = 4
wl_duration = 600
episodes = 4
#duration that one rate for a fn prevails
rate_interval = 20
poisson_lambda = 0
probability = 0
no_steps = 10
step_duration = 6
wl_stop_time = 3


wb = Workbook()
req_wl = wb.add_sheet('Requests')
req_wl.write(0, 0, 'Fn_type')
req_wl.write(0, 1, 'Time')
req_wl.write(0, 2, 'Rate')
req_wl.write(0, 3, 'Req_ID')

# filename = ("D:\WL generation\Third_work\WL.csv")
# f = open(filename, "wrread from+")
# writer = csv.writer(f)
c = 1
req_id = 1

#when having multiplw rate changes for a fn in one file
# for z in range(3):
#
#     #ini_arrivalTime = 1
#     # noArrivals = 40
#     # poisson_lambda = 8
#     # print(fn_array[x])
#     for x in range(noRateChanges):
#         ini_arrivalTime = arrivalTime
#         poisson_lambda = random.randint(1, 5)
#         while arrivalTime < ini_arrivalTime + rate_interval:
#             probability = random.random()
#
#             req_wl.write(c, 0, fn_array[z])
#             req_wl.write(c, 1, float(arrivalTime))
#             req_wl.write(c, 2, int(poisson_lambda))
#             req_wl.write(c, 3, int(req_id))
#             wb.save("D:\WL generation\Third_work\WL1.xls")
#             req_id += 1
#             c += 1
#             interArrivalTime = round((-math.log(1.0 - probability) / poisson_lambda), 2)
#             arrivalTime = round(arrivalTime + interArrivalTime, 2)
#
#         arrivalTime += 20
#     arrivalTime = random.randint(1, 5)



#creating separate episodes

# fn = fn_array[0][0]

# for fn in fn_array:
fn = fn_array[7]
for y in range(2, episodes):
    wb = Workbook()
    req_wl = wb.add_sheet('Requests')
    req_wl.write(0, 0, 'Fn_type')
    req_wl.write(0, 1, 'Time')
    req_wl.write(0, 2, 'Rate')
    req_wl.write(0, 3, 'Req_ID')
    c = 1
    req_id = 1
    # fn_list = []
    # while len(fn_list) < 3:
    #     value = random.choice(fn_array)
    #     if value not in fn_list:
    #         fn_list.append(value)
    # for fn in fn_list:
    total_time = 1
    arrivalTime = 1
    # one time step > 5 sec
    while total_time < wl_duration:
        duration = random.randint(5, 120)
        poisson_lambda = random.randint(30, 60)
        x = 1
        while x < duration:
            req_wl.write(c, 0, fn)
            req_wl.write(c, 1, float(arrivalTime))
            req_wl.write(c, 2, int(poisson_lambda))
            req_wl.write(c, 3, int(req_id))
            wb.save("D:/WL generation/fourth work/" + str(fn) + "/wl" + str(y) + ".xls")
            req_id += 1
            c += 1
            probability = random.random()
            interArrivalTime = round((-math.log(1.0 - probability) / poisson_lambda), 2)
            arrivalTime = round(arrivalTime + interArrivalTime, 2)
            x += interArrivalTime
            total_time += interArrivalTime



    #ini_arrivalTime = 1
    # noArrivals = 40
    # poisson_lambda = 8
    # print(fn_array[x])
    # for x in range(noRateChanges):
    #     ini_arrivalTime = arrivalTime
    #     poisson_lambda = random.randint(1, 5)
    #     while arrivalTime < ini_arrivalTime + rate_interval:
    #         probability = random.random()
    #
    #         req_wl.write(c, 0, fn_array[z])
    #         req_wl.write(c, 1, float(arrivalTime))
    #         req_wl.write(c, 2, int(poisson_lambda))
    #         req_wl.write(c, 3, int(req_id))
    #         wb.save("D:\WL generation\Third_work\WL1.xls")
    #         req_id += 1
    #         c += 1
    #         interArrivalTime = round((-math.log(1.0 - probability) / poisson_lambda), 2)
    #         arrivalTime = round(arrivalTime + interArrivalTime, 2)
    #
    #     arrivalTime += 20
    # arrivalTime = random.randint(1, 5)

    # for x in range(noRateChanges):
    #     # poisson_lambda = random.randint(5, 10)
    #     poisson_lambda = random.randint(1, 5)
    #     for y in range(noArrivals):
    #         probability = random.random()
    #         interArrivalTime = round((-math.log(1.0 - probability) / poisson_lambda), 2)
    #         arrivalTime = round(arrivalTime + interArrivalTime, 2)
    #
    #         req_wl.write(c, 0, fn_array[z])
    #         req_wl.write(c, 1, float(arrivalTime))
    #         req_wl.write(c, 2, int(poisson_lambda))
    #         req_wl.write(c, 3, int(req_id))
    #         wb.save("D:\WL generation\Third_work\WL.xls")
    #         req_id += 1
    #         c += 1
    #
    #     arrivalTime += 20

# f.close()

#
# wb = xlrd.open_workbook("D:\WL generation\Third_work\WL.xls")
# sheet = wb.sheet_by_index(0)
# print(sheet.nrows)
# for i in range(sheet.nrows):
#     print(i)
#     fn_name = sheet.cell_value(i + 1, 0)
#     arr_time = sheet.cell_value(i + 1, 1)
#     print(fn_name)
#     print(arr_time)
