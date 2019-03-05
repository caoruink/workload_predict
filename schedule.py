# -*- coding: utf-8 -*-
# 利用预测得到的数据进行资源调度，选择合适的结合在一起
import numpy as np


def over_limit(temp):
    global limit
    if temp['cpu'] > limit['cpu']:
        return True
    elif temp['mem'] > limit['mem']:
        return True
    elif temp['disk'] > limit['disk']:
        return True
    else:
        return False


def add_host(host1, host2):
    return {"cpu": host1["cpu"] + host2["cpu"], "mem": host1["mem"] + host2["mem"],
            "disk": host1["disk"] + host2["disk"]}


def add_host_r(host1, host2):
    return {"cpu": host1["cpu_r"] + host2["cpu_r"], "mem": host1["mem_r"] + host2["mem_r"],
            "disk": host1["disk_r"] + host2["disk_r"]}


if __name__ == "__main__":
    file = open("实验结果//max_3//predict.csv")
    contents = file.readlines()
    total_number_host = len(contents)
    limit = {"cpu": 80, "mem": 80, "disk": 75}
    tasks = []                           # 记录调度的任务
    num_error_schedule = 0
    for each in contents:
        line = np.array(each.split(','), float)
        tasks.append({"cpu": line[0], "mem": line[1] * 0.9, "disk": line[2] * 0.8, "cpu_r": line[3], "mem_r": line[4] * 0.9,
                      "disk_r": line[5] * 0.8, "label": False})
    tasks = sorted(tasks, key=lambda x: x["cpu"], reverse=True)
    new_task = []
    tag = total_number_host - 1
    for index in np.arange(total_number_host):
        if tasks[index]['label']:
            continue
        temp_new = tasks[index]         # 调度后的新host
        temp_real_new = tasks[index]    # 按照实际调度的host
        max_host = 3
        i = 0
        while (not over_limit(temp_new)) and i < max_host:
            if over_limit(add_host(tasks[index], tasks[tag])):
                break
            else:
                temp_new = add_host(tasks[index], tasks[tag])
                temp_real_new = add_host_r(tasks[index], tasks[tag])
                tasks[index]["label"] = True
                tasks[tag]["label"] = True
                tag -= 1
                i += 1
        new_task.append(temp_new)
        if over_limit(temp_real_new) and not over_limit(tasks[index]):
            num_error_schedule += 1
    print(len(new_task))
    print(total_number_host)
    print(len(new_task)/total_number_host)
    print(num_error_schedule)




