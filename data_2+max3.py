# -*- coding: utf-8 -*-
import os
from os.path import join
import numpy as np
# 生成数据样本，2天每小时+三天内的最高值


class sample(object):
    """
    每一个host就是一个样本
    """
    def __init__(self, file_route, host_name, line):
        # read file
        self.file_sample = open(file_route, 'rb')
        self.out_file_length = 2 * 24 * 6
        self.out_sample = []
        self.out_sample.append(host_name)
        self.line = line

    def construct_sample_target(self):
        # 直接生成带有目标值的文件
        self.file_sample.seek(0)
        line = self.file_sample.readlines()[self.line - 1]
        line = line.decode('utf-8').strip("\00")
        line = line.split(",")
        index = 0
        while index < self.out_file_length:
            self.out_sample.append(float(line[2 + index]))
            index += 1
            
        # self.out_sample.append(100 - np.min(np.asarray(line[(2 + self.out_file_length): (2 + self.out_file_length + 144 * 3)], float)))
        self.out_sample.append(np.max(np.asarray(line[(2 + self.out_file_length): (2 + self.out_file_length + 144 * 3)], float)))

    def get_array(self):
        return self.out_sample

    def __del__(self):
        self.file_sample.close()


if __name__ == '__main__':
    # 输出数据的文件，包含两天的每隔一小时的数据和三天内的最大值。
    outfile_unlabel = open("data\\sample\\max_3days\\MEM_sample_2days_max.csv", 'w', newline='')
    # 原来每个host文件的负载数据的文件夹
    source = 'data\\TS_data_normal\\600'
    print(source)
    line = 11        # 要提取数据的第几行，对应某一个类别
    # 读取每一个host文件，选择要的那一行整理并写出
    print(line)
    for root, dirs, files in os.walk(source):
        for one_file in files:
            print(one_file)
            onefullfilename = join(root, one_file)
            sam = sample(onefullfilename, one_file.split('.')[0], line)
            sam.construct_sample_target()
            samples = sam.get_array()
            for each in np.arange(len(samples) - 1):
                outfile_unlabel.write(str(samples[each]) + ',')
            outfile_unlabel.write(str(samples[len(samples) - 1]) + '\n')
            del sam
    outfile_unlabel.close()
