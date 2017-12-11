# -*- coding: utf-8 -*-
import os
from os.path import join
import numpy as np
# seize feature from a file contain 7 days' covariate values and 6 response values(marked 0 or 1, 1h, 6h, 1d, 24h, 2d, 3d)


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

    def construct_sample_label(self):
        # 生成带有label的文件
        self.file_sample.seek(0)
        line = self.file_sample.readlines()[self.line - 1]
        line = line.decode('utf-8').strip("\00")
        line = line.split(",")
        index = 0
        while index < self.out_file_length:
            self.out_sample.append(line[2 + index])
            index += 1
        self.out_sample.append(self.label_sample(line[(2 + self.out_file_length):(2 + self.out_file_length + 6)]))
        self.out_sample.append(self.label_sample(line[(2 + self.out_file_length):(2 + self.out_file_length + 36)]))
        self.out_sample.append(self.label_sample(line[(2 + self.out_file_length):(2 + self.out_file_length + 72)]))
        self.out_sample.append(self.label_sample(line[(2 + self.out_file_length):(2 + self.out_file_length + 144)]))
        self.out_sample.append(self.label_sample(line[(2 + self.out_file_length):(2 + self.out_file_length + 144 * 2)]))
        self.out_sample.append(self.label_sample(line[(2 + self.out_file_length):(2 + self.out_file_length + 144 * 3)]))

    def construct_sample_target(self):
        # 直接生成带有目标值的文件
        self.file_sample.seek(0)
        line = self.file_sample.readlines()[2]
        line = line.decode('utf-8').strip("\00")
        line = line.split(",")
        index = 0
        while index < self.out_file_length:
            self.out_sample.append(line[2 + index])
            index += 1

        self.out_sample.append(line[2 + self.out_file_length + 6])
        self.out_sample.append(line[2 + self.out_file_length + 36])
        self.out_sample.append(line[2 + self.out_file_length + 72])
        self.out_sample.append(line[2 + self.out_file_length + 144])
        self.out_sample.append(line[2 + self.out_file_length + 144 * 2])
        self.out_sample.append(line[2 + self.out_file_length + 144 * 3])

    def label_sample(self, sam_array):
        max_net = max([float(each) for each in self.out_sample[1:len(self.out_sample)]])
        # 如果在一段时间内有超过阈值的，则认为其过载
        for each in sam_array:
            # 设定标签阈值
            # if float(each) >= 80:
            if float(each) >= 1200000:
                return "1"
        return "0"

    def get_array(self):
        return self.out_sample

    def __del__(self):
        self.file_sample.close()


if __name__ == '__main__':
    # 用于分类的文件，target是标签
    outfile_label = open("data\\sample\\labeled\\NET_OUT_sample_2days_labeled.csv", 'w', newline='')
    # 用于回归的文件，不是标签是数值
    outfile_unlabel = open("data\\sample\\unlabeled\\NET_IOUT_sample_2days_unlabeled.csv", 'w', newline='')
    # 原来每个host文件的负载数据的文件夹
    source = 'data\\TS_data_normal\\600'
    line = 17        # 要提取数据的第几行，对应某一个类别
    # 读取每一个host文件，选择要的那一行整理并写出
    for root, dirs, files in os.walk(source):
        for one_file in files:
            print(one_file)
            onefullfilename = join(root, one_file)
            sam = sample(onefullfilename, one_file.split('.')[0], line)
            sam.construct_sample_target()
            for each in sam.get_array():
                outfile_unlabel.write(each + ',')
            outfile_unlabel.write('\n')
            del sam
            sam = sample(onefullfilename, one_file.split('.')[0], line)
            sam.construct_sample_label()
            for each in sam.get_array():
                outfile_label.write(each + ',')
            outfile_label.write('\n')
            del sam
    outfile_unlabel.close()
    outfile_label.close()
