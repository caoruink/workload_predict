# -*- coding: utf-8 -*-
# 使用决策树和GBRT预测三天内的最大值
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
import time


class SampleFile(object):
    """
    针对一个数据文件进行预测，首先按照target的个数区分X和Y，然后预测并计算结果
    """
    def __init__(self, file_route, file_cpu, file_mem, file_disk, num_target):
        """
        初始化一些参数
        :param file_route:文件的路径
        :param num_target: target的数量
        """
        global train_index, test_index
        self.file_route = file_route            # 文件路径
        self.cpu = file_cpu                     # 文件名
        self.mem = file_mem
        self.disk = file_disk
        sample_cpu = open(self.file_route + "\\" + self.cpu, 'r')
        sample_mem = open(self.file_route + "\\" + self.mem, 'r')
        sample_disk = open(self.file_route + "\\" + self.disk, 'r')
        self.predict_cpu = None                 # 预测结果
        self.predict_mem = None
        self.predict_disk = None
        self.real_cpu = None                    # 真实结果
        self.real_mem = None
        self.real_disk = None
        self.num_feature = 0                    # feature的数目，预设0，之后区分特征和目标时计算
        self.num_target = num_target            # 目标值个数
        self.hostname = []

        self.sample_x = None                    # 特征集合
        self.sample_y = None                    # 目标集合
        self.separate_XY(sample_cpu)
        self.predict_cpu, self.real_cpu = self.gbrt_pre(self.cpu_clf())

        self.sample_x = None
        self.sample_y = None
        self.separate_XY(sample_mem)
        self.predict_mem, self.real_mem = self.gbrt_pre(self.mem_clf())

        self.sample_x = None
        self.sample_y = None
        self.separate_XY(sample_disk)
        self.predict_disk, self.real_disk = self.gbrt_pre(self.disk_clf())

        sample_disk.close()
        sample_cpu.close()
        sample_mem.close()

        hostname = open("实验结果//max_3//hostname.txt", 'w')
        for each in np.array(self.hostname)[test_index]:
            hostname.write(each + '\n')
        hostname.close()
        np.savetxt("实验结果//max_3//predict.csv",
                   (np.column_stack((self.predict_cpu, self.predict_mem, self.predict_disk, self.real_cpu,
                                     self.real_mem, self.real_disk))), delimiter=',', fmt="%.2f")

    def separate_XY(self, file):
        """
        区分feature和target
        :return:
        """
        sample_content = file.readlines()
        print("separate X and Y at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        all_features = []
        for each in sample_content:
            line = each.split(",")
            self.hostname.append(line[0])
            line = line[1:len(line)]
            new_line = []
            for every in line:
                new_line.append(float(every))
            all_features.append(new_line)
        sample_content = np.array(all_features)
        del all_features
        self.num_feature = len(sample_content[0, ]) - self.num_target
        self.sample_x = sample_content[:, 0: self.num_feature]
        self.sample_y = sample_content[:, self.num_feature: self.num_feature + self.num_target]
        print("end separate X and Y at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        return

    def decision_tree_pre(self):
        self.decision_tree_outfile.write(
            "fit model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        # rng = np.random.RandomState(1)
        x_train, x_test, y_train, y_test = train_test_split(self.sample_x, self.sample_y, test_size=0.4,
                                                            random_state=0)
        # CPU参数
        clf = DecisionTreeRegressor(max_depth=16, min_samples_leaf=1, min_samples_split=6, criterion="mse",
                                    splitter="best", random_state=0)
        # MEM参数
        # clf = DecisionTreeRegressor(max_depth=16, min_samples_leaf=1, min_samples_split=6, criterion="mse",
        #                             splitter="best", random_state=0)
        # DISK参数
        # clf = DecisionTreeRegressor(max_depth=16, min_samples_leaf=1, min_samples_split=6, criterion="mse",
                                    # splitter="best", random_state=0)
        # param_grid = {"max_depth": np.arange(9, 10),
        #               "min_samples_split": np.arange(9, 10),
        #               "min_samples_leaf": np.arange(9, 10),
        #               "criterion": ["mse", "friedman_mse", "mae"],
        #               "splitter": ["best", "random"],
        #               "random_state": [1]}
        # clf = GridSearchCV(clf, param_grid=param_grid, n_jobs=4)
        start = time.time()
        clf.fit(x_train, y_train)
        print("train %.2f seconds for decision tree." % (time.time() - start))
        # print(clf.best_params_)
        y_predict = clf.predict(x_test)
        mse = mean_squared_error(y_test, y_predict)
        print("mse" + str(mse))
        print("score:" + str(clf.score(x_test, y_test)))
        # self.decision_tree_outfile.write("GridSearchCV too %.2f seconds for %d candidate parameter settings."
        #                                  % (time.time() - start, len(clf.cv_results_['params'])) + "\n")
        # self.decision_tree_outfile.write(clf.best_params_ + "\n")
        self.decision_tree_outfile.write("mse" + str(mse) + "\n")
        self.decision_tree_outfile.write("score:" + str(clf.score(x_test, y_test)) + "\n")
        self.decision_tree_outfile.write(
            "end model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        print("end decision tree!")
        return np.column_stack((y_predict, y_test))

    @staticmethod
    def cpu_clf():
        return GradientBoostingRegressor(loss='lad', learning_rate=0.1, n_estimators=500, max_depth=16,
                                         min_samples_split=5, min_samples_leaf=1, criterion='mse', random_state=1)

    @staticmethod
    def mem_clf():
        return GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=500, max_depth=16,
                                         min_samples_split=5, min_samples_leaf=1, criterion='mse', random_state=1)

    @staticmethod
    def disk_clf():
        return GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=500, max_depth=16,
                                         min_samples_split=5, min_samples_leaf=1, criterion='mse', random_state=1)

    def gbrt_pre(self, clf):
        print("fit model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        x_train = self.sample_x[train_index, :]
        x_test = self.sample_x[test_index, :]
        y_train = self.sample_y[train_index, :]
        y_test = self.sample_y[test_index, :]
        y_train = np.reshape(y_train, (len(y_train)))
        y_test = np.reshape(y_test, (len(y_test)))
        start = time.time()
        clf.fit(x_train, y_train)
        print("train %.2f seconds for GBRT." % (time.time() - start))
        y_predict = clf.predict(x_test)
        mse = mean_squared_error(y_test, y_predict)
        print("mse" + str(mse))
        print("score:" + str(clf.score(x_test, y_test)))
        print("end model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        return y_predict, y_test


if __name__ == '__main__':
    X = np.arange(5456)                 # 总共的host数目
    ss = ShuffleSplit(n_splits=1, test_size=0.4, random_state=1)
    train_index = None
    test_index = None
    for train, test in ss.split(X):
        train_index = train
        test_index = test

    file_cpu = "CPU_sample_2days_max.csv"
    file_mem = "MEM_sample_2days_max.csv"
    file_disk = "DISK_sample_2days_max.csv"

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    Sample = SampleFile("data\\sample\\max_3days", file_cpu, file_mem, file_disk, 1)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    del Sample
