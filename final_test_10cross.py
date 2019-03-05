# -*- coding: utf-8 -*-
# 使用其他方法做同一个实验1.决策树2.
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import time


class SampleFile(object):
    """
    针对一个数据文件进行预测，首先按照target的个数区分X和Y，然后预测并计算结果
    """
    def __init__(self, file_route, filename, num_target):
        """
        初始化一些参数
        :param file_route:文件的路径
        :param filename: 文件的名字
        :param num_target: target的数量
        """
        self.__file_route = file_route          # 文件路径
        self.__filename = filename              # 文件名
        sample_file = open(self.__file_route + "\\" + self.__filename, 'r')
        self.__sample_x = None                  # 特征集合
        self.__sample_y = None                  # 目标集合
        self.__hostname = []                    # hostname
        self.__num_feature = 0                  # feature的数目，预设0，之后区分特征和目标时计算
        self.__num_target = num_target          # 目标值个数
        self.__sample_content = sample_file.readlines()
        sample_file.close()
        del sample_file
        self.__num_split = 10

        # 评价指标，给每种方法都建立一个数组来记录
        self.decision_tree_error = []
        self.logistic_error = []
        self.svm_error = []
        self.nb_error = []
        self.gbdt_error = []
        self.vote_error = []

        # 预测的每一个host的结果记录
        self.decision_tree_result = []
        self.logistic_result = []
        self.svm_result = []
        self.nb_result = []
        self.gbdt_result = []
        self.vote_result = []

        # 十个汇总
        self.f_dt_error = []
        self.f_lg_error = []
        self.f_svm_error = []
        self.f_nb_error = []
        self.f_gbdt_error = []
        self.f_vote_error = []

    def separate_XY(self):
        """
        区分feature和target
        :return:
        """
        print("separate X and Y at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        all_features = []
        for each in self.__sample_content:
            line = each.split(",")
            self.__hostname.append(line[0])
            line = line[1:len(line)]
            new_line = []
            for every in line:
                new_line.append(float(every))
            all_features.append(new_line)
        self.__sample_content = np.array(all_features)
        del all_features
        self.__num_feature = len(self.__sample_content[0, ]) - self.__num_target
        self.__hostname = np.array(self.__hostname)
        self.__sample_x = self.__sample_content[:, 0: self.__num_feature]
        self.__sample_y = self.__sample_content[:, self.__num_feature: self.__num_feature + self.__num_target]
        print("end separate X and Y at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        return

    @staticmethod
    def cal_error_class(y_test, y_predict):
        """
        计算预警准确的和误报率
        :param y_test: 原始值
        :param y_predict: 与测试
        :return: 预警准确率和误报率
        """
        TP = 0  # BOTH 0,
        FP = 0  # PRE 0, ACT 1
        FN = 0  # PRE 1, ACT 0
        TN = 0  # BOTH 1
        for index in np.arange(0, len(y_test)):
            if y_test[index] == 1:
                if y_predict[index] == 1:
                    TN += 1
                else:
                    FP += 1
            else:
                if y_predict[index] == 1:
                    FN += 1
                else:
                    TP += 1
        # print(TN, TN + FP)
        alarm_rate = TN / (TN + FP)
        error_rate = (FN + FP) / (TP + FN + TN + FP)
        print(alarm_rate, error_rate)
        print(TP/(TP + FP), TP/(TP + FN))
        return alarm_rate, error_rate

    def decision_tree_pre(self, x_train, y_train, x_test, y_test):
        """
        做分类预测
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        """
        # 日志文件，给每种方法都建立一个日志文件
        decision_tree_outfile = open(self.__file_route + "/result/decision tree/res_dt_"
                                     + self.__filename.split(".")[0] + ".txt", 'w')
        decision_tree_outfile.write(
            "fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        clf = DecisionTreeClassifier(random_state=1, min_samples_split=2, min_samples_leaf=5, max_depth=15,
                                     class_weight="balanced")
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        self.decision_tree_result = y_predict
        # prob = clf.predict_proba(x_test)
        # prob = np.array(prob)
        # temp = 1
        # for each in prob:
        #     np.savetxt(self.__file_route + "//prob//decision tree//" + self.__filename.split(".")[0] + str(temp) +
        #                "_dt_prob.csv", each, delimiter=',', fmt='%.2f')
        #     temp += 1
        print(clf.score(x_test, y_test))
        for i in np.arange(0, self.__num_target):
            print(classification_report(y_test[:, i], y_predict[:, i], target_names=['未过载', '过载']))
            alarm_rate, error_rate = self.cal_error_class(y_test[:, i], y_predict[:, i])
            self.decision_tree_error.append([alarm_rate, error_rate])
            decision_tree_outfile.write("alarm_rate:" + str(alarm_rate) + "  error_rate:" + str(error_rate) + "\n")
            decision_tree_outfile.write(
                "end fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        decision_tree_outfile.close()
        print("alarm_rate      error_rate:")
        print(self.decision_tree_error)
        return

    def logistic_pre(self, x_train, y_train, x_test, y_test):
        """
        做分类预测
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        """
        # 日志文件，给每种方法都建立一个日志文件
        logistic_outfile = open(self.__file_route + "/result/logistic/res_logistic_"
                                + self.__filename.split(".")[0] + ".txt", 'w')
        logistic_outfile.write(
            "fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        # rng = np.random.RandomState(1)
        clf = LogisticRegression(penalty='l2', dual=False, class_weight="balanced", solver='newton-cg', max_iter=100,
                                 random_state=1)
        for i in np.arange(0, self.__num_target):
            clf.fit(x_train, y_train[:, i])
            y_predict = clf.predict(x_test)
            self.logistic_result.append(y_predict)
            # prob = clf.predict_proba(x_test)
            # np.savetxt(self.__file_route + "//prob//logistic//" + self.__filename.split(".")[0] + str(i) +
            #            "_logistic_pre_prob.csv", prob, delimiter=',', fmt='%.2f')
            print(classification_report(y_test[:, i], y_predict, target_names=['未过载', '过载']))
            print("score:" + str(clf.score(x_test, y_test[:, i])))
            alarm_rate, error_rate = self.cal_error_class(y_test[:, i], y_predict)
            self.logistic_error.append([alarm_rate, error_rate])
            logistic_outfile.write("alarm_rate:" + str(alarm_rate) + "    error_rate:" + str(error_rate) + "\n")
            logistic_outfile.write(
                "end fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        logistic_outfile.close()
        print("alarm_rate      error_rate:")
        print(self.logistic_error)
        return

    def svm_pre(self, x_train, y_train, x_test, y_test):
        """
                做分类预测
                :param x_train:
                :param y_train:
                :param x_test:
                :param y_test:
                :return:
                """
        svm_outfile = open(self.__file_route + "/result/svm/res_svm_"
                           + self.__filename.split(".")[0] + ".txt", 'w')
        svm_outfile.write(
            "fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        # rng = np.random.RandomState(1)
        clf = SVC(kernel='linear', degree=3, probability=True, class_weight="balanced", random_state=1)

        for i in np.arange(0, self.__num_target):
            clf.fit(x_train, y_train[:, i])
            y_predict = clf.predict(x_test)
            self.svm_result.append(y_predict)
            # prob = clf.predict_proba(x_test)
            # np.savetxt(self.__file_route + "//prob//svm//" + self.__filename.split(".")[0] + str(i) +
            #            "_svm_pre_prob.csv", prob, delimiter=',', fmt='%.2f')
            print(classification_report(y_test[:, i], y_predict, target_names=['未过载', '过载']))
            print("score:" + str(clf.score(x_test, y_test[:, i])))
            alarm_rate, error_rate = self.cal_error_class(y_test[:, i], y_predict)
            self.svm_error.append([alarm_rate, error_rate])
            svm_outfile.write("alarm_rate:" + str(alarm_rate) + "    error_rate:" + str(error_rate) + "\n")
            svm_outfile.write(
                "end fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        svm_outfile.close()
        print("alarm_rate      error_rate:")
        print(self.svm_error)
        return

    def nb_pre(self, x_train, y_train, x_test, y_test):
        """
        做分类预测
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        """
        nb_outfile = open(self.__file_route + "/result/naive bayes/res_nb_"
                          + self.__filename.split(".")[0] + ".txt", 'w')
        nb_outfile.write(
            "fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        # rng = np.random.RandomState(1)
        clf = GaussianNB()
        for i in np.arange(0, self.__num_target):
            clf.fit(x_train, y_train[:, i])
            y_predict = clf.predict(x_test)
            self.nb_result.append(y_predict)
            # prob = clf.predict_proba(x_test)
            # np.savetxt(self.__file_route + "//prob//naive bayes//" + self.__filename.split(".")[0] + str(i) +
            #            "_nb_pre_prob.csv", prob, delimiter=',', fmt='%.2f')
            print(classification_report(y_test[:, i], y_predict, target_names=['未过载', '过载']))
            print("score:" + str(clf.score(x_test, y_test[:, i])))
            alarm_rate, error_rate = self.cal_error_class(y_test[:, i], y_predict)
            self.nb_error.append([alarm_rate, error_rate])
            nb_outfile.write("alarm_rate:" + str(alarm_rate) + "    error_rate:" + str(error_rate) + "\n")
            nb_outfile.write(
                "end fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        nb_outfile.close()
        print("alarm_rate      error_rate:")
        print(self.nb_error)
        return

    def gbdt_pre(self, x_train, y_train, x_test, y_test):
        """
        做分类预测
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        """
        gbdt_outfile = open(self.__file_route + "/result/GBDT/res_gbdt_"
                            + self.__filename.split(".")[0] + ".txt", 'w')
        gbdt_outfile.write(
            "fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        # rng = np.random.RandomState(1)
        # clf = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, max_depth=15, random_state=1)
        clf = RandomForestClassifier(n_estimators=1000, max_depth=15, random_state=1, min_samples_leaf=5,
                                     min_samples_split=2, class_weight="balanced")
        for i in np.arange(0, self.__num_target):
            clf.fit(x_train, y_train[:, i])
            y_predict = clf.predict(x_test)
            self.gbdt_result.append(y_predict)
            # prob = clf.predict_proba(x_test)
            # np.savetxt(self.__file_route + "//prob//GBDT//" + self.__filename.split(".")[0] + str(i) +
            #            "_gbdt_pre_prob.csv", prob, delimiter=',', fmt='%.2f')
            print(classification_report(y_test[:, i], y_predict, target_names=['未过载', '过载']))
            print("score:" + str(clf.score(x_test, y_test[:, i])))
            alarm_rate, error_rate = self.cal_error_class(y_test[:, i], y_predict)
            self.gbdt_error.append([alarm_rate, error_rate])
            gbdt_outfile.write("alarm_rate:" + str(alarm_rate) + "    error_rate:" + str(error_rate) + "\n")
            gbdt_outfile.write(
                "end fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        gbdt_outfile.close()
        print("alarm_rate      error_rate:")
        print(self.gbdt_error)
        return

    def predict_class(self):
        """
        串起预测的整个流程
        :return:
        """
        self.separate_XY()
        kf = KFold(n_splits=self.__num_split, shuffle=True)
        for train_index, test_index in kf.split(self.__sample_x):
            # 评价指标，给每种方法都建立一个数组来记录
            self.decision_tree_error = []
            self.logistic_error = []
            self.svm_error = []
            self.nb_error = []
            self.gbdt_error = []
            self.vote_error = []

            # 预测的每一个host的结果记录
            self.decision_tree_result = []
            self.logistic_result = []
            self.svm_result = []
            self.nb_result = []
            self.gbdt_result = []
            self.vote_result = []
            x_train, x_test = self.__sample_x[train_index], self.__sample_x[test_index]
            y_train, y_test = self.__sample_y[train_index], self.__sample_y[test_index]
            # 用决策树做
            print("----------------------------decision tree-----------------------------")
            self.decision_tree_pre(x_train, y_train, x_test, y_test)
            
            #
            # 用逻辑回归做
            print("----------------------------Logistic-----------------------------")
            self.logistic_pre(x_train, y_train, x_test, y_test)

            print("----------------------------SVM-----------------------------")
            self.svm_pre(x_train, y_train, x_test, y_test)

            print("----------------------------Naive Bayes-----------------------------")
            self.nb_pre(x_train, y_train, x_test, y_test)

            print("----------------------------GBDT-----------------------------")
            self.gbdt_pre(x_train, y_train, x_test, y_test)

            print("----------------------------vote-------------------------------")
            self.vote(y_test)

            self.f_dt_error.append(self.decision_tree_error)
            self.f_lg_error.append(self.logistic_error)
            self.f_svm_error.append(self.svm_error)
            self.f_nb_error.append(self.nb_error)
            self.f_gbdt_error.append(self.gbdt_error)
            self.f_vote_error.append(self.vote_error)
        np.savetxt("实验结果//DT_error_" + self.__filename, np.mean(np.array(self.f_dt_error), axis=0), delimiter=',',
                   fmt='%.6f')
        np.savetxt("实验结果//LOGISTIC_error_" + self.__filename, np.mean(np.array(self.f_lg_error), axis=0),
                   delimiter=',', fmt='%.6f')
        np.savetxt("实验结果//SVM_error_" + self.__filename, np.mean(np.array(self.f_svm_error), axis=0), delimiter=',',
                   fmt='%.6f')
        np.savetxt("实验结果//NB_error_" + self.__filename, np.mean(np.array(self.f_nb_error), axis=0), delimiter=',',
                   fmt='%.6f')
        np.savetxt("实验结果//GBDT_error_" + self.__filename, np.mean(np.array(self.f_gbdt_error), axis=0),
                   delimiter=',', fmt='%.6f')
        np.savetxt("实验结果//vote_error_" + self.__filename, np.mean(np.array(self.f_vote_error), axis=0),
                   delimiter=',', fmt='%.6f')
        return

    def vote(self, y_test):
        self.decision_tree_result = np.array(self.decision_tree_result)
        self.logistic_result = np.array(self.logistic_result).T
        self.svm_result = np.array(self.svm_result).T
        self.nb_result = np.array(self.nb_result).T
        self.gbdt_result = np.array(self.gbdt_result).T

        for j in np.arange(self.__num_target):
            re = []
            for i in np.arange(len(self.gbdt_result)):
                item = 0
                if self.decision_tree_result[i, j] == 1:
                    item += 1
                if self.logistic_result[i, j] == 1:
                    item += 1
                if self.svm_result[i, j] == 1:
                    item += 1
                if self.nb_result[i, j] == 1:
                    item += 1
                if self.gbdt_result[i, j] == 1:
                    item += 1
                if item > 3:
                    re.append(1)
                else:
                    re.append(0)
            self.vote_result.append(re)
            alarm_rate, error_rate = self.cal_error_class(y_test[:, j], re)
            self.vote_error.append([alarm_rate, error_rate])
        self.vote_result = np.array(self.vote_result).T
        print("alarm_rate      error_rate:")
        print(self.vote_error)


if __name__ == '__main__':
    # MEM
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # vote_2days = SampleFile("data/sample/labeled/MEM", "GBDT_voteMEM_sample_2days_labeled.csv", 6)
    # vote_2days.predict_class()
    # del vote_2days

    # CPU
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    vote_2days = SampleFile("data/sample/labeled/CPU", "GBDT_voteCPU_sample_2days_labeled.csv", 6)
    vote_2days.predict_class()
    del vote_2days

    # DISK
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # vote_2days = SampleFile("data/sample/labeled/DISK", "GBDT_voteDISK_sample_2days_labeled.csv", 6)
    # vote_2days.predict_class()
    # del vote_2days
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
