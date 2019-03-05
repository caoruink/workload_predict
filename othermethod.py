# -*- coding: utf-8 -*-
# 使用其他方法做同一个实验1.决策树2.
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
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
        # 日志文件，给每种方法都建立一个日志文件
        self.decision_tree_outfile = open(self.__file_route + "//result//decision tree//res_dt_"
                                          + self.__filename.split(".")[0] + ".txt", 'w')
        self.logistic_outfile = open(self.__file_route + "//result//logistic//res_logistic_"
                                     + self.__filename.split(".")[0] + ".txt", 'w')
        self.svm_outfile = open(self.__file_route + "//result//svm//res_svm_"
                                + self.__filename.split(".")[0] + ".txt", 'w')

        self.nb_outfile = open(self.__file_route + "//result//naive bayes//res_nb_"
                               + self.__filename.split(".")[0] + ".txt", 'w')
        self.gbdt_outfile = open(self.__file_route + "//result//GBDT//res_gbdt_"
                                 + self.__filename.split(".")[0] + ".txt", 'w')
        # self.outfile = open(self.__file_route + "//result//result" + self.__filename.split(".")[0] + ".txt", 'w')
        # self.outfile.write("Begin " + self.__filename + " at " + time.strftime('%Y-%m-%d %H:%M:%S',
        #                                                                        time.localtime(time.time())) + "\n")
        # 评价指标，给每种方法都建立一个数组来记录
        self.decision_tree_error = []
        self.logistic_error = []
        self.svm_error = []
        self.nb_error = []
        self.gbdt_error = []
        self.__num_split = 10                   # 交叉验证的次数

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
        self.decision_tree_outfile.write(
            "fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        # rng = np.random.RandomState(1)
        clf = DecisionTreeClassifier(random_state=1, min_samples_split=2, min_samples_leaf=5, max_depth=7,
                                     class_weight="balanced")
        # clf = AdaBoostClassifier(dt, n_estimators=700, random_state=rng, learning_rate=1.3)
        # print(x_train.shape)
        # print(y_train.shape)
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        prob = clf.predict_proba(x_test)
        prob = np.array(prob)
        if self.__num_target == 1:
            np.savetxt(self.__file_route + "//prob//decision tree//" + self.__filename.split(".")[0] +
                       "_decision_tree_prob.csv", prob, delimiter=',', fmt='%.2f')
            print(classification_report(y_test, y_predict, target_names=['未过载', '过载']))
            print("score:" + str(clf.score(x_test, y_test)))
            alarm_rate, error_rate = self.cal_error_class(y_test, y_predict)
            self.decision_tree_error.append(alarm_rate)
            self.decision_tree_error.append(error_rate)
            self.decision_tree_outfile.write("alarm_rate:" + str(alarm_rate) + "    error_rate:" + str(error_rate) + "\n")
            self.decision_tree_outfile.write(
                "end fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
            return
        temp = 1
        for each in prob:
            np.savetxt(self.__file_route + "//prob//decision tree//" + self.__filename.split(".")[0] + str(temp) +
                       "_dt_prob.csv", each, delimiter=',', fmt='%.2f')
            temp += 1
        print(clf.score(x_test, y_test))
        for i in np.arange(0, self.__num_target):
            print(classification_report(y_test[:, i], y_predict[:, i], target_names=['未过载', '过载']))
            alarm_rate, error_rate = self.cal_error_class(y_test[:, i], y_predict[:, i])
            self.decision_tree_error.append(alarm_rate)
            self.decision_tree_error.append(error_rate)
            self.decision_tree_outfile.write("alarm_rate:" + str(alarm_rate) + "    error_rate:" + str(error_rate) + "\n")
            self.decision_tree_outfile.write(
                "end fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
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
        self.logistic_outfile.write(
            "fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        # rng = np.random.RandomState(1)
        clf = LogisticRegression(penalty='l2', dual=False, class_weight="balanced", solver='newton-cg', max_iter=100,
                                 random_state=1)
        if self.__num_target == 1:
            clf.fit(x_train, y_train)
            y_predict = clf.predict(x_test)
            prob = clf.predict_proba(x_test)
            prob = np.array(prob)
            np.savetxt(self.__file_route + "//prob//logistic//" + self.__filename.split(".")[0] +
                       "_logistic_prob.csv", prob, delimiter=',', fmt='%.2f')
            print(classification_report(y_test, y_predict, target_names=['未过载', '过载']))
            print("score:" + str(clf.score(x_test, y_test)))
            alarm_rate, error_rate = self.cal_error_class(y_test, y_predict)
            self.logistic_error.append(alarm_rate)
            self.logistic_error.append(error_rate)
            self.logistic_outfile.write("alarm_rate:" + str(alarm_rate) + "    error_rate:" + str(error_rate) + "\n")
            self.logistic_outfile.write(
                "end fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
            return

        for i in np.arange(0, self.__num_target):
            clf.fit(x_train, y_train[:, i])
            y_predict = clf.predict(x_test)
            prob = clf.predict_proba(x_test)
            np.savetxt(self.__file_route + "//prob//logistic//" + self.__filename.split(".")[0] + str(i) +
                       "_logistic_pre_prob.csv", prob, delimiter=',', fmt='%.2f')
            print(classification_report(y_test[:, i], y_predict, target_names=['未过载', '过载']))
            print("score:" + str(clf.score(x_test, y_test[:, i])))
            alarm_rate, error_rate = self.cal_error_class(y_test[:, i], y_predict)
            self.logistic_error.append(alarm_rate)
            self.logistic_error.append(error_rate)
            self.logistic_outfile.write("alarm_rate:" + str(alarm_rate) + "    error_rate:" + str(error_rate) + "\n")
            self.logistic_outfile.write(
                "end fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
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
        self.svm_outfile.write(
            "fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        # rng = np.random.RandomState(1)
        clf = SVC(kernel='linear', degree=3, probability=True, class_weight="balanced", random_state=1)
        if self.__num_target == 1:
            clf.fit(x_train, y_train)
            y_predict = clf.predict(x_test)
            prob = clf.predict_proba(x_test)
            prob = np.array(prob)
            np.savetxt(self.__file_route + "//prob//svm//" + self.__filename.split(".")[0] +
                       "_svm_prob.csv", prob, delimiter=',', fmt='%.2f')
            print(classification_report(y_test, y_predict, target_names=['未过载', '过载']))
            print("score:" + str(clf.score(x_test, y_test)))
            alarm_rate, error_rate = self.cal_error_class(y_test, y_predict)
            self.svm_error.append(alarm_rate)
            self.svm_error.append(error_rate)
            self.svm_outfile.write("alarm_rate:" + str(alarm_rate) + "    error_rate:" + str(error_rate) + "\n")
            self.svm_outfile.write(
                "end fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
            return

        for i in np.arange(0, self.__num_target):
            clf.fit(x_train, y_train[:, i])
            y_predict = clf.predict(x_test)
            prob = clf.predict_proba(x_test)
            np.savetxt(self.__file_route + "//prob//svm//" + self.__filename.split(".")[0] + str(i) +
                       "_svm_pre_prob.csv", prob, delimiter=',', fmt='%.2f')
            print(classification_report(y_test[:, i], y_predict, target_names=['未过载', '过载']))
            print("score:" + str(clf.score(x_test, y_test[:, i])))
            alarm_rate, error_rate = self.cal_error_class(y_test[:, i], y_predict)
            self.svm_error.append(alarm_rate)
            self.svm_error.append(error_rate)
            self.svm_outfile.write("alarm_rate:" + str(alarm_rate) + "    error_rate:" + str(error_rate) + "\n")
            self.svm_outfile.write(
                "end fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
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
        self.nb_outfile.write(
            "fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        # rng = np.random.RandomState(1)
        clf = GaussianNB()
        if self.__num_target == 1:
            clf.fit(x_train, y_train)
            y_predict = clf.predict(x_test)
            prob = clf.predict_proba(x_test)
            prob = np.array(prob)
            np.savetxt(self.__file_route + "//prob//naive bayes//" + self.__filename.split(".")[0] +
                       "_nb_prob.csv", prob, delimiter=',', fmt='%.2f')
            print(classification_report(y_test, y_predict, target_names=['未过载', '过载']))
            print("score:" + str(clf.score(x_test, y_test)))
            alarm_rate, error_rate = self.cal_error_class(y_test, y_predict)
            self.nb_error.append(alarm_rate)
            self.nb_error.append(error_rate)
            self.nb_outfile.write("alarm_rate:" + str(alarm_rate) + "    error_rate:" + str(error_rate) + "\n")
            self.nb_outfile.write(
                "end fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
            return

        for i in np.arange(0, self.__num_target):
            clf.fit(x_train, y_train[:, i])
            y_predict = clf.predict(x_test)
            prob = clf.predict_proba(x_test)
            np.savetxt(self.__file_route + "//prob//naive bayes//" + self.__filename.split(".")[0] + str(i) +
                       "_nb_pre_prob.csv", prob, delimiter=',', fmt='%.2f')
            print(classification_report(y_test[:, i], y_predict, target_names=['未过载', '过载']))
            print("score:" + str(clf.score(x_test, y_test[:, i])))
            alarm_rate, error_rate = self.cal_error_class(y_test[:, i], y_predict)
            self.nb_error.append(alarm_rate)
            self.nb_error.append(error_rate)
            self.nb_outfile.write("alarm_rate:" + str(alarm_rate) + "    error_rate:" + str(error_rate) + "\n")
            self.nb_outfile.write(
                "end fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
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
        self.gbdt_outfile.write(
            "fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        # rng = np.random.RandomState(1)
        clf = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, max_depth=8, random_state=1)
        if self.__num_target == 1:
            clf.fit(x_train, y_train)
            y_predict = clf.predict(x_test)
            prob = clf.predict_proba(x_test)
            prob = np.array(prob)
            np.savetxt(self.__file_route + "//prob//GBDT//" + self.__filename.split(".")[0] +
                       "_gbdt_prob.csv", prob, delimiter=',', fmt='%.2f')
            print(classification_report(y_test, y_predict, target_names=['未过载', '过载']))
            print("score:" + str(clf.score(x_test, y_test)))
            alarm_rate, error_rate = self.cal_error_class(y_test, y_predict)
            self.gbdt_error.append(alarm_rate)
            self.gbdt_error.append(error_rate)
            self.gbdt_outfile.write("alarm_rate:" + str(alarm_rate) + "    error_rate:" + str(error_rate) + "\n")
            self.gbdt_outfile.write(
                "end fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
            return

        for i in np.arange(0, self.__num_target):
            clf.fit(x_train, y_train[:, i])
            y_predict = clf.predict(x_test)
            prob = clf.predict_proba(x_test)
            np.savetxt(self.__file_route + "//prob//GBDT//" + self.__filename.split(".")[0] + str(i) +
                       "_gbdt_pre_prob.csv", prob, delimiter=',', fmt='%.2f')
            print(classification_report(y_test[:, i], y_predict, target_names=['未过载', '过载']))
            print("score:" + str(clf.score(x_test, y_test[:, i])))
            alarm_rate, error_rate = self.cal_error_class(y_test[:, i], y_predict)
            self.gbdt_error.append(alarm_rate)
            self.gbdt_error.append(error_rate)
            self.gbdt_outfile.write("alarm_rate:" + str(alarm_rate) + "    error_rate:" + str(error_rate) + "\n")
            self.gbdt_outfile.write(
                "end fit class model at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        return

    def predict_class(self):
        """
        串起预测的整个流程
        :return:
        """
        self.separate_XY()
        kf = KFold(n_splits=self.__num_split, shuffle=True)

        # 用决策树做
        # print("----------------------------decision tree-----------------------------")
        # self.decision_tree_outfile.write("Begin " + self.__filename + " at " +
        #                                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        # for train_index, test_index in kf.split(self.__sample_x):
        #     x_train, x_test = self.__sample_x[train_index], self.__sample_x[test_index]
        #     y_train, y_test = self.__sample_y[train_index], self.__sample_y[test_index]
        #     self.decision_tree_outfile.write("*****************************decision tree****************************\n")
        #     self.decision_tree_pre(x_train, y_train, x_test, y_test)
        #     self.decision_tree_outfile.write("######################################################################\n")
        #     self.decision_tree_outfile.write("\n")
        # np.savetxt("实验结果//DT_error_" + self.__filename,
        #            self.solve_error_class(self.decision_tree_error, self.decision_tree_outfile),
        #            delimiter=',', fmt='%.6f')
        # self.decision_tree_outfile.close()
        #
        # # 用逻辑回归做
        # print("----------------------------Logistic-----------------------------")
        # self.logistic_outfile.write("Begin " + self.__filename + " at " +
        #                             time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        # for train_index, test_index in kf.split(self.__sample_x):
        #     x_train, x_test = self.__sample_x[train_index], self.__sample_x[test_index]
        #     y_train, y_test = self.__sample_y[train_index], self.__sample_y[test_index]
        #     self.logistic_outfile.write("*****************************Logistic****************************\n")
        #     self.logistic_pre(x_train, y_train, x_test, y_test)
        #     self.logistic_outfile.write("######################################################################\n")
        #     self.logistic_outfile.write("\n")
        # np.savetxt("实验结果//LOGISTIC_error_" + self.__filename,
        #            self.solve_error_class(self.logistic_error, self.logistic_outfile),
        #            delimiter=',', fmt='%.6f')
        # self.logistic_outfile.close()
        #
        # print("----------------------------SVM-----------------------------")
        # self.svm_outfile.write("Begin " + self.__filename + " at " +
        #                        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        # for train_index, test_index in kf.split(self.__sample_x):
        #     x_train, x_test = self.__sample_x[train_index], self.__sample_x[test_index]
        #     y_train, y_test = self.__sample_y[train_index], self.__sample_y[test_index]
        #     self.svm_outfile.write("*****************************SVM****************************\n")
        #     self.svm_pre(x_train, y_train, x_test, y_test)
        #     self.svm_outfile.write("######################################################################\n")
        #     self.svm_outfile.write("\n")
        # np.savetxt("实验结果//SVM_error_" + self.__filename, self.solve_error_class(self.svm_error, self.svm_outfile),
        #            delimiter=',', fmt='%.6f')
        # self.svm_outfile.close()
        #
        # print("----------------------------Naive Bayes-----------------------------")
        # self.nb_outfile.write("Begin " + self.__filename + " at " +
        #                       time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        # for train_index, test_index in kf.split(self.__sample_x):
        #     x_train, x_test = self.__sample_x[train_index], self.__sample_x[test_index]
        #     y_train, y_test = self.__sample_y[train_index], self.__sample_y[test_index]
        #     self.nb_outfile.write("*****************************Naive Bayes****************************\n")
        #     self.nb_pre(x_train, y_train, x_test, y_test)
        #     self.nb_outfile.write("######################################################################\n")
        #     self.nb_outfile.write("\n")
        # np.savetxt("实验结果//NB_error_" + self.__filename, self.solve_error_class(self.nb_error, self.nb_outfile),
        #                delimiter=',', fmt='%.6f')
        # self.nb_outfile.close()

        print("----------------------------GBDT-----------------------------")
        self.gbdt_outfile.write("Begin " + self.__filename + " at " +
                                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\n")
        for train_index, test_index in kf.split(self.__sample_x):
            x_train, x_test = self.__sample_x[train_index], self.__sample_x[test_index]
            y_train, y_test = self.__sample_y[train_index], self.__sample_y[test_index]
            self.gbdt_outfile.write("*****************************GBDT****************************\n")
            self.gbdt_pre(x_train, y_train, x_test, y_test)
            self.gbdt_outfile.write("######################################################################\n")
            self.gbdt_outfile.write("\n")
        np.savetxt("实验结果//GBDT_error_" + self.__filename, self.solve_error_class(self.gbdt_error, self.gbdt_outfile),
                   delimiter=',', fmt='%.6f')
        self.gbdt_outfile.close()
        return

    def solve_error_class(self, error, outfile):
        new_error = []
        for i in np.arange(0, self.__num_split):
            new_error.append(error[i * 2 * self.__num_target: (i + 1) * 2 * self.__num_target])
        # print(new_error)
        mean_error = np.mean(new_error, 0)
        outfile.write(
            "mean alarm rate and mean error rate of " + self.__filename + " is" + str(mean_error) + "\n")
        print("mean alarm rate and mean error rate of " + self.__filename + " is" + str(mean_error) + "\n")
        alarm_rate = []
        error_rate = []
        for i in np.arange(0, 6):
            alarm_rate.append(mean_error[i * 2])
            error_rate.append(mean_error[i * 2 + 1])
        print(np.column_stack((alarm_rate, error_rate)))
        return np.column_stack((alarm_rate, error_rate))


if __name__ == '__main__':
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # vote_2days = SampleFile("data\\sample\\labeled", "GBDT_votesample_2days_labeled.csv", 6)
    # vote_2days.predict_class()
    # del vote_2days
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    #
    # vote_2days = SampleFile("data\\sample\\labeled\\MEM", "GBDT_voteMEM_sample_2days_labeled.csv", 6)
    # vote_2days.predict_class()
    # del vote_2days
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    #
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    #
    # vote_2days = SampleFile("data\\sample\\labeled\\CPU", "GBDT_voteCPU_sample_2days_labeled.csv", 6)
    # vote_2days.predict_class()
    # del vote_2days
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    #
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    vote_2days = SampleFile("data\\sample\\labeled\\DISK", "GBDT_voteDISK_sample_2days_labeled.csv", 6)
    vote_2days.predict_class()
    del vote_2days
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # vote_2days = SampleFile("data\\sample\\labeled", "GBDT_0sample_2days_labeled.csv", 1)
    # vote_2days.predict_class()
    # del vote_2days
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # vote_2days = SampleFile("data\\sample\\labeled", "GBDT_1sample_2days_labeled.csv", 1)
    # vote_2days.predict_class()
    # del vote_2days
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # vote_2days = SampleFile("data\\sample\\labeled", "GBDT_2sample_2days_labeled.csv", 1)
    # vote_2days.predict_class()
    # del vote_2days
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # vote_2days = SampleFile("data\\sample\\labeled", "GBDT_3sample_2days_labeled.csv", 1)
    # vote_2days.predict_class()
    # del vote_2days
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # vote_2days = SampleFile("data\\sample\\labeled", "GBDT_4sample_2days_labeled.csv", 1)
    # vote_2days.predict_class()
    # del vote_2days
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # vote_2days = SampleFile("data\\sample\\labeled", "GBDT_5sample_2days_labeled.csv", 1)
    # vote_2days.predict_class()
    # del vote_2days
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # 以下是七天的数据

    # vote_7days = SampleFile("data\\sample\\labeled", "GBDT_votesample_7days_labeled.csv", 6)
    # vote_7days.predict_class()
    # del vote_7days
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    #
    # vote_7days = SampleFile("data\\sample\\labeled", "sample_7days_labeled.csv", 6)
    # vote_7days.predict_class()
    # del vote_7days
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    #
    # vote_7days = SampleFile("data\\sample\\labeled", "GBDT_0sample_7days_labeled.csv", 1)
    # vote_7days.predict_class()
    # del vote_7days
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # vote_7days = SampleFile("data\\sample\\labeled", "GBDT_1sample_7days_labeled.csv", 1)
    # vote_7days.predict_class()
    # del vote_7days
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # vote_7days = SampleFile("data\\sample\\labeled", "GBDT_2sample_7days_labeled.csv", 1)
    # vote_7days.predict_class()
    # del vote_7days
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # vote_7days = SampleFile("data\\sample\\labeled", "GBDT_3sample_7days_labeled.csv", 1)
    # vote_7days.predict_class()
    # del vote_7days
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # vote_7days = SampleFile("data\\sample\\labeled", "GBDT_4sample_7days_labeled.csv", 1)
    # vote_7days.predict_class()
    # del vote_7days
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # vote_7days = SampleFile("data\\sample\\labeled", "GBDT_5sample_7days_labeled.csv", 1)
    # vote_7days.predict_class()
    # del vote_7days
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
