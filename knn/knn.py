from numpy import *
import operator
import os


def classify(input, data_set, labels, k):
    data_set_size = data_set.shape[0]
    diff_mat = tile(input, (data_set_size, 1)) - data_set  # find out the difference of X wr.t. dataSet
    sq_diff_mat = diff_mat ** 2  # square off the distances
    sq_distances = sq_diff_mat.sum(axis=1)  # sum the distances across axis-1
    distances = sq_distances ** 0.5  # find out the square root of the square distances
    sorted_distances = distances.argsort()  # sorting distances in ascending order
    class_count = {}
    for i in range(k):  # run voting scheme for k iterations on sorted_distances
        voted_label = labels[sorted_distances[i]]
        class_count[voted_label] = class_count.get(voted_label, 0) + 1  # add the vote to the class
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)  # sort in reverse to get the most voted label
    return sorted_class_count[0][0]  # return the most voted label for k iterations


def normalize(data_set):
    min_vals = data_set.min(0)  # finding the min value of data set
    max_vals = data_set.max(0)  # finding max value of data set
    ranges = max_vals - min_vals  # element wise subtraction to find individual ranges
    norm_data_set = zeros(shape(data_set))  # initialise a normalized matrix of zeroes with the same shape as dataSet
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_vals, (m, 1))  # prepare a normalized data set w.r.t. min values
    norm_data_set = norm_data_set / tile(ranges, (m, 1))  # element wise divide and not matrix division
    return norm_data_set, ranges, min_vals  # return normalized data set, range of data set and min values


def img2vector(filename):
    return_vector = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vector[0, 32 * i + j] = int(line_str[j])
    return return_vector


def hand_writing_test():
    hw_labels = []
    training_file_list = os.listdir('trainingDigits')  # load the training set
    m = len(training_file_list)
    training_mat = zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]  # take off .txt
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        training_mat[i, :] = img2vector('trainingDigits/%s' % file_name_str)
    test_file_list = os.listdir('testDigits')  # iterate through the test set
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]  # take off .txt
        class_num_str = int(file_str.split('_')[0])
        vector_under_test = img2vector('testDigits/%s' % file_name_str)
        classifier_result = classify(vector_under_test, training_mat, hw_labels, 3)  # classifying w.r.t. k=3 voters
        print("the classifier came back with: %d, the real answer is: %d" % (classifier_result, class_num_str))
        if classifier_result != class_num_str:
            error_count += 1.0
    print("\n total number of errors is: %d" % error_count)
    print("\n total error rate is: %f" % (error_count / float(m_test)))


hand_writing_test()
