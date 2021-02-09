"""
File: pre_p_testing.py
Class: CIS 422
Date: February 8, 2021
Team: The Nerd Herd
Head Programmer: Logan Levitre
Version 1.1.0

Overview: Testing for preprocessing module
"""

import preprocessing as prep

if __name__ == '__main__':
    tmp = 0
    data = "/Users/loganlevitre/Desktop/422/CIS422proj1/TestData/missing_data_test.csv"
    training = "/Users/loganlevitre/Desktop/422/CIS422proj1/TestData/missing_data_test.csv"
    data_op = prep.read_from_file(training)
    print(data_op)
    dataaa = prep.impute_missing_data(data_op)
    prep.write_to_file("da.csv", dataaa)
    # log = prep.cubic_roots(data_op)
    # prep.write_to_file("cubed.csv", log)
    # prep.write_to_file("logged.csv", prep.logarithm(data_op))
    # prep.split_data(data_op, 50, 25, 25)
    # cleaned = prep.denoise(data_op)
    # clip = prep.clip(data_op, "1/1/2010", "1/5/2010")
    # print(clip)# works
    # prep.write_to_file("cleaned.csv", cleaned)  # works
    # scaled = prep.scaling(cleaned)  # working
    # temp_data2 = "/Users/loganlevitre/Desktop/422/CIS422proj1/TestData/1_temperature_test.csv"
    # data_op_2 = prep.read_from_file(temp_data2)
    # new = prep.assign_time(data_op, "02/02/1998", 2)
    # prep.write_to_file("newtime.csv", new)
    # print(data_op)
    # prep.write_to_file("scaled.csv", scaled)  # works
    # ----------------
    # missing_data = prep.read_from_file("/Users/loganlevitre/Desktop/422/CIS422proj1/TestData/missing_data_test.csv")
    # longest_run = prep.longest_continuous_run(missing_data)  # works
    # prep.write_to_file("longest_run.csv", longest_run)         # works
    # ----------------
    # diff = prep.difference(data_op)
    # prep.write_to_file("diff.csv", diff)
    # ----------------
    # data = "/Users/loganlevitre/Desktop/422/CIS422proj1/TestData/1_temperature_train.csv"
    # data = "/Users/loganlevitre/Desktop/422/CIS422proj1/TestData/FRB_H15 NonFinancial.csv"
    # TS2DB
    # inputs = []
    # outputs = []
    # da = prep.ts2db(data, 50, 25, 25, inputs, outputs, "data_op.csv")
    # print(da)# works
    # ----------------
    # inputs from left to right(up to down) array is of t-x
    # t being time now. i.e input_index[0] = t - 0 = t  , input_index[1]  = t + (-x) = t-x
    # random index to grab data from to compare
    # m_i = []
    # t_i = []
    # m_O = []
    # t_O = []
    # prep.design__matrix(data_op, m_i, t_i, m_O, t_O)
    # print(m_i[0])
    # print(t_i[0])
    # print(m_O[0])
    # print(t_O[0])
    # for idx in range(len(inputs)):
    #    print(inputs[idx])            # works
    # print("Outputs[]-> ", end='\n')
    # array of future Ts index's t i.e t + x = (y)
    # for x in range(len(outputs)):
    #    print(outputs[x])           # works
    # ----------------
