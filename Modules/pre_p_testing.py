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
    print("--------Preprocessing Test 1--------")
    print("       ----Reading file-----         ")
    training = "/Users/loganlevitre/Desktop/422/CIS422proj1/TestData/missing_data_test.csv"
    data_op = prep.read_from_file(training)
    assert data_op is not None
    # ---------------------
    print("     ----Printing Content:----\n")
    print(data_op)
    print("--------Preprocessing Test 2--------")
    print("         ----Denoise()----          ")
    # tests both impute functions
    denoised = prep.denoise(data_op)
    print(denoised)
    # ---------------------
    print("--------Preprocessing Test 2--------")
    print("         ----longest_run()----      ")
    data = prep.read_from_file("/Users/loganlevitre/Desktop/422/CIS422proj1/TestData/missing_data_test.csv")
    longest_run = prep.longest_continuous_run(data)
    print(longest_run)
    # ---------------------
    print("--------Preprocessing Test 3--------")
    print("           ----clip()----           ")
    clipped = prep.clip(longest_run, "9/15/2008", "12/15/2008")
    print(clipped)
    # ---------------------
    print("--------Preprocessing Test 4--------")
    print("       -----assign_time()-----      ")
    data = prep.read_from_file("/Users/loganlevitre/Desktop/422/CIS422proj1/TestData/assign_time_test.csv")
    print(data)
    assigned_time = prep.assign_time(data, 0, 1)
    print(assigned_time)










    # data = "/Users/loganlevitre/Desktop/422/CIS422proj1/TestData/1_temperature_train.csv"
    # data = "/Users/loganlevitre/Desktop/422/CIS422proj1/TestData/FRB_H15 NonFinancial.csv"
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
