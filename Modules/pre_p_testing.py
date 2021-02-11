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
    print("--------Preprocessing Test--------")
    print("            -Denoise()-           ")
      #tests both impute functions
    denoised = prep.denoise(data_op)
    print(denoised)
    # ---------------------
    print("--------Preprocessing Test--------")
    print("          -longest_run()-         ")
    data = prep.read_from_file("/Users/loganlevitre/Desktop/422/CIS422proj1/TestData/missing_data_test.csv")
    longest_run = prep.longest_continuous_run(data)
    print(longest_run)
    # ---------------------
    print("--------Preprocessing Test--------")
    print("             -clip()-             ")
    clipped = prep.clip(longest_run, "9/15/2008", "12/15/2008")
    print(clipped)
    # ---------------------
    print("--------Preprocessing Test--------")
    print("         -assign_time()-          ")
    data = prep.read_from_file(
        "/Users/loganlevitre/Desktop/422/CIS422proj1/TestData/8_distribution_subsampled_train.csv")
    print(data, "\n")
    assigned_time = prep.assign_time(data, "1/10/2019", 1)
    print("With Assigned Time:\n")
    print(assigned_time)
    # ---------------------
    print("--------Preprocessing Test--------")
    print("          -difference()-          ")
    test_data = prep.read_from_file(
        "/Users/loganlevitre/Desktop/422/CIS422proj1/TestData/AtmPres2005NovMin.csv")
    difference_ts = prep.difference(test_data)
    print(difference_ts)
    # ---------------------
    print("--------Preprocessing Test--------")
    print("            -scaling()-           ")
    scaled = prep.scaling(assigned_time)
    print(scaled)
    # ----------------
    print("--------Preprocessing Test--------")
    print("         -standardize()-          ")
    standardized = prep.standardize(test_data)
    print(standardized, "\n")
    # ----------------
    print("--------Preprocessing Test--------")
    print("          -logarithm()-           ")
    log = prep.logarithm(test_data)
    print(log, "\n")
    # ----------------
    print("--------Preprocessing Test--------")
    print("            -cubic()-             ")
    cubed = prep.cubic_roots(test_data)
    print(cubed)


