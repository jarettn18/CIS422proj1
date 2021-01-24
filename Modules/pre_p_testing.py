import preprocessing as prep

if __name__ == '__main__':
    data = prep.read_from_file("/Users/loganlevitre/Desktop/422/CIS422proj1/TestData/AtmPres2005NovMin.csv")
    # data3 = prep.read_from_file("/Users/loganlevitre/Desktop/422/CIS422proj1/TestData/wind_corrales_10m_complete.csv")
    # outliers = prep.read_from_file("/Users/loganlevitre/Desktop/422/CIS422proj1/Modules/ef.csv")
    # long = prep.longest_continuous_run(outliers)
    # no_time = prep.read_from_file("/Users/loganlevitre/Desktop/422/CIS422proj1/TestData/1_temperature_test.csv")
    # no_time = prep.assign_time(no_time, "11/2/2020", 1)
    # norm = prep.standardize(data)
    # output_file_name = input("What would you like to name the output file? ")
    # prep.write_to_file(output_file_name, norm)
