import preprocessing as prep

if __name__ == '__main__':
    data = "/Users/loganlevitre/Desktop/422/CIS422proj1/TestData/WindSpeed2010Jan20mMin.csv"
    data_op = prep.read_from_file(data)
    cleaned = prep.denoise(data_op)  # works
    #prep.write_to_file("cleaned.csv", cleaned)  # works
    #scaled = prep.scaling(cleaned)  # working
    #prep.write_to_file("scaled.csv", scaled)  # works
    missing_data = prep.read_from_file("/Users/loganlevitre/Desktop/422/CIS422proj1/TestData/missing_data_test.csv")
    longest_run = prep.longest_continuous_run(missing_data)  # no
    prep.write_to_file("longest_run.csv", longest_run)         # no
    #inputs = []
    #outputs = []
    #prep.ts2db(data, 50, 25, 25, inputs, outputs, "test.csv")  # works
    #for x in range(len(inputs)):
    #    print(inputs[x])            # works
    #print("Outputs[]-> ", end='\n')
    #for x in range(len(outputs)):
    #    print(outputs[x])           # works
