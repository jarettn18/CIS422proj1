import preprocessing as prep

if __name__ == '__main__':
    #Copy Direct Path of file on your own System
    data = prep.read_from_file("/Users/loganlevitre/Desktop/422/CIS422proj1/TestData/AtmPres2005NovMin.csv")
    print(data.head())
    difference_csv = prep.difference(data)
    print(difference_csv.head())
    output_file_name = input("What would you like to name the output file? ")
    prep.write_to_file(output_file_name, data)
