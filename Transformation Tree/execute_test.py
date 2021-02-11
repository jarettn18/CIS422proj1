"""
File: treetest.py
Class: CIS 422
Date: January 23, 2020
Team: The Nerd Herd
Head Programmers: Jack Sanders
Version 0.1.1
Testing of tree.py, node.py, and save_load.py

To run the test script, run "python ./execute_test.py". There should be no errors if
all functions being tested are working and all packages have been installed.
"""

from tree import Tree
import tree
import save_load as SL


def test_execute_tree():
    print("Begin test_execute_tree\n")
    tree_test = Tree()

    tree_test.add_prep_node([], "denoise", None, None, None)
    tree_test.add_split_node(["denoise"], "ts2db")
    tree_test.add_visualize_node(["denoise"], "box_plot")
    tree_test.add_model_node(["denoise", "ts2db"], "rf")
    tree_test.add_eval_node(["denoise", "ts2db", "rf"], "MSE")

    input_file = '../TestData/4_irradiance_train.csv'

    print(tree_test.execute_tree(input_file))

def main():
    print("----------- Begin testing ------------\n")
    test_execute_tree()

    print("---------- All tests passed ----------\n")


if __name__ == '__main__':
    main()
