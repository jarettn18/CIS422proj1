"""
File: treetest.py
Class: CIS 422
Date: January 23, 2020
Team: The Nerd Herd
Head Programmers: Zeke Petersen, Callista West
Version 0.1.1
Testing of tree.py, node.py, and save_load.py

To run the test script, run "python ./execute_test.py". There should be no errors if
all functions being tested are working and all packages have been installed.
"""

from tree import Tree
import tree
import save_load as SL


def test_basics():
    print("Begin test_basics\n")
    tree_test = Tree()

    tree_test.add_prep_node([], "denoise", None, None, None)
    tree_test.add_prep_node(["denoise"], "impute_outliers", None, None, None)
    tree_test.add_prep_node(["denoise"], "impute_missing_data", None, None, None)
    assert tree_test.root.op == "denoise"
    assert tree_test.root.children[0].op == "impute_outliers"
    assert tree_test.root.children[1].parent == tree_test.root
    assert tree_test.root.children[0].parent_list == ["denoise"]

def test_all_node_types():
    print("Begin test_all_node_types\n")
    tree_test = Tree()

    tree_test.add_prep_node([], "denoise", None, None, None)
    tree_test.add_split_node(["denoise"], "ts2db")
    tree_test.add_visualize_node(["denoise"], "box_plot")
    tree_test.add_model_node(["denoise", "ts2db"], "rf")
    tree_test.add_eval_node(["denoise", "ts2db", "rf"], "MSE")
    assert tree_test.root.op == "denoise"
    assert tree_test.root.children[0].op == "ts2db"
    assert tree_test.root.children[1].op == "box_plot"
    assert tree_test.root.children[0].children[0].op == "rf"
    assert tree_test.root.children[0].children[0].children[0].op == "MSE"

def test_replicate_path():
    print("Begin test_replicate_path\n")
    tree_test = Tree()

    tree_test.add_prep_node([], "denoise", None, None, None)
    tree_test.add_split_node(["denoise"], "ts2db")
    tree_test.add_visualize_node(["denoise"], "box_plot")
    tree_test.add_model_node(["denoise", "ts2db"], "rf")
    tree_test.add_eval_node(["denoise", "ts2db", "rf"], "MSE")

    rep = tree_test.replicate_path(["denoise", "ts2db", "rf"])

    assert rep.root.op == "denoise"
    assert rep.root.children[0].op == "ts2db"
    assert len(rep.root.children) == 1
    assert rep.root.children[0].children[0].op == "rf"

def test_replicate_subtree():
    print("Begin test_replicate_subtree\n")
    tree_test = Tree()

    tree_test.add_prep_node([], "denoise", None, None, None)
    tree_test.add_split_node(["denoise"], "ts2db")
    tree_test.add_visualize_node(["denoise"], "box_plot")
    tree_test.add_model_node(["denoise", "ts2db"], "rf")
    tree_test.add_eval_node(["denoise", "ts2db", "rf"], "MSE")

    rep = tree_test.replicate_subtree(["denoise", "ts2db", "rf"])

    assert rep.root.op == "rf"
    assert rep.root.children[0].op == "MSE"

def test_add_subtree():
    print("Begin test_add_subtree\n")
    tree_test = Tree()
    tree_test.add_prep_node([], "denoise", None, None, None)
    tree_test.add_split_node(["denoise"], "ts2db")

    tree_add = Tree()
    tree_add.add_model_node([], "rf")
    tree_add.add_eval_node(["rf"], "MSE")

    tree_test.add_subtree(["denoise", "ts2db"], tree_add)

    assert tree_test.root.op == "denoise"
    assert tree_test.root.children[0].op == "ts2db"
    assert tree_test.root.children[0].children[0].op == "rf"
    assert tree_test.root.children[0].children[0].children[0].op == "MSE"


def test_replace_operator():
    print("Begin test_replace_operator\n")
    tree_test = Tree()

    tree_test.add_prep_node([], "denoise", None, None, None)
    tree_test.add_split_node(["denoise"], "ts2db")
    tree_test.add_model_node(["denoise", "ts2db"], "rf")
    tree_test.add_eval_node(["denoise", "ts2db", "rf"], "MSE")

    tree_test.replace_operator(["denoise", "ts2db", "rf"], "mlp")

    assert tree_test.root.children[0].children[0].op == "mlp"

def test_load_save():
    print("Begin test_load_save\n")
    tree_test = Tree()

    tree_test.add_prep_node([], "denoise", None, None, None)
    tree_test.add_split_node(["denoise"], "ts2db")
    tree_test.add_visualize_node(["denoise"], "box_plot")
    tree_test.add_model_node(["denoise", "ts2db"], "rf")
    tree_test.add_eval_node(["denoise", "ts2db", "rf"], "MSE")
    SL.save_tree(tree_test, "testingtree")
    tree_loaded = SL.load_tree("testingtree")
    assert tree_loaded.root.op == "denoise"
    assert tree_loaded.root.children[0].op == "ts2db"
    assert tree_loaded.root.children[1].op == "box_plot"
    assert tree_loaded.root.children[0].children[0].op == "rf"
    assert tree_loaded.root.children[0].children[0].children[0].op == "MSE"

def test_execute_tree():
    print("Begin test_execute_tree\n")
    tree_test = Tree()

    tree_test.add_prep_node([], "denoise", None, None, None)
    tree_test.add_split_node(["denoise"], "ts2db")
    tree_test.add_visualize_node(["denoise"], "box_plot")
    tree_test.add_model_node(["denoise", "ts2db"], "rf")
    tree_test.add_eval_node(["denoise", "ts2db", "rf"], "MSE")
    
    print(tree_test.execute_tree())

def main():
    print("----------- Begin testing ------------\n")
    test_execute_tree()

    print("---------- All tests passed ----------\n")


if __name__ == '__main__':
    main()
