"""
File: treetest.py
Class: CIS 422
Date: January 23, 2020
Team: The Nerd Herd
Head Programmers: Callista West, Zeke Petersen
Version 0.1.1
Testing of tree.py and node.py
"""

from tree import Tree
import tree

def test_basics():
    print("Begin test_basics\n")
    tree_test = Tree()

    tree_test.add_prep_node([], "denoise", None, None, None)
    tree_test.add_prep_node(["denoise"], "impute_outliers", None, None, None)
    tree_test.add_prep_node(["denoise"], "impute_missing_data", None, None, None)
    assert tree_test.root.op == "denoise"
    assert tree_test.root.children[0].op == "impute_outliers"
    assert tree_test.root.children[1].parent_op == "denoise"
    assert tree_test.root.children[0].parent_list == ["denoise"]

def test_all_node_types():
    print("Begin test_all_node_types\n")
    tree_test = Tree()

    tree_test.add_prep_node([], "denoise", None, None, None)
    tree_test.add_split_node(["denoise"], "design_matrix", 0)
    tree_test.add_visualize_node(["denoise"], "box_plot")
    tree_test.add_model_node(["denoise", "design_matrix"], "rf", 1, 2, 100)
    tree_test.add_eval_node(["denoise", "design_matrix", "rf"], "MSE")
    assert tree_test.root.op == "denoise"
    assert tree_test.root.children[0].op == "design_matrix"
    assert tree_test.root.children[1].op == "box_plot"
    assert tree_test.root.children[0].children[0].op == "rf"
    assert tree_test.root.children[0].children[0].children[0].op == "MSE"

def main():
    print("----------- Begin testing ------------\n")
    test_basics()
    test_all_node_types()
    print("---------- All tests passed ----------\n")


if __name__ == '__main__':
    main()

    """
    root = nodefile.add_root("read_from_file")
    #root = Node("root")

    prep = Node("preprocessing")
    root.child.append(prep)
    denoise = Node("denoise")
    prep.folder.append(denoise)
    missing_data = Node("impute_missing_data")
    prep.folder.append(missing_data)
    clip = Node("clip")
    mlp = Node("mlp_model")
    rf = Node("rf_model")
    prep.child.append(mlp)
    prep.child.append(rf)
    visual = Node("visualization")
    visual2 = Node("visualization 2")
    mlp.child.append(visual)
    rf.child.append(visual2)

    print("Level order traversal with new nodes added\n")
    test_print = tree1.printNode(root)

    test_replace = nodefile.replace(prep, denoise, clip)

    test_replicate_ST = tree1.replicate_subtree(root, rf)

    test_add_ST = tree1.add_subtree(root, mlp, visual2)

    test_add_node = nodefile.add_node(root, mlp, "new_visual")

    #print_path(root, mlp)

    test_print2 = tree1.printNode(root)
    """
