import tree as tree1
#from tree import Node
import node as nodefile
from node import Node

if __name__ == '__main__':

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
