"""
File: tree.py
Class: CIS 422
Date: January 20, 2021
Team: The Nerd Herd
Head Programmers: Callista West, Zeke Petersen, Jack Sanders, Jarett Nishijo
Version 0.1.0

Basic Tree implementation
"""

import node as nodefile
from node import Node
from node import prepNode
from node import splitNode
from node import modelNode
from node import visualizeNode
from node import evalNode
import pandas as pd
import copy
# Splits section likely to change later
OPS = {
       "preps": ["denoise", "impute_missing_data", "impute_outliers",
            "longest_continuous_run", "clip", "assign_time", "difference",
            "scaling", "standardize", "logarithm", "cubic_roots"],
       "splits": ["ts2db"],
       "models": ["mlp", "rf"],
       "evals": ["MSE", "RMSE", "MAPE", "sMAPE"],
       "visualizes":["plot", "histogram", "summary", "box_plot", "shapiro_wilk",
            "d_agostino", "anderson_darling", "qq_plot"]
      }

class Tree:
    def __init__(self):
        self.infile = None    # Filled in at execute stage
        self.root = None      # Eventually will be a node object

    # helper function
    def _find_node(self, op_list):
        """
        :op_list: list of function operator strings, represent real functions

		Starting from the root, trace node path by their op value in order.
		Returns an existing Node with the last op in the list (may be any of the 5).
		Returns None if node path is invalid or node doesn’t exist.
        """

        if self.root is None:
            print("Tree empty\n")
            return None

        l = len(op_list)

        if l == 0:
            print("No node specified\n")
            return None

        else:
            curr_node = self.root

            # Check that first op is OK
            if op_list[0] != curr_node.op:
                print("Invalid node path\n")
                return None

            # go through each string of the op_list
            for i in range(l - 1):
                # Go through the children list
                found = 0
                for j in range(len(curr_node.children)):
                    # Does a child match the next op?
                    if curr_node.children[j].op == op_list[i + 1]:
                        curr_node = curr_node.children[j]
                        found = 1
                        break
                if found == 0:
                    print("Invalid node path\n")
                    return None
            # Last node was legal
            return curr_node

    # Any node type can be the root for later purposes of adding subtrees
    # Ignores the op_list in this case
    def _add_node(self, op_list, node):
        """
        :op_list: list of function operator strings, represent real functions
        :node: Node class object, will be any of the 5 subclasses

		If the root is None, sets the node passed as the root, regardless of the
        op_list.
        If the root is not None and the op_list represents a valid path, will
        return the Node containing the last operator of the op_list.
        If the root is not None and the op_list does not represent a valid path,
        will raise an Exception.
        """

        # If the tree is empty, we just add the node to the root
        if self.root is None:
            self.root = node
        else:
            # Fill in some of the node data
            found_node = self._find_node(op_list)
            if found_node is not None:
                node.parent_list = op_list
                node.parent = found_node
                found_node.children.append(node)
            else:
                raise Exception("Unable to add node\n")


    def _print_tree(self):
        pass

    def add_prep_node(self, op_list, op, starting_date, final_date, increment):
        """
        For arguments, see the prepNode class __init__

        Adds a new prepNode to the node specified by the op_list. Raises an
        exception if not valid or not possible.
        """

        if op not in OPS["preps"]:
            raise Exception("Invalid prep operator\n")

        # Can only add prep node after another prep node
        if len(op_list) != 0:
            if op_list[-1] not in OPS["preps"]:
                raise Exception("Illegally adding prep node after wrong node type\n")

        new_node = prepNode(op, starting_date, final_date, increment)

        self._add_node(op_list, new_node)

    def add_split_node(self, op_list, op):
        """
        For arguments, see the splitNode class __init__

        Adds a new splitNode to the node specified by the op_list. Raises an
        exception if not valid or not possible.
        """

        if op not in OPS["splits"]:
            raise Exception("Invalid split operator\n")

        # Can only add a split node after a prep node
        if len(op_list) != 0:
            if op_list[-1] not in OPS["preps"]:
                raise Exception("Illegally adding split node after wrong node type\n")

        new_node = splitNode(op)

        self._add_node(op_list, new_node)

    def add_model_node(self, op_list, op):
        """
        For arguments, see the modelNode class __init__

        Adds a new modelNode to the node specified by the op_list. Raises an
        exception if not valid or not possible.
        """

        if op not in OPS["models"]:
            raise Exception("Invalid model operator\n")

        # Can only add a model node after a split node
        if len(op_list) != 0:
            if op_list[-1] not in OPS["splits"]:
                raise Exception("Illegally adding model node after wrong node type\n")

        new_node = modelNode(op)

        self._add_node(op_list, new_node)

    def add_eval_node(self, op_list, op):
        """
        For arguments, see the evalNode class __init__

        Adds a new evalNode to the node specified by the op_list. Raises an
        exception if not valid or not possible.
        """

        if op not in OPS["evals"]:
            raise Exception("Invalid eval operator\n")

        # Can only add an eval node after a model node
        if len(op_list) != 0:
            if op_list[-1] not in OPS["models"]:
                raise Exception("Illegally adding eval node after wrong node type\n")

        new_node = evalNode(op)

        self._add_node(op_list, new_node)

    def add_visualize_node(self, op_list, op):
        """
        For arguments, see the visualizeNode class __init__

        Adds a new visualizeNode to the node specified by the op_list. Raises an
        exception if not valid or not possible.
        """

        if op not in OPS["visualizes"]:
            raise Exception("Invalid visualize operator\n")

        # Can only add a visualize node after a prep node
        if len(op_list) != 0:
            if op_list[-1] not in OPS["preps"]:
                raise Exception("Illegally adding visualize node after wrong node type\n")

        new_node = visualizeNode(op)

        self._add_node(op_list, new_node)

    """
    Can assume the tree has valid node ordering, though will need to check
    to ensure the root is a prep (or I suppose a split, if we feed raw data)

    prep -> split -> model -> eval
        \-> visualize

    Can implement the execute functions of each class to simplify this function
    """
        
    def execute_tree(self, infile):

        root = self.root
        if root is None:
            return
        if root.op not in OPS["preps"]:
            raise Exception("Root not PrepNode")

        # create a queue and enqueue root to it using append
        queue = []
        queue.append(root)
        model = None

        # Carry out level order traversal.
        # The double while loop is used to make sure that
        #     each level is printed on a different line
        print("------------") # extra space in between levels
        while(len(queue) >0):

            n = len(queue)
            while(n > 0):
                # Dequeue an item from the queue and print it
                p = queue[0]
                queue.pop(0)

                print(p.op)
                if p.op in OPS["preps"]:
                    time_series = prepNode.execute(infile, p.op)
                    print(time_series)
                elif p.op in OPS["splits"]:
                    train_ts, train_inputs, test_ts, test_inputs = splitNode.execute(infile, p.op)
                elif p.op in OPS["models"]:
                    model = modelNode.execute(train_ts, train_inputs, test_inputs, p.op)
                    print(model)
                elif p.op in OPS["evals"]:
                    eval = evalNode.execute(infile, model, p.op)
                    print(eval)
                elif p.op in OPS["visualizes"]:
                    visualize = visualizeNode.execute(infile, p.op)
                    print(visualize)
                # Enqueue all of the children of the dequeued node
                for index, value in enumerate(p.children):
                    queue.append(value)
                n -= 1
            print("------------") # extra space in between levels

        print(".................end of tree...............\n")

        

    def replicate_subtree(self, op_list):
        """
        Creates an identical subtree
        :param op_list: list of function operator strings, represent real functions
        :return: a Tree obj that is a copy of another subtree
        """
        if self.root is not None:
            # finding the node from op_list
            new_root = self._find_node(op_list)
            # make deep copy of node
            if new_root is not None:
                # create new tree obj
                new_tree = Tree()
                # deep copy node
                deep_copy = copy.deepcopy(new_root)
                # add node to new tree
                new_tree._add_node(op_list, deep_copy)
                return new_tree
            else:
                raise Exception("Error cannot replicate subtree")


    def replicate_path(self, op_list):
        """
        creates a new tree that consists of a path of another tree
        :param op_list: list of function operator strings, represent real functions
        :return: a new Tree object consisting of a path in another tree
        """
        # create new Tree for path
        new_path = Tree()
        # op_list: list of function operator strings, represent real functions
        # loop through list to get individual nodes one by one\
        for idx in range(len(op_list)):
            # check if node at idx in list is in original tree
            found_node = self._find_node(op_list[:idx+1])
            if found_node is not None:
                # if in tree, create new node
                deep_copy = copy.deepcopy(found_node)
                # reset children to nothing
                deep_copy.children = []
                # add node to tree
                new_path._add_node(op_list[:idx], deep_copy)
        return new_path


    def add_subtree(self, op_list, subtree):
        """
        Adds a subtree to a node
        :param op_list: list of function operator strings, represent real functions
        :param subtree: part of a tree to be added onto a node
        :return: original tree with updated children of specified node
        """
        # find the node
        if self.root is not None:
            found_node = self._find_node(op_list)
            if found_node is not None:
                # create deep copy of root node of subtree
                deep_copy = copy.deepcopy(subtree.root)
                # append copied node to the found nodes children
                found_node.children.append(deep_copy)
            else:
                raise Exception("Unable to add subtree")

    def replace_operator(self, op_list, node):
	    # Find Node in Op Liust
        dict_list = list(OPS.values())
        val_list = None
        node_to_find = None

        # Find correct list from OPS
        for i in range(len(dict_list)):
            for j in range(len(dict_list[i])):
                if (node == dict_list[i][j]):
                    val_list = dict_list[i]
                    break
        if val_list is None:
            raise Exception("Node not found in OP dictionary")

        # Check that the node op being replaced is from same list
        for i in range(len(val_list)):
            if val_list[i] == op_list[len(op_list) - 1]:
                node_to_find = val_list[i]
                break
        if node_to_find is None:
            raise Exception("Replacement Node is not of same type as Node to be replaced")
        found_node = self._find_node(op_list)
        found_node.op = node


# END TREE DEFINITION --------------------------

# Below here are functions that may need to be modified and moved
# to the Tree class -- these were original ideas for how to implement tree functions
#-----------------------------------------------------------------------------------

def save_tree(root):
    pass
    """
    Tree starting at specified root will be create and saved to a CSV? file
    :param root: the root of the tree the user is wanting saved
    :return: ??? do we want a CSV file?
    """

def save_pipeline(root, end):
    pass
    """
    Takes a pipeline from a given root and a given end and saves/stores the
        pipeline
    :param root: the root of the pipeline
           end: the last node/leaf of the pipeline
    : return: ?? CSV file? json?
    """

"""
def replicate_path(self, array, end_node):

    Function will start the path at the root of the tree and finish at end_node,
        the parameter of the function
    :param self: binding arguments with class Node
           end_node: the last node/leaf of the path, where the path comes to a stop
    :return: returns none
"""

"""
    if(not roots):
        print("Cannot have a path without a root")
        return False
"""
"""
    #array.append(root)
    print("THIS IS THE ROOT: " + self.key)
    root_node = Node(self.key)
    print("made it here")
    end_nodes = Node(end_node.key)
    array.append(self)
    print(array[0])

    if(root_node.key == end_nodes.key):
        print("this is true")
        return True
    if(replicate_path(root_node.child, array, end_nodes)):
        print("trying this function")
        return True
    print("anything here")
    array.pop(-1)
    return False
def print_path(root, ending):
    array = []
    if(replicate_path(root, array, ending)):
        for i in range(len(array)-1):
            print(array[i], ending = "->")
        print(array[len(array)-1])
    else:
        print("no path found")


"""



def printNode(root):
    # if there is no root, return none
    if root is None:
        return

    # create a queue and enqueue root to it using append
    queue = []
    queue.append(root)

    # Carry out level order traversal.
    # The double while loop is used to make sure that
    #     each level is printed on a different line

    while(len(queue) >0):

        n = len(queue)
        while(n > 0):
            # Dequeue an item from the queue and print it
            p = queue[0]
            queue.pop(0)
            #if(p.key == "root"):
                #print("root found")

            print(p.key)
            # Enqueue all of the children of the dequeued node
            for index, value in enumerate(p.child):
                #for index, value in enumerate(p.folder):
                queue.append(value)
            for index, value in enumerate(p.folder):
                queue.append(value)

            n -= 1
        print("---------------") # extra space in between levels

    print(".................end of tree...............\n")


def replace(self, old, new):
    """
    Function to replace a process step with a different operator, such as
        replacing denoise() with clip()
    :param self: binding arguments with class Node
           old: the orignal process step that will be replaced
           new: the different operator, that will be taking the place of the original
               operator
    : return: returns None
    """
    pass
    """
    #still need to check that old and new nodes are in same "category"
    if old is not None:
        self.folder.append(new)
        self.folder.remove(old)
        return None
    """
