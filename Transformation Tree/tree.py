"""
File: tree.py
Class: CIS 422
Date: January 20, 2021
Team: The Nerd Herd
Head Programmer: Callista West
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

def replicate_subtree(self, root_node):
    """
    Replicating a Subtree and printing it using printNode
    : param self: binding arguments with class Node
            root_node: the root of the subtree wishing to be replicated
    : return: returns the root of the newly replicated subtree
    """
    print("This is the new root of a new subtree: " + root_node.key)
    now_root = Node(root_node.key)
    for index, value in enumerate(root_node.child):
        now_root.child.append(value)
        #print(value.key)
    #now_root.child = root_node.child
    printNode(now_root)
    return(now_root)

def add_subtree(self, tree_root, new_node):
    """
    Function takes the root of the subtree and adds it and the subtree to a new
        node to continue to length and layers of the original tree
        This function calls replicate_subtree()
    :param self: binding arguments with class Node
           tree_root: root of the subtree wanting to be replicated and added to a
               new node
           new_node: the node in which the subtree will be added to
    :return: returns none
    """
    #need to check that subtree can be added into an appropriate place
    #   in the tree
    holder = replicate_subtree(self, tree_root)
    new_node.child.append(holder)
    printNode(self)
    return None

def save_tree(root):
    """
    Tree starting at specified root will be create and saved to a CSV? file
    :param root: the root of the tree the user is wanting saved
    :return: ??? do we want a CSV file?
    """

def save_pipeline(root, end):
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
		



