"""
File: node.py
Class: CIS 422
Date: January 30, 2021
Team: The Nerd Herd
Head Programmer: Callista West
Version 0.1.0
Node Class and node implementation
"""

# Represents a node of an n-ary tree using a node Class
class Node:

    # creating a new tree node
    def __init__(self ,key):
        self.key = key
        self.child = []
        self.folder = []


# Prints the n-ary tree level wise
"""
def find_function(operation):
    #if operation == "some key word"
    #    call the correlating function
    #    make sure the altered data replaces the original data
    #    for the next "step"
    print("------")
"""

def add_node(self, parent, kid):
    """
    Function that will add the kid node as a child to the parent as an addition to the tree
    :param self: binding arguments with class Node
        parent: the parent of the newest added tree node (type Node)
        kid: the added new child of the parent (type string)
    : returns None
    """
    if parent is not None:
        parent.child.append(Node(kid))
    else:
        print("Cannot add a child node to a parent that doesn't exist")
        return None

def add_root(rootnode):
    """
    Function will take a string and create a root node for the tree
    :param rootnode: string name of root wanting to be created
    :returns the new root stored in root
    """
    root = Node(rootnode)
    return root
    
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
    #still need to check that old and new nodes are in same "category"
    if old is not None:
        self.folder.append(new)
        self.folder.remove(old)
        return None

class prepNode:

    def __init__(self ,ts, starting_date, final_date, increment):
        self.ts = ts
        self.starting_date = starting_date
        self.final_date = final_date
        self.increment = increment

def execute():
    """
    Use this function when the tree executes
    """

class splitNode:

    def __init__(self, ts, perc_train, prec_valid, perc_test, output_file):
        self.ts = ts
        self.perc_train = perc_train
        self.prec_valid = prec_valid

def execute_tree():
    """
    Function will execute the tree
    """


class visualizeNode:

    def __init__(self, ts):
        self.ts = ts

def execute_tree():
    """
    Function will execute the tree using the DataFram object ts
    """
    
class evalNode:

    def __init__(self, y_test, y_forecast):
        self.y_test = y_test
        self.y_forecast = y_forecast


def execute_tree():
    """
    This function will execute the tree
    """
