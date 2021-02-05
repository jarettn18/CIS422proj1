"""
File: node.py
Class: CIS 422
Date: January 30, 2021
Team: The Nerd Herd
Head Programmers: Callista West, Zeke Petersen
Version 0.1.1
Node Class and node implementation
"""

# ------- Begin Node Classes --------------

"""
Notes from ZP on 2/2/2021

- There are a handful of standalone functions that should probably be in the
tree file
"""

# Represents a node of an n-ary tree using a node Class
class Node:
    # creating a new tree node
    def __init__(self, op, parent_op):
        self.op = op
        self.parent_op = parent_op
        self.children = []         # Filled in during execution

        # Filled in when added? May run into problems when edits are made
        self.parent_list = []

    def execute():
        pass

class prepNode(Node):

    def __init__(self, op, starting_date, final_date, increment):
        super().__init__(op, None)
        self.ts = None    # Filled in during execution

        # Optional depending on op value
        self.starting_date = starting_date
        self.final_date = final_date
        self.increment = increment

    def execute():
        pass
        """
        This function will help execute the tree
        """

class modelNode(Node):

    def __init__(self, op, input_dimension, output_dimension, hidden_layers):
        super().__init__(op, None)
        self.x_train = None
        self.y_train = None
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_layers = hidden_layers

    def execute():
        pass
    """
    This function will help execute the tree
    """

class splitNode(Node):

    def __init__(self, op, prev_index):
        super().__init__(op, None)
        self.ts = None  # Filled in during execution
        self.prev_index = prev_index

    def execute():
        pass
    """
    This function will help execute the tree
    """


class visualizeNode(Node):

    def __init__(self, op):
        super().__init__(op, None)
        self.ts = None

    def execute():
        pass
    """
    This function will help execute the tree
    """

class evalNode(Node):

    def __init__(self, op):
        super().__init__(op, None)
        self.y_test = None
        self.y_forecast = None

    def execute():
        pass
    """
    This function will help execute the tree
    """

# ------- End Node Classes --------------



# Probably should be added as a tree method, not a standalone function -- ZP
def add_node(self, parent, kid):
    """
    Function that will add the kid node as a child to the parent as an addition to the tree
    :param self: binding arguments with class Node
        parent: the parent of the newest added tree node (type Node)
        kid: the added new child of the parent (type string)
    : returns None
    """
    if parent is not None:
        parent.children.append(Node(kid))
    else:
        print("Cannot add a child node to a parent that doesn't exist")
        return None

# Not entirely sure how this works, but leaving for now -- ZP
def add_root(rootnode):
    """
    Function will take a string and create a root node for the tree
    :param rootnode: string name of root wanting to be created
    :returns the new root stored in root
    """
    root = Node(rootnode)
    return root

# Should probably be a tree method with more checks on the operator
# Instead of folder, we should probably define external tables to do checks
# (assuming that is what the folder was for) -- ZP
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
# Prints the n-ary tree level wise
"""
def find_function(operation):
    #if operation == "some op word"
    #    call the correlating function
    #    make sure the altered data replaces the original data
    #    for the next "step"
    print("------")
"""
