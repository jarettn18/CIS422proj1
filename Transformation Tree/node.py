"""
File: node.py
Class: CIS 422
Date: January 30, 2021
Team: The Nerd Herd
Head Programmer: Callista West
Version 0.1.0
Node Class and node implementation
"""

# ------- Begin Node Classes --------------

"""
Notes from ZP on 2/1/2021
- We may want to also be passing the list of children (represented by their
operators) to the super() calls and the init signatures of the child classes
- I did not yet have time to check to make sure all the attributes set in each
init function is complete
- There are a handful of standalone functions that should probably be in the
tree file

- My edits for now are merely trying to represent the inheritance I had in mind
"""

# Represents a node of an n-ary tree using a node Class
class Node:
    # creating a new tree node
    def __init__(self, op, parent_op):
        self.op = op
        self.parent_op = parent_op
        self.child = []
        self.folder = []    # May remove in favor of global tables of function names for each class

    def execute():
        pass

class prepNode(Node):

    def __init__(self, op, parent_op, ts, starting_date, final_date, increment):
        super().__init__(op, parent_op)
        self.ts = ts
        self.starting_date = starting_date
        self.final_date = final_date
        self.increment = increment

    def execute():
        pass
        """
        Use this function when the tree executes
        """

class modelNode(Node):

    def __init__(self, op, parent_op, x_train, y_train, input_dimension, output_dimension, hidden_layers):
        super().__init__(op, parent_op)
        self.x_train = x_train
        self.y_train = y_train
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_layers = hidden_layers

    def execute():
        pass
    """
    Function will execute the tree
    """

class splitNode(Node):

    def __init__(self, op, parent_op, ts, perc_train, prec_valid, perc_test, output_file):
        super().__init__(op, parent_op)
        self.ts = ts
        self.perc_train = perc_train
        self.prec_valid = prec_valid
        #Incomplete

    def execute():
        pass
    """
    Function will execute the tree
    """


class visualizeNode(Node):

    def __init__(self, op, parent_op, ts):
        super().__init__(op, parent_op)
        self.ts = ts

    def execute():
        pass
    """
    Function will execute the tree using the DataFram object ts
    """

class evalNode(Node):

    def __init__(self, op, parent_op, y_test, y_forecast):
        super().__init__(op, parent_op)
        self.y_test = y_test
        self.y_forecast = y_forecast


    def execute():
        pass
    """
    This function will execute the tree
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
        parent.child.append(Node(kid))
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
    #still need to check that old and new nodes are in same "category"
    if old is not None:
        self.folder.append(new)
        self.folder.remove(old)
        return None

# Prints the n-ary tree level wise
"""
def find_function(operation):
    #if operation == "some op word"
    #    call the correlating function
    #    make sure the altered data replaces the original data
    #    for the next "step"
    print("------")
"""
