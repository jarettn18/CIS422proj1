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

import sys
sys.path.append('../Modules')
for p in sys.path:
    print(p)
import preprocessing as prep
import visualization as viz
import model as mod

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

    def execute(self):
        # pass
        """
        This function will help execute the tree
        """
        if self.op == "denoise":
            self.ts = prep.denoise(self.ts)
        elif self.op == "impute_missing_data":
            prep.impute_missing_data(self.ts)
        elif self.op == "impute_outliers":
            prep.impute_outliers(self.ts)
        elif self.op == "longest_continuous_run":
            prep.longest_continuous_run(self.ts)      
        elif self.op == "clip":
            self.ts = pre.denoise(self.ts, self.starting_date, self.starting_date, self.final_date)
        elif self.op == "assign_time":
            prep.assign_time(self.ts, self.starting_date, self.increment)
        elif self.op == "difference":
            prep.difference(self.ts)
        elif self.op == "scaling":
            prep.scaling(self.ts)
        elif self.op == "standardize":
            prep.standardize(self.ts)
        elif self.op == "logarithm":
            prep.logarithm(self.ts)  
        elif self.op == "cubic_roots":
            prep.cubic_roots(self.ts)
        return self.ts

class modelNode(Node):

    def __init__(self, op, input_dimension, output_dimension, hidden_layers):
        super().__init__(op, None)
        self.x_train = None
        self.y_train = None
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_layers = hidden_layers

    def execute():
        # pass
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
        # pass
        """
        This function will help execute the tree
        "plot", "histogram", "summary", "box_plot", "shapiro_wilk",
            "d_agostino", "anderson_darling", "qq_plot"]
        """
        return_value = None
        if self.op == "plot":
            viz.plot(self.ts)
        elif self.op == "histogram":
            viz.histogram(self.ts)
        elif self.op == "summary":
            return_value = viz.summary(self.ts)
        elif self.op == "box_plot":
            viz.box_plot(self.ts)
        elif self.op == "shapiro_wilk":
            return_value = viz.shapiro_wilk(self.ts)
        elif self.op == "d_agostino":
            return_value = viz.d_agostino(self.ts)
        elif self.op == "anderson_darling":
            return_value = viz.anderson_darling(self.ts)
        elif self.op == "qq_plot":
            viz.qq_plot(self.ts)
        return return_value

class evalNode(Node):

    def __init__(self, op):
        super().__init__(op, None)
        self.y_test = None
        self.y_forecast = None

    def execute():
        # pass
        """
        This function will help execute the tree
        """
        return_value = None
        if self.op == "MSE":
            return_value = viz.MSE(self.ts)
        elif self.op == "RMSE":
            return_value = viz.RMSE(self.ts)
        elif self.op == "MAPE":
            return_value = viz.MAPE(self.ts)
        elif self.op == "sMAPE":
            return_value = viz.sMAPE(self.ts)
        return return_value   

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
