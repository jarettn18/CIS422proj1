"""
File: tree.py
Class: CIS 422
Date: January 20, 2021
Team: The Nerd Herd
Head Programmers: Logan Levitre, Zeke Peterson, Jarett Nishijo, Callista West, Jack Sanders
Version 0.1.0

Basic Tree implementation
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
        #if old.key == new.key:
        #old = new
        self.folder.append(new)
        self.folder.remove(old)
        
        #print(old.key)
        return None
    #for child in old.child:
     #   self.replace(child, new)

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
		


#building a tree for testing purposes

root = Node("root")

prep = Node("preprocessing")
root.child.append(prep)
denoise = Node("denoise")
prep.folder.append(denoise)
missing_data = Node("impute_missing_data")
prep.folder.append(missing_data)
clip = Node("clip")
                   
#model = Node("modeling")
#prep.child.append(model)
mlp = Node("mlp_model")
rf = Node("rf_model")
prep.child.append(mlp)
prep.child.append(rf)
visual = Node("visualization")
visual2 = Node("visualization 2")
mlp.child.append(visual)
rf.child.append(visual2)
#model.folder.append(mlp)
#root.child.append(Node("denoise")) 
#root.child.append(Node("impute_missing_data"))
"""
root.child.append(Node("impute_outliers")) 
root.child.append(Node("longest_continuous_run"))
root.child.append(Node("clip"))
root.child.append(Node("assign_time"))
root.child.append(Node("difference"))
root.child.append(Node("scaling"))
root.child.append(Node("standardize"))
root.child.append(Node("logarithm"))
root.child.append(Node("cubic_root"))
root.child.append(Node("split_data"))
root.child.append(Node("design_matrix"))

root.child[2].child.append(Node("mlp_model")) 
root.child[2].child.append(Node("mlp.fit")) 
root.child[2].child.append(Node("mlp.forecast")) 
root.child[2].child.append(Node("rf_model()"))
root.child[2].child.append(Node("rf.fit"))
root.child[2].child.append(Node("rf.forecast"))


root.child[3].child.append(Node("mlp_model")) 
root.child[3].child.append(Node("mlp.fit")) 
root.child[3].child.append(Node("mlp.forecast")) 
root.child[3].child.append(Node("rf_model()"))
root.child[3].child.append(Node("rf.fit"))
root.child[3].child.append(Node("rf.forecast"))

root.child[2].child[2].child.append(Node("plot"))
root.child[2].child[2].child.append(Node("histogram"))
root.child[2].child[2].child.append(Node("box_plot"))
root.child[2].child[2].child.append(Node("normality_test"))
root.child[2].child[2].child.append(Node("mse"))
root.child[2].child[2].child.append(Node("mape"))
root.child[2].child[2].child.append(Node("smape"))
"""

print("Level order traversal with new nodes added\n")
printNode(root)

replace(prep, denoise, clip)

replicate_subtree(root, rf)

add_subtree(root, mlp, visual2)

#print_path(root, mlp)

printNode(root)
