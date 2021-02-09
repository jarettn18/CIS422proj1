## Project 1: Time Series Analysis Support For Data Scientists
---------------------------------------------------------------
README containing a description of the software and directory structure,
dependencies and instructions for installation, and a short user guide.

CIS 422 - Software Methodologies 1

Creation date: 1/15/2021

Team: The Nerd Herd

Author: Zeke Petersen


### Project Description
-----------------------
The system contained in this repository is intended to be a tool used by Data
Scientists to rapidly test Time Series pipelines. Once tested, successful
pipelines can be saved to go into production to make actionable forecasts.


### Directory Structure
-----------------------
* ./Modules - contains the primary modules, including testing and helper functions
* ./Logistics - contains internal scheduling deadlines, team meeting notes,
and internal code standards
* ./Documentation - contains the SRS and SDS
* ./TestData - contains input .csv files used to test the various modules
* ./Transformation Tree - contains node and tree class definitions and methods,
along with tree saving and loading functions
    * /pickle_objects - contains any saved Tree objects


### Dependencies and Instructions for Installation
--------------------------------------------------
In order to run the software, Python 3.8 or later must be installed. Installers
can be found at python.org/downloads. To check that Python successfully installed,
enter "python --version" at the command line or terminal.

Note: For some operating systems, Python 2 may remain the default even after installation
of Python 3. If that is the case, it may be necessary to alias 'python' to 'python3'
in your terminal profile or establish a virtual environment via pyenv:
https://github.com/pyenv/pyenv

In either case, the end result of calling "python --version" should be:
> Python 3.x.x

Pip should come preinstalled with Python and it will be needed to install the
software dependencies. To check that pip successfully installed,
enter "python -m pip --version" at the command line or terminal.

Once confirmed, run "python -m pip install -U pip" to ensure pip is upgraded.

* pip is used to install dependencies via the command "pip install \<dependency\>"
where each of the following will be substituted for \<dependency\> one at a time
(ex: entering "pip install scikit-learn" and then "pip install pandas" at a command line or
terminal).
    * scikit-learn
    * pandas
    * numpy         (may already be installed as previous package dependency)
    * pyjanitor
    * seaborn       (may already be installed as previous package dependency)
    * matplotlib    (may already be installed as previous package dependency)
    * statsmodels

### User Guide
--------------
### **Setup**

To use this software, it is assumed that this repository has been cloned onto
the user's local machine and that the required installations described above
have been completed.

To use the entirety of the Transformation Tree library, add the following
statements to the top of your Python file:

> import tree

> import save_load

Any function calls below are merely examples for how to call the various methods
and functions available to the programmer.

### **Create a Transformation Tree**

To create a new Transformation Tree, invoke the constructor of the Tree class:

> my_tree = Tree()

### **Add nodes**

To add nodes to your newly created Transformation Tree, invoke any of the five
methods available for that purpose:

> my_tree.add_prep_node([], "denoise", None, None, None)

> my_tree.add_split_node(["denoise"], "ts2db")

> my_tree.add_visualize_node(["denoise"], "box_plot")

> my_tree.add_model_node(["denoise", "ts2db"], "rf")

> my_tree.add_eval_node(["denoise", "ts2db", "rf"], "MSE")

Keep in mind that the Tree class enforces the ordering of nodes such that:
* splitNodes, visualizeNodes, and prepNodes must either be the root, or
immediately follow a prepNode.
* modelNodes must either be the root, or immediately follow a splitNode
* evalNodes must either be the root, or immediately follow a modelNode

The first argument of each call represents a list of operators of existing nodes
that represent the path to the node after which you wish to add another node.

### **Replacing a process step**

To replace an operator of an existing node, invoke the replace_operator
method:

> my_tree.replace_operator(["denoise", "ts2db", "rf"], "mlp")

The first argument represents a list of operators of existing nodes that
represent the path to the node you wish to modify. The second argument
represents the new operator you wish to apply. Keep in mind, the tree will
enforce that the new operator is of another valid operator of that existing
node class.

### **Replicate subtrees or paths**

To replicate a subtree or path invoke one of the following methods that will
return a new Tree object:

> my_subtree = replicate_subtree(["denoise", "ts2db"])

> my_path = replicate_path(["denoise", "ts2db"])

Replicating a path will copy all nodes represented by the input list of operators
into a new tree containing only those nodes. A subtree will include all nodes
following the last node in the list (which will be the root of the new tree).

### **Add a subtree or path to an existing tree**

To append one tree to another, invoke the add_subtree method:

> my_tree.add_subtree(["denoise", "ts2db", "rf"], my_subtree)

This will append the root of the subtree as a child to the node of the main tree
specified by the first argument. It will also copy the rest of the contents of
the subtree as one would expect.

Keep in mind that this method will also enforce the ordering described by normal
node additions described above.

### **Execute a tree or pipeline**

To execute a tree, invoke the execute_tree method:

> my_tree.execute_tree("my_csv.csv")

This will check to make sure a prepNode is the root (as this was not
enforced while constructing the tree). It will output forecast csv files, but
will also output any plots or evaluations if the nodes for those steps are
present.

### **Save or load a Transformation Tree**

To save a Tree object as a pickle, invoke the following function:

> save_tree(my_tree, "my_tree_filename")

To load a Tree object from a pickle, invoke the following function:

> my_loaded_tree = load_tree("my_tree_filename")

Pickle files will by default be saved in the pickle_objects directory.

### Authors
-----------
* Logan Levitre (llevitre@uoregon.edu)
* Jarett Nishijo (jnishijo@uoregon.edu)
* Zeke Petersen (ezekielp@uoregon.edu)
* Jack Sanders  (jsander5@uoregon.edu)
* Callista West (cwest10@uoregon.edu)
