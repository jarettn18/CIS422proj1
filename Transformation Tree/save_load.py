"""
File: save_load.py
Class: CIS 422
Date: February 4, 2021
Team: The Nerd Herd
Head Programmers: Callista West
Version 0.1.0
Saving and Loading a Tree
"""

import pickle
from pathlib import Path

def save_tree(tree, file_name):
    """
    :tree: a tree object wanting to be saved
    :file_name: name of file wanting tree object to be saved to (str)

    Function uses pickle to open file_name with write access to write object to a file
    Function saves the file into directory pickle_objects
    Function returns None
    """
    root = Path(".")
    my_path = root / "pickle_objects" / file_name
    pickle_out = open(my_path, "wb")
    pickle.dump(tree, pickle_out)
    pickle_out.close()
    return None

def load_tree(tree, file_name):
    """
    : tree: same tree object that was passed into save_tree()
    : file_name: same file name (file_name) that the tree object was saved to in save_tree()

    Function opens the file_name with read access and uses pickle.load() to load the file
        to a var.
    The file is saved within the directory "pickle_objects", which is stored within the
        Transformation Tree directory
    Function returns the tree object that had been saved previously by the save_tree() function
    """
    root = Path(".")
    my_path = root / "pickle_objects" / file_name
    pickle_in = open(my_path, "rb")
    tree = pickle.load(pickle_in)
    return tree
