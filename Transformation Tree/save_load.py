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

"""
class Save used to save a tree as well as load a tree
"""

class Save:
    def __init__(self):
        pass

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

        Function opwns the file_name with read access and uses pickle.load() to load the file
          to a var.
        Function returns None
        """
        pickle_in = open(file_name, "rb")
        tree = pickle.load(pickle_in)
        return None
