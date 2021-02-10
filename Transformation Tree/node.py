"""
File: node.py
Class: CIS 422
Date: January 30, 2021
Team: The Nerd Herd
Head Programmers: Callista West, Zeke Petersen, Jack Sanders
Version 0.1.1
Node Class and node implementation
"""

# ------- Begin Node Classes --------------

import sys
sys.path.append('../Modules')
"""
for p in sys.path:
    print(p)
"""
import preprocessing as prep
import visualization as viz
import model as mod

# Represents a node of an n-ary tree using a node Class
class Node:
    # creating a new tree node
    def __init__(self, op, parent):
        self.op = op
        self.parent = parent
        self.children = []         # Filled in during execution
        self.executed = 0

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

    def __init__(self, op):
        super().__init__(op, None)
        self.ts = None
        self.inputs = None
        self.ts_test = None
        self.inputs_test = None

    def execute():
        """
        This function will help execute the tree
        """
        if self.op == "mlp":
            model = mod.mlp_model()
            model.fit(self.inputs, self.ts)
            forecast = model.forecast(inputs_test)
            return forecast
        elif self.op == "rf":
            model = mod.rf_model()
            model.fit(self.inputs, self.ts)
            forecast = model.forecast(inputs_test)
            return forecast

class splitNode(Node):

    def __init__(self, op):
        super().__init__(op, None)
        self.file_name = None  # Filled in during execution
        self.file_name_test = None

    def execute():
        """
        This function will help execute the tree
        """
        if self.op == "ts2db":
            return prep.ts2db(self.file_name, self.file_name_test)


class visualizeNode(Node):

    def __init__(self, op):
        super().__init__(op, None)
        self.ts = None

    def execute():
        # pass
        """
        This function will help execute the tree
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
