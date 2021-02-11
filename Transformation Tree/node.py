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

    def execute(training, op_exec):
        """
        This function will help execute the tree
        """
        data_op = prep.read_from_file(training)
        print(data_op)
        ts = prep.impute_missing_data(data_op)

        return_value = None
        if op_exec == "denoise":
            return_value = prep.impute_missing_data(ts)
        elif op_exec == "impute_missing_data":
            return_value = prep.impute_missing_data(ts)
        elif op_exec == "impute_outliers":
            return_value = prep.impute_outliers(ts)
        elif op_exec == "longest_continuous_run":
            return_value = prep.longest_continuous_run(ts)
        elif op_exec == "clip":
            return_value = prep.denoise(ts, self.starting_date, self.starting_date, self.final_date)
        elif op_exec == "assign_time":
            return_value = prep.assign_time(ts, self.starting_date, self.increment)
        elif op_exec == "difference":
            return_value = prep.difference(ts)
        elif op_exec == "scaling":
            return_value = prep.scaling(ts)
        elif op_exec == "standardize":
            return_value = prep.standardize(ts)
        elif op_exec == "logarithm":
            return_value = prep.logarithm(ts)
        elif op_exec == "cubic_roots":
            return_value = prep.cubic_roots(ts)
        return return_value

class modelNode(Node):

    def __init__(self, op):
        super().__init__(op, None)
        self.ts = None
        self.inputs = None
        self.ts_test = None
        self.inputs_test = None

    def execute(train_ts, train_inputs, test_inputs, op_exec):
        """
        This function will help execute the tree
        """
        if op_exec == "mlp":
            model = mod.mlp_model()
            model.fit(train_inputs, train_ts)
            forecast = model.forecast(test_inputs)
            return forecast
        elif op_exec == "rf":
            model = mod.rf_model()
            model.fit(train_inputs, train_ts)
            forecast = model.forecast(test_inputs)
            return forecast

class splitNode(Node):

    def __init__(self, op):
        super().__init__(op, None)
        self.file_name = None  # Filled in during execution
        self.file_name_test = None

    def execute(input_file, op_exec):
        """
        This function will help execute the tree
        """
        if op_exec == "ts2db":
            # print(prep.ts2db(input_file, None))
            return prep.ts2db(input_file)

class visualizeNode(Node):

    def __init__(self, op):
        super().__init__(op, None)
        self.ts = None

    def execute(input_file, op_exec):
        # pass
        """
        This function will help execute the tree
        """
        # input_file = viz.read_matrix(input_file)
        input_file = viz.csv_to_ts(input_file)
        input_file = prep.impute_missing_data(input_file)
        input_file = prep.impute_outliers(input_file)

        return_value = None
        if op_exec == "plot":
            return_value = viz.plot(input_file)
        elif op_exec == "histogram":
            return_value = viz.histogram(input_file)
        elif op_exec == "summary":
            return_value = viz.summary(input_file)
        elif op_exec == "box_plot":
            return_value = viz.box_plot(input_file)
        elif op_exec == "shapiro_wilk":
            return_value = viz.shapiro_wilk(input_file)
        elif op_exec == "d_agostino":
            return_value = viz.d_agostino(input_file)
        elif op_exec == "anderson_darling":
            return_value = viz.anderson_darling(input_file)
        elif op_exec == "qq_plot":
            return_value = viz.qq_plot(input_file)
        return return_value

class evalNode(Node):

    def __init__(self, op):
        super().__init__(op, None)
        self.y_test = None
        self.y_forecast = None

    def execute(actual, forecast, op_exec):
        # pass
        """
        This function will help execute the tree
        """

        return_value = None
        if op_exec == "MSE":
            return_value = viz.MSE(actual, forecast)
        elif op_exec == "RMSE":
            return_value = viz.RMSE(actual, forecast)
        elif op_exec == "MAPE":
            return_value = viz.MAPE(actual, forecast)
        elif op_exec == "sMAPE":
            return_value = viz.sMAPE(actual, forecast)
        return return_value

# ------- End Node Classes --------------
