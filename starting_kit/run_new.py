#############################
# ChaLearn AutoML challenge #
#############################

# Usage: python run.py input_dir output_dir

# The input directory input_dir contains 5 subdirectories named by dataset,
# including:
# 	dataname/dataname_feat.type          -- the feature type "Numerical", "Binary", or "Categorical" (Note: if this file is abscent, get the feature type from the dataname.info file)
# 	dataname/dataname_public.info        -- parameters of the data and task, including metric and time_budget
# 	dataname/dataname_test.data          -- training, validation and test data (solutions/target values are given for training data only)
# 	dataname/dataname_train.data
# 	dataname/dataname_train.solution
# 	dataname/dataname_valid.data
#
# The output directory will receive the predicted values (no subdirectories):
# 	dataname_test_000.predict            -- Provide predictions at regular intervals to make sure you get some results even if the program crashes
# 	dataname_test_001.predict
# 	dataname_test_002.predict
# 	...
# 	dataname_valid_000.predict
# 	dataname_valid_001.predict
# 	dataname_valid_002.predict
# 	...

# Used for stdout print statements
verbose = True

# maximum number of training samples to use
max_samples = 50000

# maximum number of estimators to use for ensemble classifiers
max_estimators = float('Inf')

import time
overall_start = time.time()
import os
from sys import argv, path
import numpy as np
import gc

import data_io
from data_io import vprint
from data_manager import data_manager
from models import MyAutoML
import psutil

if 