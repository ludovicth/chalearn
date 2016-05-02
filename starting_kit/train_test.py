'''
Separate a dataset into training set and test set

Usage: python train_test.py train.data train.solution
'''

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import sys

Xfile = sys.argv[1]
yfile = sys.argv[2]

# import data
X =  np.genfromtxt(Xfile, delimiter=" ")
y = np.genfromtxt(yfile)

# shuffle and split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 1)

# export the data
np.savetxt("X_train.data", X_train, delimiter=" ")
np.savetxt("y_train.data", y_train, delimiter=" ")
np.savetxt("y_test.data", y_test, delimiter=" ")
np.savetxt("X_test.data", X_test, delimiter=" ")