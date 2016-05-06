import numpy as np
import scipy as sp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import *
import operator
import copy
from libscores import *
import h5py

class MyAutoML:
    '''
    Constructor initializes instance variables:
        self.label_num
        self.target_num
        self.task
        self.metric
    '''
    def __init__(self, info, verbose=True, debug_mode=False, run_on_gpu=False):
        self.label_num = info['label_num']
        self.target_num = info['target_num']
        self.task = info['task']
        self.metric = info['metric']


    def run_cycles(self, X, y, time_budget):
        '''
        Input: 
            X - data matrix (train_num, feat_num)
            y - target labels matrix (train_num, label_num)
        Output: 
            p - predictions (test_num, target_num)

        While there is still time budget left, tune all 3 classifiers in each cycle predicts on best classifier (including ensemble). For each cycle, increase size of hyperparameters to tune.
        '''

    def train_DA(self, X, y, hp):
        '''
        Input: 
            None
            X - data matrix (train_num, feat_num)
            y - target labels matrix (train_num, label_num)

        Output: 
            best_clf - best classifier trained (QDA/LDA)
            best_score - CV score of best classifier

        Find best DA classifier.
        '''
        return best_clf, best_score

    def train_NN(self, X, y, lr, bs, af, reuse_weights = False):
        '''
        Input: 
            X - data matrix (train_num, feat_num)
            y - target labels matrix (train_num, label_num)
            lr = SGD learning rate
            bs = batch size denominator
            af = activation function
            reuse_weights - use previous trained weights

        Output: 
            best_clf - best classifier trained with hp
            best_score - CV score of best classifier

        Tunes neural net with hp.    
        '''
        n_samples, n_feat = X.shape
        n_pca = max(n_feat/5, 100)
        pca = PCA(n_components=int(n_pca))
        X = pca.fit_transform(X)
        n_samples, n_feat = X.shape

        b_size = int(n_samples/bs)
        hidden_units = 400  #will be changed to use values from hyperparameter list
        cv_folds = 10
        kf = KFold(n_samples, cv_folds, shuffle=False)

        #create neural net
        clf  = Sequential()
        clf.add(Dense(hidden_units, input_dim=n_feat, activation=af))
        clf.add(Dense(self.target_num, activation = 'sigmoid'))
        sgd = SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)

        if (reuse_weights == True):
            clf.load_weights('nn_weights.h5')
        
        clf.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
        
        score_total = 0 #running total of metric score over all cv runs
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            clf.fit(X_train, y_train, nb_epoch=5, batch_size=b_size, verbose = 0)
            cv_pred = clf.predict(X_test, batch_size = b_size)
            score = eval(self.metric + '(y_test[:,None], cv_pred, "' + self.task + '")')
            score_total += score

        clf.save_weights('nn_weights.h5', overwrite=True)
        for i, pred in enumerate(cv_pred):
            print pred, y_test[i]

        best_score = score_total/cv_folds
        best_clf = clf

        return best_clf, best_score

    def train_RF(self, X, y, hp):
        '''
        Input: 
            hp - dictionary with hyperparameter name as keys and values as values (refer to format in sklearn.grid_search.GridSearchCV)
            example: {n_trees: [10, 100, 1000]}
            X - data matrix (train_num, feat_num)
            y - target labels matrix (train_num, label_num)

        Output: 
            best_clf - best classifier trained with hp
            best_score - CV score of best classifier

        Tunes random forest with hp.    
        '''
        return best_clf, best_score

    def ensemble_CV(self, da, nn, rf, X, y):
        '''
        Input:
            da - best da classifier
            nn - best nn classifier
            rf - best rf classifier
            X - data matrix (train_num, feat_num)
            y - target labels matrix (train_num, label_num)
        Output:
            score - CV score of ensemble classifier

        Used to find out if ensemble method does better at CV then individual classifier.
        '''
        return score

    def ensemble_predict(self, da, nn, rf, test):
        '''
        Input:
            da - best da classifier
            nn - best nn classifier
            rf - best rf classifier
            test - test matrix (test_num, feat_num)

        Output:
            pred - predictions (test_num, target_num)
        '''
        return pred
