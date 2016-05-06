import numpy as np
import scipy as sp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import operator
import copy

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

    def train_DA(self, X, y, hp_lda, hp_qda):
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

        # WARNING: put the callable corresponding to the metric method w-when you call Grid Search!!

        # We try the LDA
        lda = LinearDiscriminantAnalysis()
        grid_result_lda = grid_search.GridSearchCV(lda, hp_lda)
        grid_result_lda.fit(X, y)
        best_clf_lda = grid_result_lda.best_estimator_
        best_score_lda = grid_result_lda.best_score_
        
        # We do the same for the QDA
        qda = QuadraticDiscriminantAnalysis()
        grid_result_qda = grid_search.GridSearchCV(qda, hp_qda)
        grid_result_qda.fit(X, y)
        best_clf_qda = grid_result_qda.best_estimator_
        best_score_qda = grid_result_qda.best_score_
        
        # We keep the best one
        if(best_score_qda > best_score_lda):
            return best_clf_qda, best_score_qda
        else:
            return best_clf_lda, best_score_lda

    def train_NN(self, X, y, hp):
        '''
        Input: 
            hp - dictionary with hyperparameter name as keys and values as values (refer to format in sklearn.grid_search.GridSearchCV)

            X - data matrix (train_num, feat_num)
            y - target labels matrix (train_num, label_num)

        Output: 
            best_clf - best classifier trained with hp
            best_score - CV score of best classifier

        Tunes neural net with hp.    
        '''
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

        # WARNING: put the callable corresponding to the metric method w-when you call Grid Search!!

        rf = RandomForestClassifier()
        grid_result = grid_search.GridSearchCV(rf, hp)
        grid_result.fit(X, y)
        best_clf = grid_result.best_estimator_
        best_score = grid_result.best_score_
        
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

        # We predict the new values for the test matrix, using the different input classifiers
        # WARNING: is test shape adapted? What about multiclass classification problems?
        votes = np.zeros((test.shape[0], 3))
        votes[:, 0] = da.predict(test)
        votes[:, 1] = nn.predict(test)
        votes[:, 2] = rf.predict(test)
        
        # We make them vote, and build the final y_pred using those votes
        y_pred = []
        for index in range(votes.shape[0]):
            sample = votes[index]
            sample = sample.astype(int)
            counts = np.bincount(sample)
            argmax = np.argmax(counts)
            y_pred.append(sample[argmax])

        return y_pred
