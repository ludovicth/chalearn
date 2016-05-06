import numpy as np
import scipy as sp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from random import choice
import operator
import copy
import time

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


    def run_cycles(self, X, y, test, time_budget):
        '''
        Input: 
            X - data matrix (train_num, feat_num)
            y - target labels matrix (train_num, label_num)
            test - test matrix (test_num, feat_num)
            time_budget - numerical, time budget alloted to the dataset
        Output: 
            p - predictions (test_num, target_num)

        While there is still time budget left, tune all 3 classifiers in each cycle predicts on best classifier (including ensemble). For each cycle, increase size of hyperparameters to tune.
        '''
        start = time.time()
        cycle_count = 0
        tmp_time = time.time()

        hp_nn_all = {}
        hp_nn_all['learning_rate'] = [1e-2,1e-3,1e-4]
        hp_nn_all['batch_size'] = [200,300,400,500]
        hp_nn_all['activation_function'] = ['sigmoid']
        best_nn_lr = 0.1
        best_nn_bs = 100
        best_nn_af = 'sigmoid'
        reuse_weights = False

        hp_rf_all = {}
        hp_rf_all['n_estimators'] = [20,50,100,150,200,300,400,500,600,700,800,900,1000,1200,1500,2000,3000,4000,5000]
        
        hp_lda_all = {}
        hp_lda_all['n_components'] = [1,10,50,100,200,300,400,500,600,700,800,1000]
        hp_qda_all = {}
        hp_qda_all['reg_param'] =[.00001,.0001,.001,.01,.1,1,10,100,1000,10000]

        while ((tmp_time-start) < time_budget):
            # initialize best classifiers during first cycle
            if cycle_count == 0:
                hp_qda = {}
                tmp = choice(hp_qda_all['reg_param'])
                hp_qda['reg_param'] = [tmp]
                hp_qda_all['reg_param'].remove(tmp)
                
                hp_lda = {}
                tmp = choice(hp_lda_all['n_components'])
                hp_lda['n_components'] = [tmp]
                hp_lda_all['n_components'].remove(tmp)

                hp_rf = {}
                tmp = hp_rf_all['n_estimators'][0]
                hp_rf['n_estimators'] = [tmp]
                hp_rf_all['n_estimators'].remove(tmp)

                da, da_score = self.train_DA(X,y, hp_lda, hp_qda)
                nn, nn_score  = self.train_NN(X,y,best_nn_lr, best_nn_bs, best_nn_af, reuse_weights)
                rf, rf_score = self.train_RF(X,y,hp_rf)
                # initialize the best classifiers
                best_da, best_da_score = da, da_score
                best_nn, best_nn_score= nn, nn_score
                best_rf, best_rf_score = rf, rf_score
                nn_changed = True
                rf_changed = True
                da_changed = True

            if cycle_count > 0:
                #set hyperparameters to test
                hp_qda = {}
                tmp = choice(hp_qda_all['reg_param'])
                hp_qda['reg_param'] = [tmp]
                hp_qda_all['reg_param'].remove(tmp)
                
                hp_lda = {}
                tmp = choice(hp_lda_all['n_components'])
                hp_lda['n_components'] = [tmp]
                hp_lda_all['n_components'].remove(tmp)
                
                hp_rf = {}
                tmp = hp_rf_all['n_estimators'][0]
                hp_rf['n_estimators'] = [tmp]
                hp_rf_all['n_estimators'].remove(tmp)

                if len(hp_nn_all['learning_rate']) > 0:
                    nn_lr = hp_nn_all['learning_rate'][0]
                    hp_nn_all['learning_rate'].remove(nn_lr)
                else:
                    nn_lr = best_nn_lr

                if ((len(hp_nn_all['learning_rate']) == 0) and (len(hp_nn_all['batch_size']) > 0)):
                    nn_bs = hp_nn_all['batch_size'][0]
                    hp_nn_all['batch_size'].remove(nn_bs)
                else:
                    nn_bs = best_nn_bs

                if ((len(hp_nn_all['learning_rate']) == 0) and (len(hp_nn_all['batch_size']) > 0) and (len(hp_nn_all['activation_function']) > 0)):
                    nn_af = hp_nn_all['activation_function'][0]
                    hp_nn_all['activation_function'].remove(nn_af)
                else:
                    nn_af = best_nn_af

                if ((len(hp_nn_all['learning_rate']) == 0) and (len(hp_nn_all['batch_size']) > 0) and (len(hp_nn_all['activation_function']) == 0)):
                    nn_af = best_nn_af
                    nn_bs = best_nn_bs
                    nn_lr = best_nn_lr
                    reuse_weights = True

                da, da_score = self.train_DA(X,y, hp_lda,hp_qda)
                nn, nn_score  = self.train_NN(X,y,nn_lr, nn_bs, nn_af, reuse_weights)
                rf, rf_score = self.train_RF(X,y,hp_rf)
                # update the best classifiers
                if nn_score > best_nn_score:
                    best_nn, best_nn_score = nn, nn_score
                    nn_changed = True
                    best_nn_af = nn_af
                    best_nn_bs = nn_bs
                    best_nn_lr = nn_lr
                else:
                    nn_changed = False
                if rf_score > best_rf_score:
                    best_rf, best_rf_score = rf, rf_score 
                    rf_changed = True
                else:
                    rf_changed = False
                if da_score > best_da_score:
                    best_da, best_da_score = da, da_score
                    da_changed = True
                else:
                    da_changed = False

            # get cross-validation score for ensemble method only if the best weak learners have changed 
            # (otherwise we would just recompute the same thing)
            if (nn_changed or rf_changed or da_changed): 
                ensemble_score = self.ensemble_CV(best_da, best_nn, best_rf,X,y)

                # if the ensemble method has a better cv metric, use it. Otherwise, pick best classifier from the weak learners
                if ((ensemble_score >= best_nn_score) and (ensemble_score>= best_rf_score) and (ensemble_score >= best_da_score)):
                    p = self.ensemble_predict(da,nn,rf,test)
                elif ((best_nn_score >= ensemble_score) and (best_nn_score >= best_rf_score) and (best_nn_score >= best_da_score)):
                    p = best_nn.predict(test)
                elif ((best_rf_score>= ensemble_score) and (best_rf_score >= best_nn_score) and (best_rf_score >= best_da_score)):
                    p = best_rf.predict(test)
                elif ((best_da_score>= ensemble_score) and (best_da_score >= best_nn_score) and (best_da_score >= best_rf_score)):
                    p = best_da.predict(test)
                else:
                    print "Error during comparison of the cross validation metrics"

            cycle_count +=1
            tmp_time = time.time()

        return p

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
