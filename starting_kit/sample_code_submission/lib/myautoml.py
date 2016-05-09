import numpy as np
import scipy as sp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import grid_search
import operator
import copy
from sklearn.metrics import make_scorer

from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import *
from libscores import *
import h5py
from random import choice
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
        self.metric_map = { 'f1_metric': f1_metric, 'auc_metric': auc_metric, 'bac_metric': bac_metric, 'r2_metric': r2_metric, 'a_metric': a_metric, 'pac_metric': pac_metric }


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
        best_nn_af = 'tanh'
        reuse_weights = False


        hp_rf_all = [20,50,100,150,200,300,400,500,600,700,800,900,1000,1200,1500,2000,3000,4000,5000,10000,20000,40000,80000,160000]
        
        hp_lda_all = [1,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]
        hp_qda_all = [1e-6,.00001,.0001,.001,.01,.1,1,10,100,1000,10000,1e5,1e6, 5e-6,.00005,.0005,.005,.05,.5,5,50,500,5000,50000,5e5,5e6]

        while ((tmp_time-start) < time_budget):
            # initialize best classifiers during first cycle
            print "------------------------------------- "
            print "Cycle ", cycle_count
            print "------------------------------------- "
            if cycle_count == 0:
                
                hp_qda = choice(hp_qda_all)
                hp_qda_all.remove(hp_qda)
                
                hp_lda = choice(hp_lda_all)
                hp_lda_all.remove(hp_lda)

                hp_rf = hp_rf_all[0]
                hp_rf_all.remove(hp_rf)

                print "*******************"
                print "Training da..."
                print "hp_lda:", hp_lda
                print "hp_dqa:", hp_qda
                print "*******************"
                da, da_score = self.train_DA(X,y, hp_lda, hp_qda)

                print "*******************"
                print "Training nn..."
                print "nn_lr:", best_nn_lr
                print "nn_bs:", best_nn_bs
                print "nn_af:", best_nn_af
                print "reuse_weights:", reuse_weights
                print "*******************"
                nn, nn_score  = self.train_NN(X,y,best_nn_lr, best_nn_bs, best_nn_af, reuse_weights)

                print "*******************"
                print "Training rf..."
                print "hp_rf:", hp_rf
                print "*******************"
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
                hp_qda = choice(hp_qda_all)
                hp_qda_all.remove(hp_qda)
                
                hp_lda = choice(hp_lda_all)
                hp_lda_all.remove(hp_lda)

                hp_rf = hp_rf_all[0]
                hp_rf_all.remove(hp_rf)

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

                if ((len(hp_nn_all['learning_rate']) == 0) and (len(hp_nn_all['batch_size']) == 0) and (len(hp_nn_all['activation_function']) == 0)):
                    nn_af = best_nn_af
                    nn_bs = best_nn_bs
                    nn_lr = best_nn_lr

                print "*******************"
                print "Training da..."
                print "hp_lda:", hp_lda
                print "hp_dqa:", hp_qda
                print "*******************"
                da, da_score = self.train_DA(X,y, hp_lda,hp_qda)

                print "*******************"
                print "Training nn..."
                print "nn_lr:", nn_lr
                print "nn_bs:", nn_bs
                print "nn_af:", nn_af
                print "reuse_weights:", reuse_weights
                print "*******************"
                nn, nn_score  = self.train_NN(X,y,nn_lr, nn_bs, nn_af, reuse_weights)

                print "*******************"
                print "Training rf..."
                print "hp_rf:", hp_rf
                print "*******************"
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
            print "Best score NN" ,best_nn_score
            print "Best score RF" ,best_rf_score
            print "Best score DA" ,best_da_score
            # get cross-validation score for ensemble method only if the best weak learners have changed 
            # (otherwise we would just recompute the same thing)
            if (nn_changed or rf_changed or da_changed): 
                print "Ensembling..."
                # ensemble_score = self.ensemble_CV(best_da, best_nn, best_nn_bs, best_rf,X,y)
                ensemble_score=0

                # if the ensemble method has a better cv metric, use it. Otherwise, pick best classifier from the weak learners
                if ((ensemble_score >= best_nn_score) and (ensemble_score>= best_rf_score) and (ensemble_score >= best_da_score)):
                    print "Best score is by ensembling", ensemble_score
                    p = self.ensemble_predict(best_da,best_nn,best_rf,test)
                elif ((best_nn_score >= ensemble_score) and (best_nn_score >= best_rf_score) and (best_nn_score >= best_da_score)):
                    print "Best score is NN", best_nn_score
                    p = best_nn.predict(test).ravel()
                elif ((best_rf_score>= ensemble_score) and (best_rf_score >= best_nn_score) and (best_rf_score >= best_da_score)):
                    print "Best score is RF", best_rf_score
                    p = best_rf.predict(test)
                elif ((best_da_score>= ensemble_score) and (best_da_score >= best_nn_score) and (best_da_score >= best_rf_score)):
                    print "Best score is DA", best_da_score
                    p = best_da.predict(test)
                else:
                    print "Error during comparison of the cross validation metrics"

            print "nn_score", nn_score
            print "rf_score", rf_score
            print "da_score", da_score
            print "ensemble_score", ensemble_score

            cycle_count +=1
            tmp_time = time.time()

        return p

    def train_DA(self, X, y, lda_comp, qda_reg):
        '''
        Input: 
            qda_reg - reg_param
            lda_comp - n_components
            X - data matrix (train_num, feat_num)
            y - target labels matrix (train_num, label_num)

        Output: 
            best_clf - best classifier trained (QDA/LDA)
            best_score - CV score of best classifier

        Find best DA classifier.
        '''
        n_samples, n_feat = X.shape
        cv_folds = 10
        kf = KFold(n_samples, cv_folds, shuffle=False)

        
        
        lda = LinearDiscriminantAnalysis(n_components = lda_comp)
        qda = QuadraticDiscriminantAnalysis(reg_param = qda_reg)
        score_total_lda = 0 #running total of metric score over all cv runs
        score_total_qda = 0 #running total of metric score over all cv runs
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            lda.fit(X_train, y_train)
            cv_pred_lda = lda.predict(X_test)
            score_lda = eval(self.metric + '(y_test[:,None], cv_pred_lda[:,None], "' + self.task + '")')
            score_total_lda += score_lda
            
            qda.fit(X_train,y_train)
            cv_pred_qda = qda.predict(X_test)
            score_qda = eval(self.metric + '(y_test[:,None], cv_pred_qda[:,None], "' + self.task + '")')
            score_total_qda += score_qda

        score_lda = score_total_lda/cv_folds
        score_qda = score_total_qda/cv_folds
        
        # We keep the best one
        if(score_qda > score_lda):
            qda.fit(X,y)
            return qda, score_qda
        else:
            lda.fit(X,y)
            return lda, score_lda

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
            clf - best classifier trained with hp
            best_score - CV score of best classifier

        Tunes neural net with hp.    
        '''
        n_samples, n_feat = X.shape

        b_size = n_samples/bs
        hidden_units = min(n_feat/4, 400)
        cv_folds = 10
        kf = KFold(n_samples, cv_folds, shuffle=False)

        #create neural net
        clf  = Sequential()
        clf.add(Dense(hidden_units, input_dim=n_feat, activation=af))
        clf.add(Dense(self.target_num, activation = 'sigmoid'))
        sgd = SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
        
        clf.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
        
        score_total = 0 #running total of metric score over all cv runs
        for train_index, test_index in kf:
            cv_clf = copy.deepcopy(clf)
            X_train, X_test = X[train_index], X[test_index] 
            y_train, y_test = y[train_index], y[test_index]
            
            cv_clf.fit(X_train, y_train, nb_epoch=5, batch_size=b_size, verbose = 0)
            cv_pred = clf.predict(X_test, batch_size = n_samples)

            score = eval(self.metric + '(y_test[:,None], cv_pred, "' + self.task + '")')
            score_total = score_total+score
        
        best_score = score_total/cv_folds
        clf.fit(X, y, nb_epoch=5, batch_size=b_size, verbose=0)

        return clf, best_score

    def train_RF(self, X, y, n_est):
        '''
        Input: 
            n_est - number of estimators 
            example: {n_trees: [10, 100, 1000]}
            X - data matrix (train_num, feat_num)
            y - target labels matrix (train_num, label_num)

        Output: 
            rf - classifier 
            score - CV score of classifier

        Tunes random forest with hp.    
        '''
        n_samples, n_feat = X.shape
        rf = RandomForestClassifier(n_estimators = n_est)
        cv_folds = 10
        kf = KFold(n_samples, cv_folds, shuffle=False)

        score_total = 0 #running total of metric score over all cv runs
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            rf.fit(X_train, y_train)
            cv_pred = rf.predict(X_test)
            score = eval(self.metric + '(y_test[:,None], cv_pred[:,None], "' + self.task + '")')
            score_total += score
            
        score = score_total/cv_folds
        rf.fit(X,y)
        return rf, score

    def ensemble_CV(self, da, nn, nn_bs, rf, X, y):
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
        n_samples, n_feat = X.shape
        cv_folds = 10
        kf = KFold(n_samples, cv_folds, shuffle=False)

        score_total = 0

        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            da.fit(X_train, y_train)
            nn.fit(X_train, y_train, verbose=0, batch_size=nn_bs)
            rf.fit(X_train, y_train)
            
            cv_pred = self.ensemble_predict(da, nn, rf, X_test)
            score = eval(self.metric + '(y_test[:,None], cv_pred[:,None], "' + self.task + '")')
            score_total += score

        return score_total/cv_folds

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
        votes[:, 1] = nn.predict(test).ravel()
        votes[:, 2] = rf.predict(test)
        
        y_pred = np.mean(votes, axis = 1)
        return np.asarray(y_pred)
