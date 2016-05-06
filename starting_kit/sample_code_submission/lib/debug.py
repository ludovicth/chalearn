from data_manager import DataManager
from myautoml import MyAutoML
import time

start = time.time()
verbose = False
debug_mode = 0
max_time = 180
max_cycle = 1
max_samples = 50000
 
phase = 1
 
input_dir = "..\..\phase"+str(phase)+"_input\\"
output_dir = "phase"+str(phase)+"_res"
 

'''
D = DataManager('jasmine', input_dir, replace_missing=True, filter_features=True, max_samples=max_samples, verbose=verbose)
M = MyAutoML(D.info, verbose=False, debug_mode = debug_mode)

n_est = 20
clf, score = M.train_RF(D.data['X_train'], D.data['Y_train'],n_est)
'''

'''
D = DataManager('jasmine', input_dir, replace_missing=True, filter_features=True, max_samples=max_samples, verbose=verbose)
M = MyAutoML(D.info, verbose=False, debug_mode = debug_mode)
clf, score = M.train_NN(D.data['X_train'], D.data['Y_train'], lr = 1e-1, bs = 400, af = 'tanh', reuse_weights = False)
'''

'''
D = DataManager('jasmine', input_dir, replace_missing=True, filter_features=True, max_samples=max_samples, verbose=verbose)
M = MyAutoML(D.info, verbose=False, debug_mode = debug_mode)

lda_comp = 10
qda_reg = 1000
clf, score = M.train_DA(D.data['X_train'], D.data['Y_train'],lda_comp, qda_reg)
'''


D = DataManager('jasmine', input_dir, replace_missing=True, filter_features=True, max_samples=max_samples, verbose=verbose)
M = MyAutoML(D.info, verbose=False, debug_mode = debug_mode)

n_est = 1
rf, score_rf = M.train_RF(D.data['X_train'], D.data['Y_train'],n_est)

nn, score_nn, pca = M.train_NN(D.data['X_train'], D.data['Y_train'], lr = 1e-1, bs = 400, af = 'tanh', reuse_weights = False)
lda_comp = 10
qda_reg = 1000
da, score_da = M.train_DA(D.data['X_train'], D.data['Y_train'],lda_comp, qda_reg)

score_ens = M.ensemble_CV(da,nn,pca,rf,D.data['X_train'], D.data['Y_train'])

print "LDA",score_da
print "NN",score_nn
print "RF",score_rf 
print "Score ensemble is", score_ens
end = time.time()
print "Duration", end-start