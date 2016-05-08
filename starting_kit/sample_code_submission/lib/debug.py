from data_manager import DataManager
from myautoml import MyAutoML
import time
from libscores import *
import data_io

start = time.time()
verbose = False
debug_mode = 0
max_time = 180
max_cycle = 1
max_samples = 50000
 
phase = 1
 
input_dir = "../../phase"+str(phase)+"_input"
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
'''
D = DataManager('jasmine', input_dir, replace_missing=True, filter_features=True, max_samples=max_samples, verbose=verbose)
M = MyAutoML(D.info, verbose=False, debug_mode = debug_mode)

n_est = 1
rf, score_rf = M.train_RF(D.data['X_train'], D.data['Y_train'],n_est)

nn, score_nn = M.train_NN(D.data['X_train'], D.data['Y_train'], lr = 1e-1, bs = 400, af = 'tanh', reuse_weights = False)
lda_comp = 10
qda_reg = 1000
da, score_da = M.train_DA(D.data['X_train'], D.data['Y_train'],lda_comp, qda_reg)

score_ens = M.ensemble_CV(da,nn,rf,D.data['X_train'], D.data['Y_train'])

print "LDA",score_da
print "NN",score_nn
print "RF",score_rf 
print "Score ensemble is", score_ens''
'''
#print score

input_dir = "../../phase1_input"
datanames = data_io.inventory_data(input_dir)
for basename in datanames:
    D = DataManager(basename, input_dir, replace_missing=True, filter_features=True, max_samples=max_samples, verbose=verbose)
    solution = D.data['Y_test']
    M = MyAutoML(D.info, verbose=False, debug_mode = debug_mode)
    p = M.run_cycles(D.data['X_train'], D.data['Y_train'], D.data['X_test'], 600)
    np.where(p>.5,1,0).tofile('final_prediction.csv',sep='\n', format='%10.5f')
    test_score = eval(D.info['metric'] + '(solution, p[:,None], "' + D.info['task'] + '")')
    end = time.time()
    print "Duration", end-start
    print "Test score:" , basename , test_score