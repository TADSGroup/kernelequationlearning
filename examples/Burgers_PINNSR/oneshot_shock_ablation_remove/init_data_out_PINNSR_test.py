import numpy as np

data_out_PINNSR_test = []
data_out_PINNSR_test.append('a') # enforce a dtype=object array to store coeffs
data_out_PINNSR_test.append('b') # enforce a dtype=object array to store u_pred
        
np.save(f'data_out_PINNSR_test.npy', np.array(data_out_PINNSR_test, dtype=object), allow_pickle=True)

