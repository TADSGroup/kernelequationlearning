import numpy as np

data_out_PINNSR = []
data_out_PINNSR.append('a') # enforce a dtype=object array to store coeffs
data_out_PINNSR.append('b') # enforce a dtype=object array to store u_pred
        
np.save(f'data_out_PINNSR.npy', np.array(data_out_PINNSR, dtype=object), allow_pickle=True)

