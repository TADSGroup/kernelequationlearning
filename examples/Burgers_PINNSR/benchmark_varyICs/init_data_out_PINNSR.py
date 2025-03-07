import numpy as np

data_out_PINNSR = []
NUM_RUNS = 10
N_OBS = [10,30,50,100,200,300,400,500,600]
for n_obs in N_OBS:
    for run in range(NUM_RUNS):
        element = []
        element.append(n_obs)
        element.append(run)
        element.append('a') # enforce a dtype=object array to store coeffs
        element.append('b') # enforce a dtype=object array to store u_pred
        data_out_PINNSR.append(element)
        
np.save(f'data_out_PINNSR.npy', np.array(data_out_PINNSR, dtype=object), allow_pickle=True)

