import numpy as np 

def collected_infos(obs):
    n_c = int(obs[0][0])
    output = np.zeros((n_c, 9))
    for i in range(n_c):
                output[i] = obs[i][13 : 22]
                
    return output
