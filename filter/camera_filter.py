import numpy as np 

def collected_infos(obs):
    n_c = int(obs[0][0])
    n_t = int(obs[0][1])
    output = np.zeros((n_c, 9 + n_t))
    for i in range(n_c):
                output[i][:9] = obs[i][13 : 22]
                for t in range(n_t):
                    output[i][9+t] = obs[i][22+5*t+4]
                
    return output
