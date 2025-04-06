import numpy as np 

def split(obs):
    n_t = int(obs[1])
    n_o = int(obs[2])
    output = np.zeros((n_o, 4))
    for i in range(n_o):
        output[i] = obs[22 + 5*n_t + 4*i : 22 + 5*n_t + 4*i + 4]
    
    return output

def collected_infos(obs):
    n_c = int(obs[0][0])
    n_t = int(obs[0][1])
    n_o = int(obs[0][2])
    output = np.zeros((n_o, 4))
    for j in range(n_c):
        for i in range(n_o):
            if output[i][3] != 1: # Not filled yet
                output[i] = obs[j][22 + 5*n_t + 4*i : 22 + 5*n_t + 4*i + 4] 
                
    
    # Nếu hàng nào toàn 0 thì set lại bằng -1
    mask = np.all(output == 0, axis=1)
    output[mask] = -1

    output = np.delete(output, 3, axis=1)
    return output

