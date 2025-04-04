import numpy as np

def collected_infos(obs):
    obs = np.array(obs)
    n_t = obs.shape[0]
    output = np.zeros((n_t, 7))
    for i in range(n_t):
        output[i][0] = obs[i][0] / 1000
        output[i][1] = obs[i][1] / 1000
        output[i][2] = obs[i][3]
        output[i][3:7] = (obs[i][6:10] > 0).astype(int)  # Gán 1 nếu >0, ngược lại =0


    return output