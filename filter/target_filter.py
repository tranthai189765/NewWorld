import numpy as np 

def split(obs):
    n_t = int(obs[1])
    output = np.zeros((n_t, 5))
    for i in range(n_t):
        output[i] = obs[22 + 5*i : 22 + 5*i + 5]
    
    return output

def collected_infos(obs):
    n_c = int(obs[0][0])
    n_t = int(obs[0][1])
    output = np.zeros((n_t, 5))
    for j in range(n_c):
        for i in range(n_t):
            if output[i][3] != 1: # Not filled yet
                output[i] = obs[j][22 + 5*i : 22 + 5*i + 5]
    
    # Nếu hàng nào toàn 0 thì set lại bằng -1
    mask = np.all(output == 0, axis=1)
    output[mask] = -1

    output = np.delete(output, 4, axis=1)
    return output

def has_unseen_targets(obs, k):
    """
    Kiểm tra xem có ÍT NHẤT k targets chưa bị phát hiện không.
    Nếu có ít nhất k hàng trong obs chứa toàn -1 thì trả về True, ngược lại trả về False.
    """
    unseen_count = np.sum(np.all(obs == -1, axis=1))
    return unseen_count >= k

def normalize(x):
    for i in range(8):
        for j in range(20):
            if j % 4 != 3:
                x[i][j] /= 1000
    
    return x