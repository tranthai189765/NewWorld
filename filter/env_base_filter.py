import numpy as np
def collected_infos(obs):
    output = np.zeros(12)  # Khởi tạo mảng có 12 phần tử
    
    for i in range(3):
        output[i] = obs[0][i]
    
    for i in range(3, 12):
        output[i] = obs[0][i+1]

    output[3:12] /= 1000  # Chia các phần tử từ 3 đến 10 cho 1000
    
    return output