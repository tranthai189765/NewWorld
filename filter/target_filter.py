import numpy as np 

def split(obs):
    n_t = int(obs[1])
    output = np.zeros((n_t, 5))
    for i in range(n_t):
        output[i] = obs[22 + 5*i : 22 + 5*i + 5]
    
    return output

def collected_infos(obs):
    n_c = int(obs[0][0])  # Số camera
    n_t = int(obs[0][1])  # Số target

    output = np.zeros((n_t, 4 + n_c))  # 4 thông tin target + n_c cờ quan sát

    for j in range(n_c):
        for i in range(n_t):
            # Check xem camera j có thấy target i không
            visible = obs[j][22 + 5*i + 4] == 1
            if visible:
                # Gán cờ camera j thấy target i
                output[i][4 + j] = 1
                # Nếu thông tin 4 chiều đầu của target vẫn toàn 0, thì gán dữ liệu từ camera j
                if np.all(output[i][:4] == 0):
                    output[i][:4] = obs[j][22 + 5*i : 22 + 5*i + 4]  # lấy 4 chiều đầu thôi

    # Nếu target nào chưa được thấy bởi camera nào (4 giá trị đầu vẫn = 0) thì gán thành -1
    mask = np.all(output[:, :4] == 0, axis=1)
    output[mask, :4] = -1

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
