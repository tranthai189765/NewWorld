import numpy as np
def _normalize_state(state):
    #print("raw_state = ", state)
    n_c = int(state[0][0])
    n_t = int(state[0][1])
    n_o = int(state[0][2])
    for camera_index in range(n_c):
        val = int(state[camera_index][3])
        mapping = {
            0: [0, 0, 0, 1],
            1: [0, 0, 1, 0],
            2: [0, 1, 0, 0],
            3: [1, 0, 0, 0]
        }
        original_radius = state[camera_index][19]
        encoded_bits = mapping.get(val, [0,0,0,0])  # fallback nếu giá trị ngoài [0,3]
        state[camera_index][0:4] = np.array(encoded_bits, dtype=float)
        state[camera_index][4:16] = state[camera_index][4:16]/1000.0
        state[camera_index][16:18] = state[camera_index][16:18]/state[camera_index][19]
        state[camera_index][18] = state[camera_index][18]/180.0
        state[camera_index][19] = 1.0
        state[camera_index][20:22] = state[camera_index][20:22]/180.0
        for target_index in range(n_t):
            state[camera_index][22 + 5 * target_index : 22 + 5 * target_index + 3] = state[camera_index][22 + 5 * target_index : 22 + 5 * target_index + 3]/1000.0
        for obstacle_index in range(n_o):
            state[camera_index][22+5*n_t+4*obstacle_index : 22+5*n_t+4*obstacle_index+3]  = state[camera_index][22+5*n_t+4*obstacle_index : 22+5*n_t+4*obstacle_index+3]/1000.0
        for tm_index in range(n_c):
            state[camera_index][22+5*n_t+4*n_o+7*tm_index: 22+5*n_t+4*n_o+7*tm_index+3] = state[camera_index][22+5*n_t+4*n_o+7*tm_index: 22+5*n_t+4*n_o+7*tm_index+3]/1000.0
            state[camera_index][22+5*n_t+4*n_o+7*tm_index+3: 22+5*n_t+4*n_o+7*tm_index+5] = state[camera_index][22+5*n_t+4*n_o+7*tm_index+3: 22+5*n_t+4*n_o+7*tm_index+5]/original_radius
            state[camera_index][22+5*n_t+4*n_o+7*tm_index+5] = state[camera_index][22+5*n_t+4*n_o+7*tm_index+5]/180.0

    return state

def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])

def concat_state_latent_total(s, z, n):
    N = s.shape[0]
    out = []
    for i in range(N):
        out.append(concat_state_latent(s[i], z[i], n))
    return np.array(out)

