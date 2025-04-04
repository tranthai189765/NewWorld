import numpy as np

class TargetBuffer:
    def __init__(self, max_size, obs_shape):
        """
        max_size: số timestep tối đa lưu trong buffer.
        obs_shape: hình dạng của mỗi observation (số lượng targets, số feature mỗi target).
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.buffer = np.zeros((max_size, *obs_shape))
        # Thêm 5 mảng zero vào buffer trước
        for _ in range(4):
            self.store(np.zeros(obs_shape))

    def store(self, obs_targets):
        """
        Lưu observation mới vào buffer.
        obs_targets: numpy array có shape (số target, số feature mỗi target).
        """
        index = self.mem_cntr % self.mem_size
        self.buffer[index] = obs_targets
        self.mem_cntr += 1

    def take(self, num_steps):
        """
        Lấy num_steps timestep gần nhất và ghép lại.
        Output: numpy array có shape (num_steps * số target, số feature).
        """
        indices = np.arange(max(0, self.mem_cntr - num_steps), self.mem_cntr) % self.mem_size
        return np.concatenate(self.buffer[indices], axis=1)