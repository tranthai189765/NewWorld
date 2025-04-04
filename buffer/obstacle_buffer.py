import numpy as np

class ObstacleBuffer:
    def __init__(self, max_size, obs_shape):
        """
        max_size: số timestep tối đa lưu trong buffer.
        obs_shape: hình dạng của mỗi observation (số lượng obstacles, số feature mỗi obstacle).
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.buffer = np.zeros((max_size, *obs_shape))  # Mảng buffer ban đầu toàn 0

    def store(self, obs_obstacles):
        """
        Lưu obstacles mới phát hiện vào buffer, giữ nguyên các obstacles cũ.
        obs_obstacles: numpy array có shape (số obstacles, số feature mỗi obstacle).
        """
        index = self.mem_cntr % self.mem_size  # Xác định vị trí trong buffer

        # Chỉ cập nhật những obstacles có giá trị khác 0
        mask = obs_obstacles != 0  
        self.buffer[index][mask] = obs_obstacles[mask]

        self.mem_cntr += 1  # Cập nhật bộ đếm timestep

    def take(self):
        """
        Lấy thông tin tổng hợp của tất cả obstacles đã phát hiện trong buffer.
        Output: numpy array có shape (số obstacles, số feature mỗi obstacle).
        """
        combined = np.max(self.buffer, axis=0)  # Lấy max trên từng feature của từng obstacle
        return combined