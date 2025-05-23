import numpy as np

class ObstacleBuffer:
    def __init__(self, max_size, obs_shape):
        """
        max_size: số timestep tối đa lưu trong buffer.
        obs_shape: (số lượng obstacles, số feature mỗi obstacle).
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.buffer = np.full((max_size, *obs_shape), fill_value=-1.0)  # Khởi tạo toàn -1

    def store(self, obs_obstacles):
        """
        Lưu các obstacles nhìn thấy (tức là khác [-1, -1, -1]) vào buffer.
        """
        index = self.mem_cntr % self.mem_size  # Xác định vị trí lưu mới

        # Copy buffer hiện tại (vì mình không muốn làm mất thông tin obstacles cũ)
        self.buffer[index] = self.buffer[(index - 1) % self.mem_size]

        # Tìm những hàng hợp lệ (tức là không phải [-1, -1, -1])
        valid_mask = ~np.all(obs_obstacles == -1, axis=1)

        # Cập nhật các hàng hợp lệ vào buffer tại index
        self.buffer[index][valid_mask] = obs_obstacles[valid_mask]

        self.mem_cntr += 1  # Cập nhật bộ đếm timestep

    def take(self):
        """
        Trả về thông tin tổng hợp về obstacles (ưu tiên thông tin mới nhất khác -1).
        Output shape: (số lượng obstacles, số feature mỗi obstacle)
        """
        # Bắt đầu từ thông tin cũ nhất đến mới nhất, chọn thông tin gần nhất khác -1
        combined = np.full_like(self.buffer[0], fill_value=-1.0)
        for i in range(self.mem_size):
            idx = (self.mem_cntr - 1 - i) % self.mem_size
            mask = ~np.all(self.buffer[idx] == -1, axis=1)
            combined[mask] = self.buffer[idx][mask]
        return combined
