import torch

# Load file .pt (giả sử nó là dictionary chứa key "labels")
dataset = torch.load("dataset/10k_dataset_modified.pt")

# Lấy tensor labels và chuyển sang float32
labels = torch.tensor(dataset["labels"], dtype=torch.float32)

# Lấy các cột từ 3 đến 7 (tức index 3, 4, 5, 6, 7) -> shape: (batchsize, 8, 5)
labels = labels[..., 2:8]

# Đảo giá trị tại vị trí index 0 của chiều cuối (tức cột 3 ban đầu)
# Nếu là 0 -> 1, 1 -> 0
labels[..., 0] = 1 - labels[..., 0]

# Cập nhật lại labels trong dataset
dataset["labels"] = labels

# Chia giá trị ở cột thứ 3 (index = 2) của obstacles cho 1000
obstacles = dataset["inputs"]["obstacles"]  # shape: (batchsize, 9, 3)
obstacles[:, :, 2] = obstacles[:, :, 2] / 1000.0

# Cập nhật lại obstacles
dataset["inputs"]["obstacles"] = obstacles

# Lưu lại file mới
torch.save(dataset, "dataset/10k_dataset_modified_labels3.pt")
