import torch

# Load file .pt (giả sử nó là dictionary chứa key "labels")
dataset = torch.load("mate/dataset/10k_dataset_modified.pt")

# Chuyển về tensor float32 và chọn các cột 4, 5, 6, 7
labels = torch.tensor(dataset["labels"], dtype=torch.float32)[..., 3:7+1]
# Chia các giá trị ở chiều cuối cùng cho 1000
# Lấy tensor obstacles
obstacles = dataset["inputs"]["obstacles"]  # shape: (batchsize, 9, 3)

# Chia cột thứ 3 (index = 2) cho 1000
obstacles[:, :, 2] = obstacles[:, :, 2] / 1000.0

# Cập nhật lại vào dataset
dataset["inputs"]["obstacles"] = obstacles

# Cập nhật lại trong dataset nếu muốn
dataset["labels"] = labels

# Lưu lại file .pt nếu cần
torch.save(dataset, "mate/dataset/10k_dataset_modified_labels.pt")