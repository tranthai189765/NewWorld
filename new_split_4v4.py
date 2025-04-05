import torch

# Load file .pt (giả sử nó là dictionary chứa key "labels")
dataset = torch.load("dataset_4v4/75k_dataset_4v4.pt")

# Lấy tensor labels và chuyển sang float32
labels = torch.tensor(dataset["labels"], dtype=torch.float32)

# Lấy các cột từ 3 đến 7 (tức index 3, 4, 5, 6, 7) -> shape: (batchsize, 8, 5)
labels = labels[..., 2:8]

# Đảo giá trị tại vị trí index 0 của chiều cuối (tức cột 3 ban đầu)
# Nếu là 0 -> 1, 1 -> 0
labels[..., 0] = 1 - labels[..., 0]

# Cập nhật lại labels trong dataset
dataset["labels"] = labels

# ===================== Xử lý cameras ======================
# cameras shape: (batchsize, 4, 45)
cameras = dataset["inputs"]["cameras"]

# Lặp qua index cuối (0 -> 44), giữ lại các index không phải 5, 7, 8 trong mỗi nhóm 9
mask = torch.ones_like(cameras)
for i in range(45):
    if i % 9 in [5, 7, 8]:
        mask[:, :, i] = 0  # không chia
    else:
        mask[:, :, i] = 1  # sẽ chia

# Thực hiện chia 1000 với các phần tử không bị mask
cameras = cameras / (1 + (mask == 1) * 999)  # cách tránh chia cho 0

# Cập nhật lại cameras
dataset["inputs"]["cameras"] = cameras

# ===================== Xử lý targets ======================
# targets shape: (batchsize, 4, 20)
targets = dataset["inputs"]["targets"]

# Tạo mask chọn các vị trí index % 4 == 3
indices = torch.arange(20)
condition = indices % 4 == 3
idx_3 = condition.nonzero().squeeze()  # vị trí index % 4 == 3

# Lặp qua từng index trong idx_3 để xử lý
for idx in idx_3:
    valid_mask = targets[:, :, idx] != -1  # chỉ xử lý nếu giá trị khác -1
    for offset in [0, 1, 2]:
        targets[:, :, idx - offset - 1][valid_mask] /= 1000.0

# Cập nhật lại targets
dataset["inputs"]["targets"] = targets

# Lưu lại file mới
torch.save(dataset, "dataset_4v4/75k_dataset_modified.pt")
