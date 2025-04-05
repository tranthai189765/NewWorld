import torch

# Danh sách chứa dữ liệu từ tất cả các file
all_cameras, all_obstacles, all_targets, all_labels = [], [], [], []

# Duyệt qua tất cả các file dataset_0.pt -> dataset_366.pt
for i in range(500):  # 0 -> 366
    file_name = f"dataset_{i}.pt"
    dataset = torch.load(file_name)

    # Thêm dữ liệu từ từng file vào danh sách
    all_cameras.append(torch.tensor(dataset["inputs"]["cameras"], dtype=torch.float32))
    all_obstacles.append(torch.tensor(dataset["inputs"]["obstacles"], dtype=torch.float32))
    all_targets.append(torch.tensor(dataset["inputs"]["targets"], dtype=torch.float32))
    all_labels.append(torch.tensor(dataset["labels"], dtype=torch.float32))

# Gộp tất cả các tensor lại
merged_dataset = {
    "inputs": {
        "cameras": torch.cat(all_cameras, dim=0),
        "obstacles": torch.cat(all_obstacles, dim=0),
        "targets": torch.cat(all_targets, dim=0),
    },
    "labels": torch.cat(all_labels, dim=0),
}

# Lưu dataset đã gộp vào file
torch.save(merged_dataset, "10k_dataset_4v4.pt")

print("Gộp dữ liệu thành công, lưu vào 10k_dataset_4v4.pt")