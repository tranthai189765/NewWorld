import torch

# Load file .pt
dataset = torch.load("dataset/10k_dataset_modified_labels3.pt")

# Lấy labels (shape: batchsize, 8, 5)
labels = dataset["labels"]

# Tìm các batch index có giá trị tại vị trí [i, 0, 0] == 1
mask = labels[:, 0, 0] == 1  # shape: (batchsize,)

# Lấy tất cả tensor thỏa điều kiện
matching_tensors = labels[mask]

# In ra toàn bộ các tensor đó
print("Các tensor labels[i] có labels[i, 0, 0] == 1:")
for i, tensor in enumerate(matching_tensors):
    print(f"\nTensor {i}:")
    print(tensor)