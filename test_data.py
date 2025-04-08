import torch
import numpy as np

# Thiết lập để in đầy đủ numpy array
np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=True)

# Load dataset từ file
dataset = torch.load("dataset_4v4/75k_dataset_30_times_4v4_modified.pt")  # Thay bằng đường dẫn thực tế

# Kiểm tra keys
print("Dataset keys:", dataset.keys())

# In thông tin shape
print("Cameras shape:", dataset["inputs"]["cameras"].shape)
print("Obstacles shape:", dataset["inputs"]["obstacles"].shape)
print("Targets shape:", dataset["inputs"]["targets"].shape)
print("Labels shape:", dataset["labels"].shape)

# In đầy đủ sample đầu tiên (chuyển sang numpy nếu là tensor)
print("\n--- Sample dữ liệu đầu tiên ---")

cameras_sample = dataset["inputs"]["cameras"][0].numpy()
obstacles_sample = dataset["inputs"]["obstacles"][0].numpy()
targets_sample = dataset["inputs"]["targets"][0].numpy()
labels_sample = dataset["labels"][0].numpy()

print("Cameras Sample:\n", cameras_sample)
print("Obstacles Sample:\n", obstacles_sample)
print("Targets Sample:\n", targets_sample)
print("Labels Sample:\n", labels_sample)