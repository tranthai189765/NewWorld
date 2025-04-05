import torch
# Load dataset từ file
dataset = torch.load("dataset_4v4/10k_dataset_modified.pt")  # Thay bằng tên file thực tế

# Kiểm tra keys (nếu lưu dưới dạng dictionary)
print(dataset.keys())

# Xem thử một số dữ liệu (giả sử dữ liệu lưu dưới dạng dictionary)
print("Cameras Sample:", dataset["inputs"]["cameras"].shape)
print("Obstacles Sample:", dataset["inputs"]["obstacles"].shape)
print("Targets Sample:", dataset["inputs"]["targets"].shape)
print("Targets Label: ", dataset["labels"].shape)

# Mỗi cameras_input có số chiều [4, 45]: tôi cần chia 1000 cho các giá trị index % 9 != {5,7,8} theo chiều thú 2
# Mỗi obstacles_input có cố chiều là [9, 3] : tôi cần chia các giá trị cho 1000 đối với các giá trị khác -1
# Mỗi targets_input có số chiều là [8, 20] : tôi cần chia 1000 cho các index % 4! = 3 theo chiều thứ 2 , đối với các giá trị khác -1
