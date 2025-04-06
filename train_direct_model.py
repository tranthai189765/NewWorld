import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import os

from direct_model import WorldModel  # Import mô hình của bạn
from focal_loss import FocalLoss
import datetime
import time
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


# Load dataset
dataset = torch.load("dataset_4v4/100k_dataset_modified.pt")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 

# Lấy từng phần dữ liệu
inputs_cameras = torch.tensor(dataset["inputs"]["cameras"], dtype=torch.float32)
inputs_obstacles = torch.tensor(dataset["inputs"]["obstacles"], dtype=torch.float32)
inputs_targets = torch.tensor(dataset["inputs"]["targets"], dtype=torch.float32)
labels = torch.tensor(dataset["labels"], dtype=torch.float32)

# Tạo dataset gồm 3 đầu vào riêng biệt
full_dataset = TensorDataset(inputs_cameras, inputs_obstacles, inputs_targets, labels)

# Chia dataset thành train (70%), valid (15%), test (15%)
train_size = int(0.7 * len(full_dataset))
valid_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - valid_size

train_dataset, valid_dataset, test_dataset = random_split(full_dataset, [train_size, valid_size, test_size])

# DataLoader
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Tập hợp toàn bộ label từ train_dataset để tính class_weights
all_train_labels = []

for _, _, _, label in train_dataset:
    # label: (seq_len, num_classes) → cần chuyển về class index
    label_indices = label.argmax(dim=1).tolist()
    all_train_labels.extend(label_indices)

# print("test :",all_train_labels )
# Tính class_weights
class_weights_np = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_train_labels),
    y=all_train_labels
)

# Chuyển sang tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
manual_weights = torch.tensor([0.5, 1.5, 1.5, 1.5, 1.5], device=device)


# print("weight  = ",class_weights )
# time.sleep(100000)
# Khởi tạo mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = WorldModel(embed_dim=64, num_heads=8, ff_dim=256, num_layers=1).to(device)
model = WorldModel(embed_dim=512, num_heads=2, ff_dim=1024, num_layers=1).to(device)

# Loss & Optimizer
criterion = FocalLoss(gamma=4, weight=manual_weights)
# criterion = nn.CrossEntropyLoss(weight=manual_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Early Stopping
early_stopping_patience = 50  # Số epoch chờ trước khi dừng nếu không cải thiện
best_valid_loss = float("inf")
patience_counter = 0

# TensorBoard setup
log_dir = "runs/worldmodel_training"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

# Tạo file result.txt trong thư mục dataset
log_file_path = "dataset/result2.txt"
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Ghi log vào file với thời gian
def log_to_file(message):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # Lấy thời gian hiện tại
    log_message = f"[{current_time}] {message}"  # Thêm thời gian vào message
    with open(log_file_path, "a") as f:
        f.write(log_message + "\n")
        

# Timelast
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
timelast = timestamp
# Train model với Early Stopping
num_epochs = 1000
recent_checkpoint_epoch = -1  # Để lưu checkpoint trước epoch cuối cùng
for epoch in range(num_epochs):
    ### Training ###
    model.train()
    total_train_loss = 0
 
    for cameras, obstacles, targets, labels_batch in train_loader:
        # print("old_labels = ", labels_batch)
        cameras, obstacles, targets, labels_batch = (
            cameras.to(device),
            obstacles.to(device),
            targets.to(device),
            labels_batch.argmax(dim=2).to(device),
        )

        optimizer.zero_grad()
        outputs,_ = model(targets, obstacles, cameras)  # Truyền 3 đầu vào riêng biệt vào WorldModel
        # print("output = ", outputs, "current label = ", labels_batch)
        outputs = outputs.view(-1, outputs.size(-1))          # (batch_size * seq_len, num_classes)
        labels_batch = labels_batch.view(-1)                  # (batch_size * seq_len)
        # print("output = ", outputs, "current label = ", labels_batch)
        # time.sleep(10000)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    ### Validation ###
    model.eval()
    total_valid_loss = 0

    with torch.no_grad():
        for cameras, obstacles, targets, labels_batch in valid_loader:
            cameras, obstacles, targets, labels_batch = (
                cameras.to(device),
                obstacles.to(device),
                targets.to(device),
                labels_batch.argmax(dim=2).to(device),
            )

            outputs,_ = model(targets, obstacles, cameras)
            outputs = outputs.view(-1, outputs.size(-1))          # (batch_size * seq_len, num_classes)
            labels_batch = labels_batch.view(-1)                  # (batch_size * seq_len)
            loss = criterion(outputs, labels_batch)
            total_valid_loss += loss.item()

    avg_valid_loss = total_valid_loss / len(valid_loader)

    # Log Loss lên TensorBoard
    writer.add_scalar("Loss/Train", avg_train_loss, epoch)
    writer.add_scalar("Loss/Validation", avg_valid_loss, epoch)

    log_message = f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}"
    log_to_file(log_message)
    print(log_message)

    # Early Stopping và Lưu checkpoint
    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        patience_counter = 0
        log_message = f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Best Valid Loss: {avg_valid_loss:.4f} ----- "
        log_to_file(log_message)
        print(log_message)
        torch.save(model.state_dict(), f"best_directmodel1234_{timestamp}.pth")  # Lưu checkpoint tốt nhất
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered. Training stopped.")
            break

    # Lưu checkpoint gần nhất (trước epoch cuối)
    if epoch == num_epochs - 2:  # Lưu checkpoint trước epoch cuối
        torch.save(model.state_dict(), f"recent_checkpoint1234_{timestamp}.pth")

    # Lưu checkpoint cuối cùng
    if epoch == num_epochs - 1:
        torch.save(model.state_dict(), f"last_checkpoint1234_{timestamp}.pth")

print("Training completed!")

# Đánh giá trên tập Test
model.load_state_dict(torch.load(f"best_directmodel1234_{timestamp}.pth"))  # Load mô hình tốt nhất

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for cameras, obstacles, targets, labels_batch in data_loader:
            cameras, obstacles, targets, labels_batch = (
                cameras.to(device),
                obstacles.to(device),
                targets.to(device),
                labels_batch.argmax(dim=2).to(device),
            )

            outputs,_ = model(targets, obstacles, cameras)
            outputs = outputs.view(-1, outputs.size(-1))          # (batch_size * seq_len, num_classes)
            labels_batch = labels_batch.view(-1)                  # (batch_size * seq_len)
            loss = criterion(outputs, labels_batch)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

test_loss = evaluate_model(model, test_loader, criterion)
log_message = f"Test Loss: {test_loss:.4f}"
log_to_file(log_message)
print(log_message)

# Log Test Loss lên TensorBoard
writer.add_scalar("Loss/Test", test_loss, 0)

def compute_metrics(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for cameras, obstacles, targets, labels_batch in data_loader:
            cameras, obstacles, targets, labels_batch = (
                cameras.to(device),
                obstacles.to(device),
                targets.to(device),
                labels_batch.argmax(dim=2).to(device),
            )

            outputs, _ = model(targets, obstacles, cameras)
            outputs = outputs.view(-1, outputs.size(-1))          # (batch_size * seq_len, num_classes)
            labels_batch = labels_batch.view(-1)                  # (batch_size * seq_len)

            predictions = torch.argmax(outputs, dim=1)            # logits -> predicted class
            # print("predictions ", predictions, "labels",  labels_batch)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    # Tính các metric
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')  # hoặc 'weighted' tùy bạn
    report = classification_report(all_labels, all_preds, digits=4)

    return acc, f1, report

# Gọi hàm và in kết quả
accuracy, f1_macro, report = compute_metrics(model, test_loader)
log_message = f"Test Accuracy: {accuracy:.4f}, F1-macro: {f1_macro:.4f}"
log_to_file(log_message)
print(log_message)
print("Classification Report:\n", report)
writer.add_scalar("Metric/Test_Accuracy", accuracy, 0)
writer.add_scalar("Metric/Test_F1_macro", f1_macro, 0)
writer.close()