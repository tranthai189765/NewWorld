import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import os
from worldmodel import WorldModel  # Import mô hình của bạn

# Load dataset
dataset = torch.load("10k_dataset_modified.pt")

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
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Khởi tạo mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WorldModel().to(device)

# Loss & Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early Stopping
early_stopping_patience = 1000000  # Số epoch chờ trước khi dừng nếu không cải thiện
best_valid_loss = float("inf")
patience_counter = 0

# TensorBoard setup
log_dir = "runs/worldmodel_training"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

# Train model với Early Stopping
num_epochs = 1000000
recent_checkpoint_epoch = -1  # Để lưu checkpoint trước epoch cuối cùng
for epoch in range(num_epochs):
    ### Training ###
    model.train()
    total_train_loss = 0
 
    for cameras, obstacles, targets, labels_batch in train_loader:
        cameras, obstacles, targets, labels_batch = (
            cameras.to(device),
            obstacles.to(device),
            targets.to(device),
            labels_batch.to(device),
        )

        optimizer.zero_grad()
        outputs = model(targets, obstacles, cameras)  # Truyền 3 đầu vào riêng biệt vào WorldModel
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
                labels_batch.to(device),
            )

            outputs = model(targets, obstacles, cameras)
            loss = criterion(outputs, labels_batch)
            total_valid_loss += loss.item()

    avg_valid_loss = total_valid_loss / len(valid_loader)

    # Log Loss lên TensorBoard
    writer.add_scalar("Loss/Train", avg_train_loss, epoch)
    writer.add_scalar("Loss/Validation", avg_valid_loss, epoch)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")

    # Early Stopping và Lưu checkpoint
    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        patience_counter = 0
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Best Valid Loss: {avg_valid_loss:.4f} ----- ")
        torch.save(model.state_dict(), "best_worldmodel.pth")  # Lưu checkpoint tốt nhất
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered. Training stopped.")
            break

    # Lưu checkpoint gần nhất (trước epoch cuối)
    if epoch == num_epochs - 2:  # Lưu checkpoint trước epoch cuối
        torch.save(model.state_dict(), "recent_checkpoint.pth")

    # Lưu checkpoint cuối cùng
    if epoch == num_epochs - 1:
        torch.save(model.state_dict(), "last_checkpoint.pth")

print("Training completed!")

# Đánh giá trên tập Test
model.load_state_dict(torch.load("best_worldmodel.pth"))  # Load mô hình tốt nhất

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for cameras, obstacles, targets, labels_batch in data_loader:
            cameras, obstacles, targets, labels_batch = (
                cameras.to(device),
                obstacles.to(device),
                targets.to(device),
                labels_batch.to(device),
            )

            outputs = model(targets, obstacles, cameras)
            loss = criterion(outputs, labels_batch)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

test_loss = evaluate_model(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}")

# Log Test Loss lên TensorBoard
writer.add_scalar("Loss/Test", test_loss, 0)
writer.close()
