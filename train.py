import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_loader import UnderwaterDataset
from models.cnn_model import SimpleCNN

# Device
device = torch.device("cpu")

# Dataset & DataLoader
dataset = UnderwaterDataset("dataset/train/input")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model
model = SimpleCNN().to(device)

# Loss & Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---- TRAINING LOOP (1 EPOCH) ----
model.train()

for batch_idx, images in enumerate(loader):
    images = images.to(device)

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, images)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Batch {batch_idx + 1}, Loss: {loss.item():.4f}")

    # Run only a few batches for safety
    if batch_idx == 4:
        break

print("Training loop executed successfully")
# Save trained model
torch.save(model.state_dict(), "underwater_cnn.pth")
print("Model saved as underwater_cnn.pth")
