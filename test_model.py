from models.cnn_model import SimpleCNN
import torch

model = SimpleCNN()
print(model)

# Dummy input: batch=1, channels=3, image size=128x128
x = torch.randn(1, 3, 128, 128)
y = model(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)

