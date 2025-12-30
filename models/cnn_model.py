import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=16, 
            kernel_size=3, 
            padding=1
        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=16, 
            out_channels=3, 
            kernel_size=3, 
            padding=1
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x
