from torch.utils.data import DataLoader
from dataset_loader import UnderwaterDataset

dataset = UnderwaterDataset("dataset/train/input")

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

batch = next(iter(loader))
print("Batch shape:", batch.shape)
