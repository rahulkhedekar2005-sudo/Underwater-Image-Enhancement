import os
import cv2
import torch
from torch.utils.data import Dataset

class UnderwaterDataset(Dataset):
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.image_files = sorted(os.listdir(input_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.input_dir, img_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))

        # Convert to tensor (C, H, W)
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1) / 255.0

        return img
