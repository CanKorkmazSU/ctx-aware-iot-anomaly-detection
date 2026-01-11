import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class RoomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(root_dir) else []

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

class CustomTransforms:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # Resize
        img = img.resize((self.size, self.size), Image.BILINEAR)
        # ToTensor
        img_np = np.array(img).astype(np.float32) / 255.0
        # HWC -> CHW
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        return img_tensor

def get_transforms(img_size=64):
    return CustomTransforms(img_size)
