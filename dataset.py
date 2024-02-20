from torch.utils.data import Dataset
from PIL import Image
import os
import torch

class StandfordCarsDataset(Dataset):
    def __init__(self, data_df, transforms):
        image_paths = []
        for idx, row in data_df.iterrows():
            image_path = row["image_path"]

            image_paths.append(image_path)
        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        return {"image": image}
    

class DiffusionDataset(Dataset):
        def __init__(self, root_dir, split='train', transform=None):
            """
            Args:
                root_dir (string): Directory with all the images.
                split (string): One of 'train' or 'test' to specify the split.
                transform (callable, optional): Optional transform to be applied on a sample.
            """
            self.root_dir = os.path.join(root_dir, split)
            self.transform = transform
            self.image_paths = [os.path.join(self.root_dir, fname) for fname in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, fname))]
            
        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            img_name = self.image_paths[idx]
            image = Image.open(img_name).convert('RGB')
            
            if self.transform:
                image = self.transform(image)

            return {'image': image}