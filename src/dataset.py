import torch

class OctoDataset(torch.utils.data.Dataset):
    def __init__(self, data, gt, sample_num):
        
        self.data = data
        self.gt = gt
        self.sample_num = sample_num

    def __len__(self):
        return len(self.data) // self.sample_num

    def __getitem__(self, idx):
        random_idx = torch.randint(0, len(self.data), (self.sample_num,))
        x = self.data[random_idx]
        y = self.gt[random_idx]
        return x, y