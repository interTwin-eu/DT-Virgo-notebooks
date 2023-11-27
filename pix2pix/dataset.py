import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_preprocessing import load_image_train, load_image_test


BATCH_SIZE = 1

class Dataset(Dataset):
    def __init__(self, dir_path, size=256, train=True):
        self.is_train = train
        self.sizes = (size, size)
        items = []
        labels = []
        for data in os.listdir(dir_path):
            item = os.path.join(dir_path, data)
            items.append(item)
            labels.append(data)
        self.items = items
        self.labels = labels
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        if self.is_train:
            data = load_image_train(self.items[idx])
        else:
            data = load_image_test(self.items[idx])
        return data, self.labels[idx]
    
data_path = '../pix2pix/facades/train'
ds = Dataset(data_path, size=256)
ds.items.remove('../pix2pix/facades/train/.ipynb_checkpoints')

dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)