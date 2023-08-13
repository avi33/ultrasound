import torch
import glob
from PIL import Image

class KaggleUltrasoundDataset():
    def __init__(self, data_path, mode=True, transform=None):
        # Call the super constructor to initialize the dataset
        super().__init__()
        self.transform = transform
        self.mode = mode
        fnames = glob.glob(data_path + '/*.png')
        self.fnames = sorted([f for f in fnames if f.split('/')[-1].split('_')[-1][:-4] != 'mask'])        

    def __getitem__(self, index):
        fname = self.fnames[index]
        img = Image.open(fname)

        if self.transform is not None:        
            img = self.transform(img)

        return img
    
    def __len__(self):
        return len(self.fnames)