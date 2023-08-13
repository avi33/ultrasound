from copy import deepcopy
import os
from statistics import mode
import torch
from torchvision.datasets import CIFAR10 as Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import argparse
from pathlib import Path
import yaml
from kornia.metrics import psnr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f_res", default='outputs/1', type=Path)
    args = parser.parse_args()
    return args

def inference_cifar():
    global device
    args = parse_args() 
    f_res = args.f_res
    with open(f_res / "args.yml", "r") as f:
        args = yaml.load(f, yaml.Loader)
    args.f_res = f_res
    
    data_path = r'/media/avi/54561652561635681/datasets/ultrasound-kaggle/Dataset_BUSI_with_GT/benign'
    from datasets.kaggle_dataset import KaggleUltrasoundDataset as Dataset
    test_augs = T.Compose([T.Grayscale(),
                                T.ToTensor(),
                               T.Resize((128, 128)),
                            T.Normalize([0.5], [0.5])])

    test_set = Dataset(mode='test', transform=test_augs, data_path=data_path)
    
    test_loader = DataLoader(test_set, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            drop_last=False, 
                            num_workers=4, 
                            pin_memory=True)
    
    from modules.models import Net    
    net = Net(emb_dim=128, nf=16, factors=[2, 2, 2])
    net.eval()
    net.to(device)
    chkpnt = torch.load(args.f_res / 'chkpnt.pt', map_location=torch.device(device))
    net.load_state_dict(chkpnt['model_dict'], strict=True)    
    psnr_test = 0
    for i, x in enumerate(test_loader):
        if i % 10 == 0:
            print("{}/{}".format(i, len(test_loader)))
        x = x.to(device)        
        with torch.no_grad():
            x_est = net(x)
        psnr_test += psnr(x_est, x, max_val=1)
    psnr_test = psnr_test/len(test_loader)
    print(psnr_test)


if __name__ == "__main__":    
    inference_cifar()