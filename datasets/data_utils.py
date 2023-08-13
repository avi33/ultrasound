import torchvision.transforms as T

def create_dataset(args):
    if args.dataset == 'kaggle':
        data_path = r'/media/avi/54561652561635681/datasets/ultrasound-kaggle/Dataset_BUSI_with_GT/benign'
        from datasets.kaggle_dataset import KaggleUltrasoundDataset as Dataset
        train_augs = T.Compose([#T.ColorJitter(hue=.05, saturation=.05),
                                T.Grayscale(),
                                T.RandomHorizontalFlip(p=0.5),                                
                                T.RandomRotation(degrees=15),
                                T.RandomResizedCrop(size=(128, 128), scale=(0.5, 1.0)),
                                T.ToTensor(), 
                                T.Normalize([0.5], [0.5])])
        
        test_augs = T.Compose([T.Grayscale(),
                                T.ToTensor(),
                               T.Resize((128, 128)),
                            T.Normalize([0.5], [0.5])])

        train_set = Dataset(mode='train', transform=train_augs, data_path=data_path)    
        test_set = Dataset(mode='train', transform=test_augs, data_path=data_path)
    
    else:
        raise ValueError("wrong dataset {}".format(args.dataset))
    
    return train_set, test_set