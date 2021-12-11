import os
import torch
from torch.utils.data import DataLoader
from crossView import  Argoverse
from opt import get_args
import tqdm
from datetime import datetime


from utils import mean_IU, mean_precision
import wandb


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


class test:
    def __init__(self):
        self.opt = get_args()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Data Loaders
        fpath = os.path.join(
            os.path.dirname(__file__),
            "splits",
            "argo",
            "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        self.val_filenames = val_filenames
        self.train_filenames = train_filenames

        train_dataset = Argoverse(self.opt, train_filenames)
        val_dataset = Argoverse(self.opt, val_filenames, is_train=False)

        self.train_loader = DataLoader(
            dataset = train_dataset,
            batch_size = self.opt.batch_size,
            shuffle = True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)
        
        self.val_loader = DataLoader(
            dataset = val_dataset,
            batch_size = 1,
            shuffle = True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)
        
        for inputs in self.train_loader:
            for key, input in inputs.items():
                if key != "filename":
                    print(key, input.shape)
            break

        

if __name__ == "__main__":
    
    trainer = test()
    