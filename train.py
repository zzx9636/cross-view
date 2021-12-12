import os
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from crossView import PVA_transformer, Argoverse
from opt import get_args
from datetime import datetime


from utils import mean_IU, mean_precision
import wandb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, record = False):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self.record = record
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        if self.record:
            wandb.log({"lr": rate})
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
    def state_dict(self):
        return self.optimizer.state_dict()

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines      

class Trainer:
    def __init__(self):
        self.opt = get_args()
        self.models = {}
        self.weight = {"static": self.opt.static_weight, "dynamic": self.opt.dynamic_weight}
        self.seed = self.opt.global_seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.create_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.epoch = 0
        self.start_epoch = 0
                 
        if self.seed != 0:
            self.set_seed()  # set seed

        # Initializing models
        self.model = PVA_transformer()
        if torch.cuda.device_count() > 1:
            print("Use ", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        
       
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
        
        # Optimization
        self.optimizer = NoamOpt(512, 1, int(0.1*len(self.train_loader)*self.opt.num_epochs),
            torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), record = self.opt.record)     

        if self.opt.load_weights_folder != "":
            self.load_model()
            
        # Save log and models path
        now = datetime.now()

        self.opt.save_path = os.path.join(self.opt.save_path, now.strftime("%Y%m%d-%H%M%S"))
        if self.opt.record:
            wandb.init(project="cross-view", entity="zzx9636", config={"epochs": self.opt.num_epochs, 
                        "batch_size": self.opt.batch_size})

            wandb.define_metric("eval/*", step_metric="eval/step")

        print(
            "There are {:d} training items and {:d} validation items\n".format(
                len(train_dataset),
                len(val_dataset)))

    def train(self):
        #self.validation()
        for self.epoch in range(self.start_epoch, self.opt.num_epochs + 1):
            self.run_epoch()
            self.validation()
            if (self.epoch%5)==0:
                self.save_model()

    def run_epoch(self):
        for inputs in self.train_loader:
            self.model.train()
            self.model.zero_grad()
            for key, input in inputs.items():
                if key != "filename":
                    inputs[key] = input.to(self.device)
            _, losses = self.model(inputs)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5, error_if_nonfinite=True)
            self.optimizer.step()
            if self.opt.record:
                wandb.log({"loss": losses["loss"], "topview_loss": losses["topview_loss"], 
                        "transform_loss": losses["transform_loss"]}) 

    def validation(self):
        iou, mAP = np.array([0., 0., 0.]), np.array([0., 0., 0.])
        with torch.no_grad():
            for inputs in self.val_loader:
                
                self.model.eval()
                for key, input in inputs.items():
                    if key != "filename":
                        inputs[key] = input.to(self.device)
                outputs, _ = self.model(inputs)
                pred = np.squeeze(
                    torch.argmax(
                        outputs.detach(),
                        1).cpu().numpy())
                true = np.squeeze(
                    inputs["combine"].detach().cpu().numpy())
                #print(mean_IU(pred, true), mean_precision(pred, true))
                iou += mean_IU(pred, true)
                mAP += mean_precision(pred, true)
        iou /= len(self.val_loader)
        mAP /= len(self.val_loader)
        print("Epoch: %d | Validation: mIOU: %.4f, %.4f mAP: %.4f, %.4f" % (self.epoch, iou[1], iou[2], mAP[1], mAP[2]))
        
        if self.opt.record:    
            log_dict = {"eval/step": self.epoch, "eval/map/mIOU": iou[1], "eval/map/mAP": mAP[1],
                        "eval/vehicle/mIOU": iou[2], "eval/vehicle/mAP": mAP[2]}
        
            wandb.log(log_dict)
            

    def save_model(self):
        save_path = os.path.join(
            self.opt.save_path,
            "weights_{}".format(
                self.epoch)
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model_path = os.path.join(save_path, "model.pth")
        state_dict = self.model.state_dict()
        state_dict['epoch'] = self.epoch
        torch.save(state_dict, model_path)
        
        # save optimizer
        optim_path = os.path.join(save_path, "{}.pth".format("adam"))
        torch.save(self.optimizer.state_dict(), optim_path)
        print("Save models to ", save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(
            self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print(
            "loading model from folder {}".format(
                self.opt.load_weights_folder))

        path = os.path.join(
            self.opt.load_weights_folder,
            "model.pth")
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path)
        if 'epoch' in pretrained_dict:
            self.start_epoch = pretrained_dict['epoch']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        # loading adam state
        if self.opt.load_weights_folder == "":
            optimizer_load_path = os.path.join(
                self.opt.load_weights_folder, "adam.pth")
            if os.path.isfile(optimizer_load_path):
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.optimizer.load_state_dict(optimizer_dict)
            else:
                print("Cannot find Adam weights so Adam is randomly initialized")

    def set_seed(self):
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
     
if __name__ == "__main__":
    start_time = time.ctime()
    print(start_time)
    trainer = Trainer()
    trainer.train()
    end_time = time.ctime()
    print(end_time)