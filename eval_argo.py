import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from crossView import PVA_model, Argoverse
from opt import get_args
from tqdm import tqdm
from datetime import datetime

from utils import mean_IU, mean_precision, extract_masks
import wandb
from PIL import Image



def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


class eval_argo:
    def __init__(self):
        self.opt = get_args()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.create_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())

        # Initializing models
        self.model = PVA_model(self.opt)
        self.model.to_device(self.device)
       
        # Data Loaders
        fpath = os.path.join(
            os.path.dirname(__file__),
            "splits",
            "argo",
            "{}_files.txt")

        val_filenames = readlines(fpath.format("val"))
        self.val_filenames = val_filenames

        val_dataset = Argoverse(self.opt, val_filenames, is_train=False)

        
        self.val_loader = DataLoader(
            dataset = val_dataset,
            batch_size = 1,
            shuffle = True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)

        
        self.load_model()
            
        # Save log and models path
        now = datetime.now()

        self.opt.save_path = os.path.join(self.opt.save_path, now.strftime("%Y%m%d-%H%M%S"))
        



    def run_epoch(self):
        for inputs in self.train_loader:
            self.model.train()
            self.optimizer.zero_grad()
            for key, input in inputs.items():
                if key != "filename":
                    inputs[key] = input.to(self.device)
            _, losses = self.model(inputs)
            losses["loss"].backward()
            self.optimizer.step()
            if self.opt.record:
                wandb.log({"loss": losses["loss"], "topview_loss": losses["topview_loss"], 
                        "transform_loss": losses["transform_loss"]}) 
                        #"transform_topview_loss": losses["transform_topview_loss"]})

    def validation(self):
        iou, mAP = np.array([0., 0., 0.]), np.array([0., 0., 0.])
        softmax = torch.nn.Softmax(dim=1)
        #trans_iou, trans_mAP = np.array([0., 0.]), np.array([0., 0.])
        with torch.no_grad():
            for inputs in tqdm(self.val_loader):
                self.model.eval()
                for key, input in inputs.items():
                    if key != "filename":
                        inputs[key] = input.to(self.device)
                outputs, _ = self.model(inputs)
                outputs_scale = softmax(outputs["topview"].detach().cpu())
                pred = np.squeeze(torch.argmax(outputs_scale, 1).numpy())
                true = np.squeeze(inputs["combine"].detach().cpu().numpy())

                self.save_topview(inputs['filename'][0], true, pred)
                
                iou += mean_IU(pred, true)
                mAP += mean_precision(pred, true)
        iou /= len(self.val_loader)
        mAP /= len(self.val_loader)
        print(" Validation: mIOU Map: %.4f, mIOU Vehicle:%.4f; mAP Map: %.4f, mAP Map: %.4f" % (iou[1], iou[2], mAP[1], mAP[2]))
    
    def generate_img(self, segm):
        h,w = segm.shape
        output = np.zeros((h,w,3))
        # 0 is background - black

        # 1 is road
        output[segm==1] = [1,1,1]   
        # 2 is car
        output[segm==2] = [0,0,1]
        return Image.fromarray(np.uint8(output*255))
        
    
    def save_topview(self, filename, gt, pred):
        h,w = gt.shape
        new_im = Image.new('RGB', (2*w,h))
        gt_im = self.generate_img(gt)
        pred_im = self.generate_img(pred)
        
        new_im.paste(pred_im, (0,0))
        new_im.paste(gt_im, (w,0))
        
        sub_path = filename.split('/')[:-2]
        file_name = filename.split('/')[-1]
        full_path = os.path.join(self.opt.out_dir, self.opt.model_name, *sub_path)
        
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        new_im.save(os.path.join(full_path, file_name))


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

        for key in self.model.models.keys():
            if "discriminator" not in key:
                print("Loading {} weights...".format(key))
                path = os.path.join(
                    self.opt.load_weights_folder,
                    "{}.pth".format(key))
                model_dict = self.model.models[key].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.model.models[key].load_state_dict(model_dict)

      
if __name__ == "__main__":
    start_time = time.ctime()
    print(start_time)
    eval = eval_argo()
    eval.validation()
    end_time = time.ctime()
    print(end_time)
