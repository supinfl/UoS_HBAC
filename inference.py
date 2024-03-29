import sys
import os
import gc
import copy
import yaml
import random
import shutil
from time import time
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedGroupKFold

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.cuda import amp

import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from datetime import datetime
import functools

import sys
# sys.path.append('/kaggle/input/kaggle-kl-div')
from kaggle_kl_div import score

# setting about environment and data path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ROOT = Path.cwd()
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
SRC = ROOT / "src"

#input from competitions file
# DATA = INPUT / "hms-harmful-brain-activity-classification"
# TRAIN_SPEC = DATA / "train_spectrograms"
# TEST_SPEC = DATA / "test_spectrograms"
DATA = Path("./original_data")
TRAIN_SPEC = DATA / "train_spectrograms"
TEST_SPEC = DATA / "test_spectrograms"



# TMP = ROOT / "tmp"
# TRAIN_SPEC_SPLIT = TMP / "train_spectrograms_split"
# TEST_SPEC_SPLIT = TMP / "test_spectrograms_split"

TRAIN_SPEC_SPLIT = DATA / "train_spectrograms_split"
TEST_SPEC_SPLIT = DATA / "test_spectrograms_split"
DATA.mkdir(exist_ok=True)
TRAIN_SPEC_SPLIT.mkdir(exist_ok=True)
TEST_SPEC_SPLIT.mkdir(exist_ok=True)


#Here is the wrapper defination for logging running time

log_filename = ROOT/'fixed_training_log.log'

logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def log_time(func):
    """warpper for logging running time"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        logging.info(f"{func.__name__} took {end_time - start_time:.4f} seconds.")
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds.")
        return result

    return wrapper


#Read data
@log_time
def load_data(DATA):
    train = pd.read_csv(DATA / "train.csv")
    # convert vote to probability format
    train[CLASSES] /= train[CLASSES].sum(axis=1).values[:, None]
    print(train.shape)
    return train

# Split the data using StratifiedGroupKFold
@log_time
def fold_train_data(train):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDAM_SEED)

    # initialized fold num to -1
    # fold based on expert_consensus(output label)
    train["fold"] = -1

    for fold_id, (_, val_idx) in enumerate(
        sgkf.split(train, y=train["expert_consensus"], groups=train["patient_id"])
    ):
        train.loc[val_idx, "fold"] = fold_id
    return train

# Reading spectrogram parquet file based on spectrogram_id, save into npy file
@log_time
def reading_spectrogram(train):
    logging.info('train.head():',train.head(20))
    logging.info('train.groupby("spectrogram_id").head():',train.groupby("spectrogram_id").head(20))
    print('train.head():',train.head(20))
    print('train.groupby("spectrogram_id").head():',train.groupby("spectrogram_id").head(20))
    for spec_id, df in tqdm(train.groupby("spectrogram_id")):
        # print('df:',df)
        spec = pd.read_parquet(TRAIN_SPEC / f"{spec_id}.parquet")
        #Transform as hz, time style
        spec_arr = spec.fillna(0).values[:, 1:].T.astype("float32")  # (Hz, Time) = (400, 300)
        # for spec_offset, label_id in df[
        #     ["spectrogram_label_offset_seconds", "label_id"]
        # ].astype(int).values: 不需要转换类型，否则win平台上文件名精度会出问题，出现负数
        for spec_offset, label_id in df[
            ["spectrogram_label_offset_seconds", "label_id"]
        ].values:
            spec_offset = spec_offset // 2
            split_spec_arr = spec_arr[:, spec_offset: spec_offset + 300]
            np.save(TRAIN_SPEC_SPLIT / f"{label_id}.npy" , split_spec_arr)
    return 

# def model structure
class HMSHBACSpecModel(nn.Module):

    def __init__(
            self,
            model_name: str,
            pretrained: bool,
            in_channels: int,
            num_classes: int,
        ):
        super().__init__() # Call the initialization method of the parent class (nn.Module)
        self.model = timm.create_model(
            model_name=model_name, pretrained=pretrained,
            num_classes=num_classes, in_chans=in_channels)

    def forward(self, x):
        h = self.model(x)      

        return h
    
# def filepath and label style 
FilePath = tp.Union[str, Path]
Label = tp.Union[int, float, np.ndarray]

# Define a custom dataset class for spectrogram images
# initializing from torch.utils.data.Dataset
class HMSHBACSpecDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        image_paths: tp.Sequence[FilePath],
        labels: tp.Sequence[Label],
        transform: A.Compose,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        label = self.labels[index]

        img = np.load(img_path)  # shape: (Hz, Time) = (400, 300)
        
        # log transform
        img = np.clip(img,np.exp(-4), np.exp(8)) #clipping values for stability
        img = np.log(img)
        
        # normalize per image
        eps = 1e-6
        img_mean = img.mean(axis=(0, 1))
        img = img - img_mean
        img_std = img.std(axis=(0, 1))
        img = img / (img_std + eps)

        img = img[..., None] # shape: (Hz, Time) -> (Hz, Time, Channel)
        img = self._apply_transform(img)

        return {"data": img, "target": label}

    def _apply_transform(self, img: np.ndarray):
        """apply transform to image and mask"""
        transformed = self.transform(image=img)
        img = transformed["image"]
        return img
    
#Combine original metric with logits
class KLDivLossWithLogits(nn.KLDivLoss):

    def __init__(self):
        super().__init__(reduction="batchmean")

    def forward(self, y, t):
        y = nn.functional.log_softmax(y,  dim=1)
        loss = super().forward(y, t)

        return loss


class KLDivLossWithLogitsForVal(nn.KLDivLoss):
    
    def __init__(self):
        """"""
        super().__init__(reduction="batchmean")
        self.log_prob_list  = []
        self.label_list = []

    def forward(self, y, t):
        y = nn.functional.log_softmax(y, dim=1)
        self.log_prob_list.append(y.numpy())
        self.label_list.append(t.numpy())
        
    def compute(self):
        log_prob = np.concatenate(self.log_prob_list, axis=0)
        label = np.concatenate(self.label_list, axis=0)
        final_metric = super().forward(
            torch.from_numpy(log_prob),
            torch.from_numpy(label)
        ).item()
        self.log_prob_list = []
        self.label_list = []
        
        return final_metric
    


def set_random_seed(seed: int = 42, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    # torch.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = deterministic  # type: ignore
    
def to_device(
    tensors: tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor]],
    device: torch.device, *args, **kwargs
):
    if isinstance(tensors, tuple):
        return (t.to(device, *args, **kwargs) for t in tensors)
    elif isinstance(tensors, dict):
        return {
            k: t.to(device, *args, **kwargs) for k, t in tensors.items()}
    else:
        return tensors.to(device, *args, **kwargs)


def get_path_label(val_fold, train_all: pd.DataFrame):
    """Get file path and target info."""
    print(train_all.head(10))
    train_idx = train_all[train_all["fold"] != val_fold].index.values
    val_idx   = train_all[train_all["fold"] == val_fold].index.values
    img_paths = []
    labels = train_all[CLASSES].values
    for label_id in train_all["label_id"].values:
        img_path = TRAIN_SPEC_SPLIT / f"{label_id}.npy"
        img_paths.append(img_path)

    train_data = {
        "image_paths": [img_paths[idx] for idx in train_idx],
        "labels": [labels[idx].astype("float32") for idx in train_idx]}

    val_data = {
        "image_paths": [img_paths[idx] for idx in val_idx],
        "labels": [labels[idx].astype("float32") for idx in val_idx]}
    
    return train_data, val_data, train_idx, val_idx

# resize images and convert to tensors
def get_transforms(CFG):
    train_transform = A.Compose([
        A.Resize(p=1.0, height=CFG.img_size, width=CFG.img_size),
        ToTensorV2(p=1.0)
    ])
    val_transform = A.Compose([
        A.Resize(p=1.0, height=CFG.img_size, width=CFG.img_size),
        ToTensorV2(p=1.0)
    ])
    return train_transform, val_transform

@log_time
def train_one_fold(CFG, val_fold, train_all, output_path):
    """Main"""
    print('into train_one_fold function')
    torch.backends.cudnn.benchmark = True
    set_random_seed(CFG.seed, deterministic=CFG.deterministic)
    device = torch.device(CFG.device)
    
    train_path_label, val_path_label, _, _ = get_path_label(val_fold, train_all)
    train_transform, val_transform = get_transforms(CFG)
    
    train_dataset = HMSHBACSpecDataset(**train_path_label, transform=train_transform)
    val_dataset = HMSHBACSpecDataset(**val_path_label, transform=val_transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CFG.batch_size, num_workers=4, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=CFG.batch_size, num_workers=4, shuffle=False, drop_last=False)
    
    # model = HMSHBACSpecModel(
    #     model_name=CFG.model_name, pretrained=True, num_classes=6, in_channels=1)
    model = HMSHBACSpecModel(
        model_name=CFG.model_name, pretrained=False, num_classes=6, in_channels=1)
    model.to(device)
    print('train_one_fold function: start to optimize')
    optimizer = optim.AdamW(params=model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    #Use dynamic learning rate
    scheduler = lr_scheduler.OneCycleLR(
        optimizer=optimizer, epochs=CFG.max_epoch,
        pct_start=0.0, steps_per_epoch=len(train_loader),
        max_lr=CFG.lr, div_factor=25, final_div_factor=4.0e-01
    )
    
    loss_func = KLDivLossWithLogits()
    loss_func.to(device)
    loss_func_val = KLDivLossWithLogitsForVal()
    
    use_amp = CFG.enable_amp
    scaler = amp.GradScaler(enabled=use_amp)
    
    best_val_loss = 1.0e+09
    best_epoch = 0
    train_loss = 0
    
    for epoch in range(1, CFG.max_epoch + 1):
        epoch_start = time()
        model.train()
        for batch in tqdm(train_loader):
            batch = to_device(batch, device)
            x, t = batch["data"], batch["target"]
                
            optimizer.zero_grad()
            with amp.autocast(use_amp):
                y = model(x)
                loss = loss_func(y, t)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
            
        model.eval()
        for batch in val_loader:
            x, t = batch["data"], batch["target"]
            x = to_device(x, device)
            with torch.no_grad(), amp.autocast(use_amp):
                y = model(x)
            y = y.detach().cpu().to(torch.float32)
            loss_func_val(y, t)
        val_loss = loss_func_val.compute()        
        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            print("save model")
            torch.save(model.state_dict(), str(output_path / f'snapshot_epoch_{epoch}.pth'))
        
        elapsed_time = time() - epoch_start
        print(f"[epoch {epoch}] train loss: {train_loss: .6f}, val loss: {val_loss: .6f}, elapsed_time: {elapsed_time: .3f}")
        logging.info(f"[epoch {epoch}] train loss: {train_loss: .6f}, val loss: {val_loss: .6f}, elapsed_time: {elapsed_time: .3f}")
        if epoch - best_epoch > CFG.es_patience:
            print("Early Stopping!")
            logging.info("Early Stopping!")
            break
            
        train_loss = 0
            
    return val_fold, best_epoch, best_val_loss

@log_time
def fold_training(CFG,train):
    print('into fold_training')
    score_list = []
    for fold_id in FOLDS:
        output_path = Path(f"fold{fold_id}")
        output_path.mkdir(exist_ok=True)
        print(f"[fold{fold_id}]")
        print('best_score now:',train_one_fold(CFG, fold_id, train, output_path))
        score_list.append(train_one_fold(CFG, fold_id, train, output_path))
    return score_list

@log_time
def run_inference_loop(model, loader, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x = to_device(batch["data"], device)
            y = model(x)
            pred_list.append(y.softmax(dim=1).detach().cpu().numpy())
        
    pred_arr = np.concatenate(pred_list)
    del pred_list
    return pred_arr

# Setting for training
RANDAM_SEED = 1086
CLASSES = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
N_CLASSES = len(CLASSES)
# FOLDS = [0, 1, 2, 3, 4] 
FOLDS = [0, 1] 
N_FOLDS = len(FOLDS)

#training config
class CFG:
    model_name = None
    img_size = 512
    max_epoch = 9
    batch_size = 8
    # batch_size = 16
    lr = 1.0e-03
    weight_decay = 1.0e-02
    es_patience =  5
    seed = 1086
    deterministic = True
    enable_amp = False
    device = "cuda"


if __name__ == "__main__":
    overall_start_time = time()
    print(f"Log file path: {log_filename.absolute()}")
    
    logging.info('--------------------------------------------------')
    # logging.info(f'Into loading stage')

    # train = load_data(DATA)
    # # train = train.groupby("spectrogram_id").head(1).reset_index(drop=True)
    # print('train.shape:',train.shape)
    # train = fold_train_data(train)
    # print(train.groupby("fold")[CLASSES].sum())
    # reading_spectrogram(train)
    # train.to_csv("modified_train.csv", index=False)

    # # Training stage
    train = pd.read_csv("modified_train.csv")
    # model_list = ['efficientnet_b2','efficientnet_b4','efficientnet_b7','res2net50d']
    model_list = ['efficientnet_b4']
    for model_name in model_list:
        logging.info('--------------------------------------------------')
        logging.info(f'Into training stage of {model_name}')
        cfg = CFG()
        cfg.model_name = model_name
        score_list = fold_training(cfg,train)
        print(score_list)
        logging.info(f'score_list of {model_name}:{score_list}')
        # score_list = [(0, 4, 3.112013339996338), (1, 3, 2.617021322250366)]
        logging.info(f'Move models of {model_name}into new folder')
        for (fold_id, best_epoch, _) in score_list:
            exp_dir_path = Path(f"fold{fold_id}")
            best_model_path = exp_dir_path / f"snapshot_epoch_{best_epoch}.pth"
            target_dir = Path(f"./best_model_fold/{model_name}")
            target_dir.mkdir(parents=True, exist_ok=True)
            copy_to = f"./best_model_fold/{model_name}/{fold_id}.pth"
            shutil.copy(best_model_path, copy_to)
            # for p in exp_dir_path.glob("*.pth"):
            #     p.unlink()

        ## Inference stage
        train = pd.read_csv("modified_train.csv")
        logging.info(f'Into inference stage of {model_name}')
        best_log_list = []
        print(train.head())
        label_arr = train[CLASSES].values
        oof_pred_arr = np.zeros((len(train), N_CLASSES))
        score_list = []

        for fold_id in range(N_FOLDS):
            logging.info(f'Into inference stage of {model_name}:fold{fold_id}')
            print(f"\n[fold {fold_id}]")
            device = torch.device(CFG.device)
            cfg = CFG()
            cfg.model_name = model_name
            
            # get_dataloader
            _, val_path_label, _, val_idx = get_path_label(fold_id, train)
            _, val_transform = get_transforms(CFG)
            val_dataset = HMSHBACSpecDataset(**val_path_label, transform=val_transform)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=CFG.batch_size, num_workers=4, shuffle=False, drop_last=False)
            
            # # get model
            model_path = f"./best_model_fold/{model_name}/{fold_id}.pth"
            model = HMSHBACSpecModel(
                model_name=cfg.model_name, pretrained=False, num_classes=6, in_channels=1)
            model.load_state_dict(torch.load(model_path, map_location=device))
            
            # # inference
            val_pred = run_inference_loop(model, val_loader, device)
            oof_pred_arr[val_idx] = val_pred
            
            del val_idx, val_path_label
            del model, val_loader
            # torch.cpu.empty_cache()
            torch.cuda.empty_cache()
            gc.collect()



        true = train[["label_id"] + CLASSES].copy()
        print('true:',true)
        oof = pd.DataFrame(oof_pred_arr, columns=CLASSES)
        oof.insert(0, "label_id", train["label_id"])
        print('oof:',oof)
        cv_score = score(solution=true, submission=oof, row_id_column_name='label_id')
        print(f'CV Score KL-Div for {model_name}',cv_score)
        logging.info(f'CV Score KL-Div for {model_name}: {cv_score}')

        overall_end_time = time()
        total_time_taken = overall_end_time - overall_start_time
        logging.info(f" {model_name} program execution time: {total_time_taken:.4f} seconds.")
        print(f" {model_name} program execution time: {total_time_taken:.4f} seconds.")
    overall_end_time = time()
    total_time_taken = overall_end_time - overall_start_time
    logging.info(f"Total program execution time: {total_time_taken:.4f} seconds.")
    print(f"Total program execution time: {total_time_taken:.4f} seconds.")