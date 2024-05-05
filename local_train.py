import albumentations as A
import gc
import matplotlib.pyplot as plt
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
import random
import time
import timm
import torch
import torch.nn as nn
from pathlib import Path


from albumentations.pytorch import ToTensorV2
from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, List

from sklearn.model_selection import KFold, GroupKFold
from skimage.transform import resize
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F
import logging
import functools
import pywt

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using', torch.cuda.device_count(), 'GPU(s)')



class config:
    AMP = True
    BATCH_SIZE_TRAIN = 8
    BATCH_SIZE_VALID = 8
    EPOCHS = 3
    FOLDS = 3
    FREEZE = False
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1e7
    MODEL = "tf_efficientnet_b0"
    NUM_FROZEN_LAYERS = 39
    NUM_WORKERS = 0 # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    TRAIN_FULL_DATA = False
    VISUALIZE = False
    WEIGHT_DECAY = 0.01
    
    
class paths:
    # OUTPUT_DIR = "/kaggle/working/"
    # PRE_LOADED_EEGS = '/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy'
    # PRE_LOADED_SPECTOGRAMS = '/kaggle/input/brain-spectrograms/specs.npy'
    # TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    # TRAIN_EEGS = "/kaggle/input/brain-eeg-spectrograms/EEG_Spectrograms/"
    # TRAIN_SPECTOGRAMS = "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    ROOT = Path.cwd()
    INPUT = ROOT / "input"
    OUTPUT_DIR = ROOT / "output"
    DATA = Path("./original_data")

    PRE_LOADED_EEGS = './kaggle/input/brain-eeg-spectrograms/eeg_specs.npy'
    PRE_LOADED_SPECTROGRAMS = './kaggle/input/brain-spectrograms/specs.npy'
    PRE_LOADED_Wavelets = './kaggle/input/brain-wavelets/specs.npy'
    
    TRAIN_SPECTROGRAMS = DATA / "train_spectrograms"
    TRAIN_EEGS = DATA / "train_eegs"
    TRAIN_CSV = DATA / "train.csv"

log_filename = paths.ROOT/'new_version_training_record.log'

logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
def log_time(func):
    """warpper for logging running time"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} took {end_time - start_time:.4f} seconds.")
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds.")
        return result

    return wrapper


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s: float):
    "Convert to minutes."
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since: float, percent: float):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))



def plot_spectrogram(spectrogram_path: str):
    """
    Source: https://www.kaggle.com/code/mvvppp/hms-eda-and-domain-journey
    Visualize spectogram recordings from a parquet file.
    :param spectrogram_path: path to the spectogram parquet.
    """
    sample_spect = pd.read_parquet(spectrogram_path)
    
    split_spect = {
        "LL": sample_spect.filter(regex='^LL', axis=1),
        "RL": sample_spect.filter(regex='^RL', axis=1),
        "RP": sample_spect.filter(regex='^RP', axis=1),
        "LP": sample_spect.filter(regex='^LP', axis=1),
    }
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    axes = axes.flatten()
    label_interval = 5
    for i, split_name in enumerate(split_spect.keys()):
        ax = axes[i]
        img = ax.imshow(np.log(split_spect[split_name]).T, cmap='viridis', aspect='auto', origin='lower')
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label('Log(Value)')
        ax.set_title(split_name)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time")

        ax.set_yticks(np.arange(len(split_spect[split_name].columns)))
        ax.set_yticklabels([column_name[3:] for column_name in split_spect[split_name].columns])
        frequencies = [column_name[3:] for column_name in split_spect[split_name].columns]
        ax.set_yticks(np.arange(0, len(split_spect[split_name].columns), label_interval))
        ax.set_yticklabels(frequencies[::label_interval])
    plt.tight_layout()
    plt.show()
    
@log_time   
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 

   
def sep():
    print("-"*100)
    

class CustomDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, config,
        augment: bool = False, mode: str = 'train',
        specs: Dict[int, np.ndarray] = None,
        eeg_specs: Dict[int, np.ndarray] = None,
        wavelets_spectrograms: Dict[int, np.ndarray] = None
    ): 
        self.df = df
        self.config = config
        self.batch_size = self.config.BATCH_SIZE_TRAIN
        self.augment = augment
        self.mode = mode
        self.spectrograms = specs if specs is not None else {}
        self.eeg_spectrograms = eeg_specs if eeg_specs is not None else {}
        self.wavelets_spectrograms = wavelets_spectrograms if wavelets_spectrograms is not None else {}
        
    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """
        return len(self.df)
        
    def __getitem__(self, index):
        """
        Generate one batch of data.
        """
        X, y = self.__data_generation(index)
        if self.augment:
            X = self.__transform(X) 
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def log_and_Standarize(self,img):
        # Log transform spectogram
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            # Standarize per image
            ep = 1e-6
            mu = np.nanmean(img.flatten())
            std = np.nanstd(img.flatten())
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)
            return img

    def __data_generation(self, index):
        """
        Generates data containing batch_size samples.
        """
        X = np.zeros((128, 256, 12), dtype='float32')
        y = np.zeros(6, dtype='float32')
        img = np.ones((128,256), dtype='float32')
        row = self.df.iloc[index]
        if self.mode=='test': 
            r = 0
        else: 
            r = int((row['min'] + row['max']) // 4)
            
        for region in range(4):
            img = self.spectrograms[row.spectrogram_id][r:r+300, region*100:(region+1)*100].T
            img = self.log_and_Standarize(img)
            X[14:-14, :, region] = img[:, 22:-22] / 2.0
            
        img = self.eeg_spectrograms[row.eeg_id]
        img = img.to_numpy()
        img = self.log_and_Standarize(img)
        img = resize(img, (128, 256, 4))
        X[:, :, 4:8] = img

        # Combine wavelet features
        img = self.wavelets_spectrograms[row.spectrogram_id]
        img = self.log_and_Standarize(img)
        img = resize(img, (128, 256,4))
        X[:, :, 8:12] = img


        if self.mode != 'test':
            y = row[label_cols].values.astype(np.float32)
    
        return X, y
    
    def __transform(self, img):
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
        ])
        return transforms(image=img)['image']


class CustomModel(nn.Module):
    def __init__(self, config, num_classes: int = 6, pretrained: bool = True):
        super(CustomModel, self).__init__()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = False 
        self.USE_WAVELET_SPECTROGRAMS = False
        self.model = timm.create_model(
            config.MODEL,
            pretrained=pretrained,
            drop_rate = 0.1,
            drop_path_rate = 0.2,
        )
        # add code on logging parameter
        logging.info("config.MODEL: {}".format(config.MODEL))
        logging.info("USE_KAGGLE_SPECTROGRAMS: {}".format(self.USE_KAGGLE_SPECTROGRAMS))
        logging.info("USE_EEG_SPECTROGRAMS: {}".format(self.USE_EEG_SPECTROGRAMS))
        logging.info("USE_WAVELET_SPECTROGRAMS: {}".format(self.USE_WAVELET_SPECTROGRAMS))

        if config.FREEZE:
            for i,(name, param) in enumerate(list(self.model.named_parameters())\
                                             [0:config.NUM_FROZEN_LAYERS]):
                param.requires_grad = False

        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes)
        )

    def __reshape_input(self, x):
        """
        Reshapes input torch.Size([8, 128, 256, 12]) -> [8, 3, 512, 768] monotone image.
        """ 
        components = []
        if self.USE_KAGGLE_SPECTROGRAMS:
            spectograms = [x[:, :, :, i:i+1] for i in range(4)]
            components.append(torch.cat(spectograms, dim=1))
        if self.USE_EEG_SPECTROGRAMS:
            eegs = [x[:, :, :, i:i+1] for i in range(4,8)]
            eegs = torch.cat(eegs, dim=1)
            components.append(eegs)

        if self.USE_WAVELET_SPECTROGRAMS:
            wavelets = [x[:, :, :, i:i+1] for i in range(8,12)]
            wavelets = torch.cat(wavelets, dim=1)
            components.append(wavelets)

        if components:
            x = torch.cat(components, dim=2)

        x = torch.cat([x, x, x], dim=3)  
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        x = self.__reshape_input(x)
        x = self.features(x)
        x = self.custom_layers(x)
        return x


@log_time
def train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    """One epoch training pass."""
    model.train() 
    criterion = nn.KLDivLoss(reduction="batchmean")
    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    
    # ========== ITERATE OVER TRAIN BATCHES ============
    with tqdm(train_loader, unit="train_batch", desc='Train') as tqdm_train_loader:
        for step, (X, y) in enumerate(tqdm_train_loader):
            X = X.to(device)
            y = y.to(device)
            batch_size = y.size(0)
            with torch.cuda.amp.autocast(enabled=config.AMP):
                y_preds = model(X) 
                loss = criterion(F.log_softmax(y_preds, dim=1), y)
            if config.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

            if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                scheduler.step()
            end = time.time()

            # ========== LOG INFO ==========
            if step % config.PRINT_FREQ == 0 or step == (len(train_loader)-1):
                print('Epoch: [{0}][{1}/{2}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.avg:.4f} '
                      'Grad: {grad_norm:.4f}  '
                      'LR: {lr:.8f}  '
                      .format(epoch+1, step, len(train_loader), 
                              remain=timeSince(start, float(step+1)/len(train_loader)),
                              loss=losses,
                              grad_norm=grad_norm,
                              lr=scheduler.get_last_lr()[0]))

    return losses.avg

@log_time
def valid_epoch(valid_loader, model, criterion, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    losses = AverageMeter()
    prediction_dict = {}
    preds = []
    start = end = time.time()
    with tqdm(valid_loader, unit="valid_batch", desc='Validation') as tqdm_valid_loader:
        for step, (X, y) in enumerate(tqdm_valid_loader):
            X = X.to(device)
            y = y.to(device)
            batch_size = y.size(0)
            with torch.no_grad():
                y_preds = model(X)
                loss = criterion(F.log_softmax(y_preds, dim=1), y)
            if config.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            losses.update(loss.item(), batch_size)
            y_preds = softmax(y_preds)
            preds.append(y_preds.to('cpu').numpy())
            end = time.time()

            # ========== LOG INFO ==========
            if step % config.PRINT_FREQ == 0 or step == (len(valid_loader)-1):
                print('EVAL: [{0}/{1}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.avg:.4f} '
                      .format(step, len(valid_loader),
                              remain=timeSince(start, float(step+1)/len(valid_loader)),
                              loss=losses))
                
    prediction_dict["predictions"] = np.concatenate(preds)
    return losses.avg, prediction_dict

@log_time
def train_loop(df, fold):
    
    logging.info(f"========== Fold: {fold} training ==========")

    # ======== SPLIT ==========
    train_folds = df[df['fold'] != fold].reset_index(drop=True)
    valid_folds = df[df['fold'] == fold].reset_index(drop=True)
    
    # ======== DATASETS ==========
    train_dataset = CustomDataset(train_folds, config, mode="train", augment=True, specs=all_spectrograms, eeg_specs=all_eegs,wavelets_spectrograms = all_wavelet_spectrograms )
    valid_dataset = CustomDataset(valid_folds, config, mode="train", augment=False, specs=all_spectrograms, eeg_specs=all_eegs,wavelets_spectrograms = all_wavelet_spectrograms)
    
    # ======== DATALOADERS ==========
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE_TRAIN,
                              shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.BATCH_SIZE_VALID,
                              shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False)
    
    # ======== MODEL ==========
    model = CustomModel(config)
    model.to(device)

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=config.WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=100,
    )

    # ======= LOSS ==========
    criterion = nn.KLDivLoss(reduction="batchmean")
    
    best_loss = np.inf
    # ====== ITERATE EPOCHS ========
    for epoch in range(config.EPOCHS):
        start_time = time.time()

        # ======= TRAIN ==========
        avg_train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # ======= EVALUATION ==========
        avg_val_loss, prediction_dict = valid_epoch(valid_loader, model, criterion, device)
        predictions = prediction_dict["predictions"]
        
        # ======= SCORING ==========
        elapsed = time.time() - start_time

        logging.info(f'Epoch {epoch+1} - avg_train_loss: {avg_train_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            logging.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                        paths.OUTPUT_DIR / f"{config.MODEL.replace('/', '_')}_fold_{fold}_best.pth")

    ## TypeError: unsupported operand type(s) for +: 'WindowsPath' and 'str'
    # predictions = torch.load(paths.OUTPUT_DIR + f"/{config.MODEL.replace('/', '_')}_fold_{fold}_best.pth", 
    #                          map_location=torch.device('cpu'))['predictions']
    paths.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = torch.load(paths.OUTPUT_DIR / f"{config.MODEL.replace('/', '_')}_fold_{fold}_best.pth",
                         map_location=torch.device('cpu'))['predictions']

    valid_folds[target_preds] = predictions

    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds

@log_time
def train_loop_full_data(df):
    train_dataset = CustomDataset(df, config, mode="train", augment=True,specs=all_spectrograms, eeg_specs=all_eegs,wavelets_spectrograms = all_wavelet_spectrograms)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE_TRAIN,
                              shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    model = CustomModel(config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=config.WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=100,
    )
    criterion = nn.KLDivLoss(reduction="batchmean")
    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        start_time = time.time()
        avg_train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, device)
        elapsed = time.time() - start_time
        logging.info(f'Epoch {epoch+1} - avg_train_loss: {avg_train_loss:.4f}  time: {elapsed:.0f}s')
        torch.save(
            {'model': model.state_dict()},
            paths.OUTPUT_DIR + f"/{config.MODEL.replace('/', '_')}_epoch_{epoch}.pth")
    torch.cuda.empty_cache()
    gc.collect()
    return 

@log_time
def get_result(oof_df):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    labels = torch.tensor(oof_df[label_cols].values)
    preds = torch.tensor(oof_df[target_preds].values)
    preds = F.log_softmax(preds, dim=1)
    result = kl_loss(preds, labels)
    return result

@log_time
def preparing_data(df):
    train_df = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg({
        'spectrogram_id':'first',
        'spectrogram_label_offset_seconds':'min'
    })
    train_df.columns = ['spectrogram_id','min']

    aux = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg({
        'spectrogram_label_offset_seconds':'max'
    })
    train_df['max'] = aux

    aux = df.groupby('eeg_id')[['patient_id']].agg('first')
    train_df['patient_id'] = aux

    aux = df.groupby('eeg_id')[label_cols].agg('sum')
    for label in label_cols:
        train_df[label] = aux[label].values
        
    y_data = train_df[label_cols].values
    y_data = y_data / y_data.sum(axis=1,keepdims=True)
    train_df[label_cols] = y_data

    aux = df.groupby('eeg_id')[['expert_consensus']].agg('first')
    train_df['target'] = aux

    train_df = train_df.reset_index()
    return train_df

def compute_wavelet_features(signal, wavelet='db4', level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # 从小波系数中提取特征而不是直接用小波系数，因为有不规则大小。
    features = []
    for coeff in coeffs:
        features.extend([np.mean(coeff), np.std(coeff)])
    return np.array(features)


@log_time
def loading_parquet(train_df, config = config, READ_SPEC_FILES = True,READ_EEG_SPEC_FILES = True,wavelet = 'None'):
    # paths_spectograms = glob(paths.TRAIN_SPECTOGRAMS + "*.parquet")
    paths_spectrograms = glob(str(paths.TRAIN_SPECTROGRAMS / "*.parquet"))
    print(f'There are {len(paths_spectrograms)} spectrogram parquets in total path')

    if READ_SPEC_FILES:    
        all_spectrograms = {}
        all_wavelet_spectrograms = {}
        spectogram_ids = train_df['spectrogram_id'].unique()
        print(f'There are {len(spectogram_ids)} spectrogram parquets in this training process')
        for spec_id in tqdm(spectogram_ids):
        # for file_path in tqdm(paths_spectograms):
            file_path = f"{paths.TRAIN_SPECTROGRAMS}/{spec_id}.parquet"
            aux = pd.read_parquet(file_path)
            spec_arr = aux.fillna(0).values[:, 1:].T.astype("float32")  # (Hz, Time) = (400, 300)
            wavelet_features = np.array([compute_wavelet_features(row, wavelet=wavelet) for row in spec_arr])
            name = int(file_path.split("/")[-1].split('.')[0])
            # all_spectrograms[name] = aux.iloc[:,1:].values  
            all_spectrograms[name] = aux.fillna(0).iloc[:,1:].values.astype("float32")
            all_wavelet_spectrograms[name] = wavelet_features
            del aux
            del wavelet_features
        os.makedirs(os.path.dirname(paths.PRE_LOADED_SPECTROGRAMS), exist_ok=True)
        os.makedirs(os.path.dirname(paths.PRE_LOADED_Wavelets), exist_ok=True)
        np.save(paths.PRE_LOADED_SPECTROGRAMS, all_spectrograms, allow_pickle=True)
        np.save(paths.PRE_LOADED_Wavelets, all_wavelet_spectrograms, allow_pickle=True)
    else:
        all_spectrograms = np.load(paths.PRE_LOADED_SPECTROGRAMS, allow_pickle=True).item()
        all_wavelet_spectrograms = np.load(paths.PRE_LOADED_Wavelets, allow_pickle=True).item()
        
    if config.VISUALIZE:
        idx = np.random.randint(0,len(paths_spectrograms))
        spectrogram_path = paths_spectrograms[idx]
        plot_spectrogram(spectrogram_path)

    # Read EEG Spectrograms
    # paths_eegs = glob(paths.TRAIN_EEGS + "*.parquet")
    paths_eegs = glob(str(paths.TRAIN_EEGS / "*.parquet"))
    print(f'There are {len(paths_eegs)} EEG spectrograms in total path')
    if READ_EEG_SPEC_FILES:
        all_eegs = {}
        eeg_ids = train_df['eeg_id'].unique()
        print(f'There are {len(eeg_ids)} EEG spectrograms in this training path')
        for eeg_id in tqdm(eeg_ids):
            file_path = f"{paths.TRAIN_EEGS}/{eeg_id}.parquet"
            eeg_spectogram =  pd.read_parquet(file_path)
            all_eegs[eeg_id] = eeg_spectogram
            del eeg_spectogram
        os.makedirs(os.path.dirname(paths.PRE_LOADED_EEGS), exist_ok=True)
        np.save(paths.PRE_LOADED_EEGS, all_eegs, allow_pickle=True)
    else:
        all_eegs = np.load(paths.PRE_LOADED_EEGS, allow_pickle=True).item()



    
    return all_spectrograms,all_eegs,all_wavelet_spectrograms


if __name__ == "__main__":
    overall_start_time = time.time()
    print(f"Log file path: {log_filename.absolute()}")
    logging.info('--------------------------------------------------')
    logging.info(f'training on local balanced data')
    logging.info(f'Into loading stage')

    target_preds = [x + "_pred" for x in ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']]
    label_to_num = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other':5}
    num_to_label = {v: k for k, v in label_to_num.items()}
    seed_everything(config.SEED)

    df = pd.read_csv(paths.TRAIN_CSV)
    label_cols = df.columns[-6:]
    print(f"Train cataframe shape is: {df.shape}")
    print(f"Labels: {list(label_cols)}")
    print(df.head())

    #处理train_df，eeg_id,只保留第一个spectrogram_id，min及max spec offset，第一个patient_id等
    train_df = preparing_data(df)
    print('Train non-overlapp eeg_id shape:', train_df.shape )
    print(train_df.head())
    train_df.to_csv('./local_train_df.csv', index=False)

    logging.info(f'Into loading stage: combine wavelet feature into X')
    logging.info(f'Into loading stage: loading single npy from local file')
    all_spectrograms,all_eegs,all_wavelet_spectrograms = loading_parquet(train_df, config = config, READ_SPEC_FILES = False,READ_EEG_SPEC_FILES = False,wavelet='db1')
    

    # Validation 
    gkf = GroupKFold(n_splits=config.FOLDS)
    for fold, (train_index, valid_index) in enumerate(gkf.split(train_df, train_df.target, train_df.patient_id)):
        train_df.loc[valid_index, "fold"] = int(fold)
        
    print(train_df.groupby('fold').size()), sep()
    print(train_df.head())

    train_dataset = CustomDataset(train_df, config, mode="train", 
                                  specs=all_spectrograms, eeg_specs=all_eegs,wavelets_spectrograms = all_wavelet_spectrograms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE_TRAIN,
        shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    X, y = train_dataset[0]
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    if config.VISUALIZE:
        ROWS = 2
        COLS = 3
        for (X, y) in train_loader:
            plt.figure(figsize=(20,8))
            for row in range(ROWS):
                for col in range(COLS):
                    plt.subplot(ROWS, COLS, row*COLS + col+1)
                    t = y[row*COLS + col]
                    img = X[row*COLS + col, :, :, 0]
                    mn = img.flatten().min()
                    mx = img.flatten().max()
                    img = (img-mn)/(mx-mn)
                    plt.imshow(img)
                    tars = f'[{t[0]:0.2f}'
                    for s in t[1:]:
                        tars += f', {s:0.2f}'
                    eeg = train_df.eeg_id.values[row*config.BATCH_SIZE_TRAIN + row*COLS + col]
                    plt.title(f'EEG = {eeg}\nTarget = {tars}',size=12)
                    plt.yticks([])
                    plt.ylabel('Frequencies (Hz)',size=14)
                    plt.xlabel('Time (sec)',size=16)
            plt.show()
            break

    # #dynamic learning rate
    # EPOCHS = config.EPOCHS
    # BATCHES = len(train_loader)
    # steps = []
    # lrs = []
    # optim_lrs = []
    # model = CustomModel(config)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # scheduler = OneCycleLR(
    #     optimizer,
    #     max_lr=1e-3,
    #     epochs=config.EPOCHS,
    #     steps_per_epoch=len(train_loader),
    #     pct_start=0.05,
    #     anneal_strategy="cos",
    #     final_div_factor=100,
    # )
    # for epoch in range(EPOCHS):
    #     for batch in range(BATCHES):
    #         scheduler.step()
    #         lrs.append(scheduler.get_last_lr()[0])
    #         steps.append(epoch * BATCHES + batch)

    # max_lr = max(lrs)
    # min_lr = min(lrs)
    # print(f"Maximum LR: {max_lr} | Minimum LR: {min_lr}")
    # plt.figure()
    # plt.plot(steps, lrs, label='OneCycle')
    # plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    # plt.xlabel("Step")
    # plt.ylabel("Learning Rate")
    # plt.show()


    if not config.TRAIN_FULL_DATA:
        oof_df = pd.DataFrame()
        for fold in range(config.FOLDS):
            _oof_df = train_loop(train_df, fold,)
            oof_df = pd.concat([oof_df, _oof_df])
            logging.info(f"========== Fold {fold} result: {get_result(_oof_df)} ==========")
            print(f"========== Fold {fold} result: {get_result(_oof_df)} ==========")
        oof_df = oof_df.reset_index(drop=True)
        logging.info(f"========== CV: {get_result(oof_df)} ==========")
        logging.info(f"----------------------------------------------------------------------------------")
        # oof_df.to_csv(paths.OUTPUT_DIR + '/oof_df.csv', index=False)
        oof_df.to_csv(os.path.join(paths.OUTPUT_DIR, 'oof_df.csv'), index=False)
    else:
        train_loop_full_data(train_df)