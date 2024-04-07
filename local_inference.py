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
    BATCH_SIZE = 8
    EPOCHS = 4
    FOLDS = 5
    FREEZE = False
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1e7
    MODEL = "efficientnet_b4"
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
    TEST_CSV = DATA / "test.csv"
    TEST_SPECTROGRAMS = DATA / "test_spectrograms"
    TEST_EEGS = DATA / "test_eegs"

log_filename = paths.ROOT/'new_version_inference_record.log'

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
            spectrogram_id_int = int(row.spectrogram_id)  # numpy.int64 → Python int
            img = self.spectrograms[spectrogram_id_int][r:r+300, region*100:(region+1)*100].T
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
        self.USE_EEG_SPECTROGRAMS = True
        self.USE_WAVELET_SPECTROGRAMS = True
        self.model = timm.create_model(
            config.MODEL,
            pretrained=pretrained,
            drop_rate = 0.1,
            drop_path_rate = 0.2,
        )
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
def get_result(oof_df):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    labels = torch.tensor(oof_df[label_cols].values)
    preds = torch.tensor(oof_df[target_preds].values)
    preds = F.log_softmax(preds, dim=1)
    result = kl_loss(preds, labels)
    return result



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
    paths_spectrograms = glob(str(paths.TEST_SPECTROGRAMS / "*.parquet"))
    print(f'There are {len(paths_spectrograms)} spectrogram parquets in total path')

        
    all_spectrograms = {}
    all_wavelet_spectrograms = {}
    spectogram_ids = train_df['spectrogram_id'].unique()
    print(f'There are {len(spectogram_ids)} spectrogram parquets in this training process')
    for spec_id in tqdm(spectogram_ids):
    # for file_path in tqdm(paths_spectograms):
        file_path = f"{paths.TEST_SPECTROGRAMS}/{spec_id}.parquet"
        aux = pd.read_parquet(file_path)
        spec_arr = aux.fillna(0).values[:, 1:].T.astype("float32")  # (Hz, Time) = (400, 300)
        wavelet_features = np.array([compute_wavelet_features(row, wavelet=wavelet) for row in spec_arr])
        name = int(file_path.split("/")[-1].split('.')[0])
        # all_spectrograms[name] = aux.iloc[:,1:].values  
        all_spectrograms[name] = aux.fillna(0).iloc[:,1:].values.astype("float32")
        all_wavelet_spectrograms[name] = wavelet_features
        del aux
        del wavelet_features

        
    if config.VISUALIZE:
        idx = np.random.randint(0,len(paths_spectrograms))
        spectrogram_path = paths_spectrograms[idx]
        plot_spectrogram(spectrogram_path)

    # Read EEG Spectrograms
    # paths_eegs = glob(paths.TRAIN_EEGS + "*.parquet")
    paths_eegs = glob(str(paths.TEST_EEGS / "*.parquet"))
    print(f'There are {len(paths_eegs)} EEG spectrograms in total path')
    all_eegs = {}
    eeg_ids = train_df['eeg_id'].unique()
    print(f'There are {len(eeg_ids)} EEG spectrograms in this training path')
    for eeg_id in tqdm(eeg_ids):
        file_path = f"{paths.TEST_EEGS}/{eeg_id}.parquet"
        eeg_spectogram =  pd.read_parquet(file_path)
        all_eegs[eeg_id] = eeg_spectogram
        del eeg_spectogram

    return all_spectrograms,all_eegs,all_wavelet_spectrograms


def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    prediction_dict = {}
    preds = []
    with tqdm(test_loader, unit="test_batch", desc='Inference') as tqdm_test_loader:
        for step, (X, y) in enumerate(tqdm_test_loader):
            X = X.to(device)
            y = y.to(device)
            batch_size = y.size(0)
            with torch.no_grad():
                y_preds = model(X)
            y_preds = softmax(y_preds)
            preds.append(y_preds.to('cpu').numpy()) 
                
    prediction_dict["predictions"] = np.concatenate(preds) 
    return prediction_dict




def maddest(d, axis: int = None):
    """
    Denoise function.
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise(x: np.ndarray, wavelet: str = 'haar', level: int = 1): 
    coeff = pywt.wavedec(x, wavelet, mode="per") # multilevel 1D Discrete Wavelet Transform of data.
    sigma = (1/0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    output = pywt.waverec(coeff, wavelet, mode='per')
    return output


if __name__ == "__main__":
    overall_start_time = time.time()
    print(f"Log file path: {log_filename.absolute()}")
    logging.info('--------------------------------------------------')
    logging.info(f'Into loading stage')

    target_preds = [x + "_pred" for x in ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']]
    label_to_num = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other':5}
    num_to_label = {v: k for k, v in label_to_num.items()}
    seed_everything(config.SEED)

    test_df = pd.read_csv(paths.TEST_CSV)
    label_cols = test_df.columns[-6:]
    print(test_df.head())

    logging.info(f'Into loading stage: combine wavelet feature into X')
    all_spectrograms,all_eegs,all_wavelet_spectrograms = loading_parquet(test_df, config = config, READ_SPEC_FILES = True,READ_EEG_SPEC_FILES = True,wavelet='db1')
    print('all_spectrograms:',all_spectrograms)
    for key in list(all_spectrograms.keys())[:5]:  # 打印前5个键的类型
        print('type(key):',type(key))

    print('all_eegs:',all_eegs)
    print('all_wavelet_spectrograms:',all_wavelet_spectrograms)
    test_dataset = CustomDataset(test_df, config, mode="test",specs=all_spectrograms, eeg_specs=all_eegs,wavelets_spectrograms = all_wavelet_spectrograms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False
    )
    X, y = test_dataset[0]
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    

    model_weights = [x for x in glob(str(paths.OUTPUT_DIR/"*.pth"))]

    # model_weights = [x for x in glob("/kaggle/input/hms-efficientnetb0-5-folds/*.pth")]
    predictions = list()
    for model_weight in model_weights:
        test_dataset = CustomDataset(test_df, config, mode="test", augment=False)
        train_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True, drop_last=False
        )
        model = CustomModel(config)
        checkpoint = torch.load(model_weight)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        prediction_dict = inference_function(test_loader, model, device)
        predictions.append(prediction_dict["predictions"])
        torch.cuda.empty_cache()
        gc.collect()
        
    predictions = np.array(predictions)
    predictions = np.mean(predictions, axis=0)

    TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    sub = pd.DataFrame({'eeg_id': test_df.eeg_id.values})
    sub[TARGETS] = predictions
    sub.to_csv('submission.csv',index=False)
    print(f'Submissionn shape: {sub.shape}')
    print(sub.head())