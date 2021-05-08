import pandas as pd
import os
import shutil
import re
from tqdm.notebook import tqdm
from pathlib import Path
from sklearn import preprocessing, model_selection
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import wget
import tarfile

pd.set_option("display.max_colwidth", None, "display.max_row", None)
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def download_pets(root_dir = '../data/oxford_iiit_pet'):
    
    data_url = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz'
    annotations_url = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz'
        
    #download dataset
    if not os.path.exists(root_dir + '/images'):
        filename = wget.download(data_url, out=root_dir)
        
        tar = tarfile.open(root_dir + '/images.tar.gz', "r:gz")
        tar.extractall(root_dir)
        tar.close()
    
    #download annotations
    if not os.path.exists(root_dir + '/annotations'):
        filename = wget.download(annotations_url, out=root_dir)
        
        tar = tarfile.open(root_dir + '/annotations.tar.gz', "r:gz")
        tar.extractall(root_dir)
        tar.close()
        


def transforms(trn:bool=False):
    h = 32 #@param{type:"integer"}
    w = 32 #@param{type:"integer"}
    
    if trn: 
        tfms = [A.CLAHE(), A.IAAPerspective(), A.IAASharpen(), A.RandomBrightness(),
                A.Rotate(limit=60), A.HorizontalFlip()]
    else: tfms = []
    tfms.append(A.Resize(h,w, always_apply=True))
    tfms.append(A.Normalize(always_apply=True))
    tfms.append(ToTensorV2(always_apply=True))
    tfs = A.Compose(tfms)
    return tfs


class ParseData(Dataset):
    def __init__(self, pth, tfms_fn):
        self.df = pd.read_csv(pth)
        self.tfms = tfms_fn
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        try: pth = self.df.fnames[idx]
        except Exception: print(idx)
        im = cv2.cvtColor(cv2.imread(pth), cv2.COLOR_BGR2RGB)
        im = self.tfms(image=im)["image"]
        lbl = self.df.targets[idx]
        return im, lbl
    
def get_PETS_data(root_dir ='../data/oxford_iiit_pet',
                  batch_size = 16,
                 test_size = 0.20,
                val_size = 0.20):
    
    download_pets(root_dir)
    
    pat = r'/([^/]+)_\d+.jpg$'
    pat = re.compile(pat)
    
    #collect list of images
    desc = Path(root_dir + "/images")
    ims = list(desc.iterdir())
    im_list = []

    for im in ims:
        if str(im).split(os.path.sep)[-1].split(".")[-1] == "jpg": im_list.append(str(im))
    
    #check for and remove corrupted images
    print('checking for corrupted images')
    for im in tqdm(im_list):
        try: _ = cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB)
        except:
            im_list.remove(im)
            print(f"[INFO] Corrupted Image: {im}")
        
    df = pd.DataFrame()
    df["fnames"] = im_list
    df["labels"] = [ pat.search(fname).group(1).lower() for fname in df.fnames]
    df["targets"] = preprocessing.LabelEncoder().fit_transform(df.labels.values)
    df = df.sample(frac=1).reset_index(drop=True)
    
    y = df.labels.values
    
    #train, val, test split 
    X_train, X_test, y_train, y_test = model_selection.train_test_split(df, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=val_size / (1-test_size), random_state=42)

    X_train.to_csv(root_dir + '/train_images.csv', index = False)
    X_val.to_csv(root_dir + '/val_images.csv', index = False)
    X_test.to_csv(root_dir + '/test_images.csv', index = False)
    
    train_loader = DataLoader(ParseData(root_dir + '/train_images.csv', transforms(True)),batch_size=batch_size,shuffle=True, pin_memory=True)
    val_loader = DataLoader(ParseData(root_dir + '/val_images.csv', transforms(False)),batch_size=batch_size,shuffle=False, pin_memory=True)
    test_loader = DataLoader(ParseData(root_dir + '/test_images.csv', transforms(False)),batch_size=batch_size,shuffle=False, pin_memory=True)
    
    return train_loader, val_loader, test_loader