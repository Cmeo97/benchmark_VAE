from torchvision import transforms, datasets
import os
import zipfile
import pandas as pd
import gdown
from natsort import natsorted
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

## Root directory for the dataset
data_root = '/home/cristianmeo/Datasets/celeba'
## Path to folder with individual images
img_folder = f'{data_root}/img_align_celeba'
## URL for the CelebA dataset
##url = 'https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=share_link&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ'
## Path to download the dataset to
#download_path = f'{data_root}/img_align_celeba.zip'
## Create required directories 
#if not os.path.exists(data_root):
#  os.makedirs(data_root)
#  os.makedirs(img_folder)
#
## Download the dataset from google drive
##gdown.download(url, download_path, quiet=False)
#
## Unzip the downloaded file 
#with zipfile.ZipFile(download_path, 'r') as ziphandler:
#  ziphandler.extractall(img_folder)

X = pd.read_csv(img_folder+'/list_eval_partition.txt', sep="\t", header=None).to_numpy()
list_ids = np.zeros((202599), int)
for i in range(202599):
    list_ids[i] = int(str(X[i])[13])



