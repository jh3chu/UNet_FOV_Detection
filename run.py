import os
import numpy as np
import torch
import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import GeneralizedDiceLoss

from utils.utils import (
    copy_files, 
    preprocess, 
    get_loaders, 
    normalize_preprocess,
)
from model.tensorboard_utils import(
    image_grid,
)

from utils.train import train_fn
import model.hyperparameters as hp

# Directories
TRAIN_IMG_DIR = r'D:\Jonathan\2_Projects\Fov_Detection\FoV_Data\train_img\T1'
TRAIN_SEG_DIR = r'D:\Jonathan\2_Projects\Fov_Detection\FoV_Data\train_seg\T1'
VAL_IMG_DIR = r'D:\Jonathan\2_Projects\Fov_Detection\FoV_Data\val_img\T1'
VAL_SEG_DIR = r'D:\Jonathan\2_Projects\Fov_Detection\FoV_Data\val_seg\T1'

# Using val and test
# TRAIN_IMG_DIR = r'D:\Jonathan\2_Projects\Fov_Detection\FoV_Data\val_img\T1'
# TRAIN_SEG_DIR = r'D:\Jonathan\2_Projects\Fov_Detection\FoV_Data\val_seg\T1'
# VAL_IMG_DIR = r'D:\Jonathan\2_Projects\Fov_Detection\FoV_Data\test_img\T1'
# VAL_SEG_DIR = r'D:\Jonathan\2_Projects\Fov_Detection\FoV_Data\test_seg\T1'

MODEL_SAVE_PATH = 'test'

COPY_FILES = False
NORMALIZE = False
SHOW_PATIENT = False
LOAD_MODEL = False



def main():
    if NORMALIZE:
        normalize_preprocess(TRAIN_IMG_DIR, VAL_IMG_DIR)

    train_file_paths, val_file_paths = preprocess(
        os.path.join(TRAIN_IMG_DIR, 'Normalized'), 
        TRAIN_SEG_DIR, 
        os.path.join(VAL_IMG_DIR, 'Normalized'),
        VAL_SEG_DIR
    )
     
    train_loader, val_loader = get_loaders(
        train_files=train_file_paths, 
        val_files=val_file_paths, 
        batch_size=hp.BATCH_SIZE, 
        num_workers=hp.NUM_WORKERS)

    logdir = 'logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(logdir)
    writer = SummaryWriter(logdir, filename_suffix='_lr_{}_epoch{}'.format(hp.LEARNING_RATE, hp.NUM_EPOCH))
    
    # show_patient(train_loader)

    # if SHOW_PATIENT:
    #     image_grid(train_loader, TB_DIR)

    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        act='tanh',
        num_res_units=2, # residual unet
        norm=Norm.BATCH, # use batch normalization
        dropout=0.1 
    ).to(hp.DEVICE)

    # loss_fn = DiceLoss(
    #     # to_onehot_y=True, 
    #     # softmax=True, 
    #     # squared_pred=True
    # )
    loss_fn = GeneralizedDiceLoss(
        # to_onehot_y=True,
        )

    optimizer = optim.Adam(model.parameters(), lr=hp.LEARNING_RATE)

    if LOAD_MODEL:
        train_fn(
        train_loader=train_loader, 
        val_loader=val_loader, 
        model=model, 
        optimizer=optimizer, 
        loss_fn=loss_fn, 
        num_epoch=hp.NUM_EPOCH, 
        device=hp.DEVICE,
        model_save_path=MODEL_SAVE_PATH,
        writer=writer,
        val_interval=1,
        checkpoint=MODEL_SAVE_PATH
        )    
    else:
        train_fn(
            train_loader=train_loader, 
            val_loader=val_loader, 
            model=model, 
            optimizer=optimizer, 
            loss_fn=loss_fn, 
            num_epoch=hp.NUM_EPOCH, 
            device=hp.DEVICE,
            model_save_path=MODEL_SAVE_PATH,
            writer=writer,
            val_interval=1,
            checkpoint=None
        )    
        

if __name__ == '__main__':
    if COPY_FILES:
        copy_files()
    main()
