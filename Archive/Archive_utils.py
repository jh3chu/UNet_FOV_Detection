import numpy as np
import os
import re
import pandas as pd
import shutil 
import torch
import torchvision

from dataset import SpineGenericDataset
from torch.utils.data import DataLoader

def copy_files(
    root_dir=r'D:\Jonathan\2_Projects\Fov_Detection\SpineGen_T1-T2-results\data_processed', 
    fn_keys=('img', 'seg'), 
    p1 = r'w.nii.gz$', 
    p2='_RPI_r_seg.nii.gz$',
    new_dir = r'D:\Jonathan\2_Projects\Fov_Detection\FoV_Data',
):

    img_filenames = []
    filenames = []

    regex1 = p1

    for root, dir_names, fn in os.walk(root_dir): # top down by name
        for f in fn:
            if re.search(regex1, f):
                img_fname = os.path.join(root, f)
                img_filenames.append(img_fname)

    # Loop through T1 and T2 images for labelled segmentations
    for i in img_filenames:
        regex2 = i.split('\\')[-1].split('.')[0].split('_')[1]
        
        # Read labelled data
        # regex2 = str(regex2 + '_RPI_r_seg_labeled.nii.gz$')

        # Read segmentation data
        regex2 = str(regex2 + p2)

        # Check for labelled segmentation of T1 or T2 img
        for root, dir_names, fn in os.walk(i.rsplit('\\', 1)[0]):
            for f in fn:
                if re.search(regex2, f):
                    label_fname = os.path.join(root, f)
                    filenames.append({'img': i, 'seg': label_fname})

    # Copy img to new folders
    for idx in range(len(filenames)):
        path = filenames[idx]['img']
        fn = path.split('\\')[-1]
        new_path = os.path.join(new_dir, 'train_img', fn)
        if os.path.exists(new_path) == False:
            try:
                shutil.copy(path, new_path)

            # If source and destination are same
            except shutil.SameFileError:
                print("Source and destination represents the same file.")
            
            # If there is any permission issue
            except PermissionError:
                print("Permission denied.")
            
            # For other errors
            except:
                print("Error occurred while copying file.")
    # Copy seg new folders
    for idx in range(len(filenames)):
        path = filenames[idx]['seg']
        fn = path.split('\\')[-1]
        new_path = os.path.join(new_dir, 'train_seg', fn)
        if os.path.exists(new_path) == False:
            try:
                shutil.copy(path, new_path)

            # If source and destination are same
            except shutil.SameFileError:
                print("Source and destination represents the same file.")
            
            # If there is any permission issue
            except PermissionError:
                print("Permission denied.")
            
            # For other errors
            except:
                print("Error occurred while copying file.")

    print('MOVE FILES TO VALIDATION FOLDER')

def preprocess():
    pass

def transform():
    pass

def get_loaders(
    train_dir,
    train_segdir,
    val_dir,
    val_segdir,
    batch_size,
    #train_transform,
    #val_transform,
    num_workers=4,
    pin_memory=True,
):

    # Create image training dataset
    train_ds = SpineGenericDataset(
        img_dir=train_dir,
        seg_dir=train_segdir,
        # transform=train_transform,
    )

    # Load training dataset
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = SpineGenericDataset(
        img_dir=val_dir,
        seg_dir=val_segdir,
        # transform=val_transform,
    )

    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader