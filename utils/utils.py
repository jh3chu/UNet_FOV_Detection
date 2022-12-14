import os
import re
import numpy as np
import pandas as pd
import shutil 
import torch
import torchvision
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from glob import glob

from monai.transforms import (
    Compose,
    AddChanneld,
    EnsureChannelFirstd,
    LoadImaged,
    Resized,
    DivisiblePadd,
    ToTensord,
    RandFlipd,
    RandRotated,
)
from monai.utils import first
from monai.data import DataLoader, Dataset
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

# Copy files to new directory in sorted modality sorted folders
def copy_files(
    root_dir=r'D:\Jonathan\2_Projects\Fov_Detection\SpineGen_T1-T2-results\data_processed', 
    fn_keys=('img', 'seg'), 
    p1 = '_RPI_r.nii.gz', 
    p2='_RPI_r_seg.nii.gz$',
    new_dir = r'D:\Jonathan\2_Projects\Fov_Detection\FoV_Data',
):
    '''
    NEED TO MANUAL SEPARATE DATA INTO T1/T2 FOLDERS AND MOVE VALIDATION SET
    '''
    if not os.path.exists(os.path.join(new_dir, 'train_img')):
        os.makedirs(os.path.join(new_dir, 'train_img'))
    if not os.path.exists(os.path.join(new_dir, 'train_seg')):
        os.makedirs(os.path.join(new_dir, 'train_seg'))
    if not os.path.exists(os.path.join(new_dir, 'val_img')):
        os.makedirs(os.path.join(new_dir, 'val_img'))
    if not os.path.exists(os.path.join(new_dir, 'val_seg')):
        os.makedirs(os.path.join(new_dir, 'val_seg'))

    img_filenames = []
    filenames = []

    regex1 = p1

    # Get T1 and T2 images
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

        # Check for labelled segmentation of T1 or T2 img and return img/seg dictionary pair
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

    print('MOVED FILES TO {}'.format(new_dir))

def normalize_preprocess(
    train_root_dir, 
    val_root_dir,
):
    '''
    Manual normalization preprocessing
    :params: 
        train_root_dir: root dir of training images for saving
        val_root_dir: root dir of validation images for saving
    :outputs:
    '''
    # Normalize training set
    train_file_paths = [x for x in os.listdir(train_root_dir) if x not in ['Normalized', 'Archive']]
    train_norm_dir = os.path.join(train_root_dir, 'Normalized')
    if not os.path.exists(train_norm_dir):
        os.makedirs(train_norm_dir)
        

    for i in range(len(train_file_paths)):
        train_img = nib.load(os.path.join(train_root_dir, train_file_paths[i]))

        train_image_data = train_img.get_fdata()
        
        train_image_min = np.min(train_image_data)
        train_image_max = np.max(train_image_data)

        train_image_norm = 2 * ((train_image_data - train_image_min) / (train_image_max - train_image_min)) - 1

        if i == 0:
            plt.figure(figsize=(15, 15))
            plt.subplot(321)
            plt.imshow(train_image_data[100, :, :], cmap='gray')
            plt.subplot(323)
            plt.imshow(train_image_data[:, 100, :], cmap='gray')
            plt.subplot(325)
            plt.imshow(train_image_data[:, :, 100], cmap='gray')

            plt.subplot(322)
            plt.imshow(train_image_norm[100, :, :], cmap='gray')
            plt.subplot(324)
            plt.imshow(train_image_norm[:, 100, :], cmap='gray')
            plt.subplot(326)
            plt.imshow(train_image_norm[:, :, 100], cmap='gray')
            plt.show()

        train_data = nib.Nifti1Image(train_image_norm, train_img.affine)
        nib.save(train_data, os.path.join(train_norm_dir, 'Norm_'+train_file_paths[i]))
        
    # Normalize validation set
    val_file_paths = [x for x in os.listdir(val_root_dir) if x not in ['Normalized', 'Archive']]
    val_norm_dir = os.path.join(val_root_dir, 'Normalized')
    if not os.path.exists(val_norm_dir):
        os.makedirs(val_norm_dir)
        

    for i in range(len(val_file_paths)):
        val_img = nib.load(os.path.join(val_root_dir, val_file_paths[i]))

        val_image_data = val_img.get_fdata()
        
        val_image_min = np.min(val_image_data)
        val_image_max = np.max(val_image_data)

        val_image_norm = 2 * ((val_image_data - val_image_min) / (val_image_max - val_image_min)) - 1

        if i == 0:
            plt.figure(figsize=(15, 15))
            plt.subplot(321)
            plt.imshow(val_image_data[100, :, :], cmap='gray')
            plt.subplot(323)
            plt.imshow(val_image_data[:, 100, :], cmap='gray')
            plt.subplot(325)
            plt.imshow(val_image_data[:, :, 100], cmap='gray')

            plt.subplot(322)
            plt.imshow(val_image_norm[100, :, :], cmap='gray')
            plt.subplot(324)
            plt.imshow(val_image_norm[:, 100, :], cmap='gray')
            plt.subplot(326)
            plt.imshow(val_image_norm[:, :, 100], cmap='gray')
            plt.show()

        val_data = nib.Nifti1Image(val_image_norm, val_img.affine)
        nib.save(val_data, os.path.join(val_norm_dir, 'Norm_'+val_file_paths[i]))

def preprocess(train_dir, train_seg_dir, val_dir, val_seg_dir):
    '''
    Preprocessing to generate dictionary of images and corresponding labels
    :params:
        train_dir: root directory of training images
        train_seg_dir: root directory of segmentation images
        val_dir: root directory of validation images
        val_sef_images: root directory of validation segmentations
    :outputs:
        train_files: dictionary of training images ['img'] and corresponding labels ['seg']
        val_files: dictionary of training images ['img'] and corresponding labels ['seg']
    '''
    path_train_images = sorted(glob(os.path.join(train_dir, '*.nii.gz')))
    path_train_seg = sorted(glob(os.path.join(train_seg_dir, '*.nii.gz')))

    path_val_images = sorted(glob(os.path.join(val_dir, '*.nii.gz')))
    path_val_seg = sorted(glob(os.path.join(val_seg_dir, '*.nii.gz')))

    train_files = [{'img': img_name, 'seg': label_name} for img_name, label_name in zip(path_train_images, path_train_seg)]
    
    val_files = [{'img': img_name, 'seg': label_name} for img_name, label_name in zip(path_val_images, path_val_seg)]

    return train_files, val_files

def get_loaders(
    train_files, 
    val_files, 
    batch_size, 
    num_workers,
    pixdim=(1.5, 1.5, 1.0), 
    a_min=0, a_max=1500,
    spatial_size=[160, 160, 160],
    ):
    '''
    Transforms input and creates dataloaders
    :params:
        train_files: training file paths in dictionary with ['img'] and ['seg] paths
        val_files: validation file paths in dictionary with ['img'] and ['seg] paths
        batch_size: batch size for loader
        shuffle: bool to shuffle data to loader
        spatial_size: Corresponds to the min number of actual spatial slices divisible by {struid}**{# layers} -> eg. 2**5
    :outputs:
        train_loader: training loader
        val_loader: validation loader
    '''
    train_transform = Compose(
        [
            LoadImaged(keys=['img', 'seg']),
            EnsureChannelFirstd(keys=['img', 'seg']),
            RandFlipd(keys=['img', 'seg'], prob=0.25),
            RandRotated(
                keys=['img', 'seg'], 
                range_x=20*np.pi/180, 
                range_y=20*np.pi/180, 
                range_z=20*np.pi/180, 
                prob=0.25
            ),
            DivisiblePadd(keys=['img', 'seg'], k=32),
            ToTensord(keys=['img', 'seg']),
        ]
    )

    val_transform = Compose(
        [
            LoadImaged(keys=['img', 'seg']),
            EnsureChannelFirstd(keys=['img', 'seg']),
            RandFlipd(keys=['img', 'seg'], prob=0.25),
            RandRotated(
                keys=['img', 'seg'], 
                range_x=20*np.pi/180, 
                range_y=20*np.pi/180, 
                range_z=20*np.pi/180, 
                prob=0.25
            ),
            DivisiblePadd(keys=['img', 'seg'], k=32),
            ToTensord(keys=['img', 'seg']),
        ]
    )

    train_ds = Dataset(data=train_files, transform=train_transform)
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        #collate_fn=pad_list_data_collate,
        num_workers=num_workers,
        # pin_memory=True,
    )

    val_ds = Dataset(data=val_files, transform=val_transform)
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        #collate_fn=pad_list_data_collate,
        num_workers=num_workers,
        # pin_memory=True,
    )

    return train_loader, val_loader

def show_patient(data, SLICE_NUMBER=120):
    '''
    Archive
    '''
    view_train_patient = first(data)

    print('File: {}'.format(data.dataset.data[0]['img']))
    plt.figure("Visualization Train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"img {SLICE_NUMBER}")
    plt.imshow(view_train_patient["img"][0, 0, :, SLICE_NUMBER, :], cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title(f"seg {SLICE_NUMBER}")
    plt.imshow(view_train_patient["seg"][0, 0, :, SLICE_NUMBER, :], cmap='gray')
    plt.show()

def dice_coefficient():
    pass

def dice_metric(pred, label):
    '''
    In this function we take `predicted` and `target` (label) to calculate the dice coefficient then we use it 
    to calculate a metric value for the training and the validation.
    '''
    dice_value = DiceLoss(
        # to_onehot_y=True, 
        # sigmoid=True, 
        # squared_pred=True
    )
    value = 1 - dice_value(pred, label).item()
    return value

    

    

