import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib
from torchvision.ops import masks_to_boxes
from glob import glob
import model.hyperparameters as hp

from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    DivisiblePadd,
    ToTensord,
    RandFlipd,
    RandRotated,
)
from monai.losses import DiceLoss

from monai.networks.nets import UNet
from monai.networks.layers import Norm

from utils.utils import dice_metric

MODEL_PATH ='model\SAVE_MODELS'
TEST_IMG_PATH = r'D:\Jonathan\2_Projects\Fov_Detection\FoV_Data\test_img\T1\Normalized'
TEST_SEG_PATH = r'D:\Jonathan\2_Projects\Fov_Detection\FoV_Data\test_seg\T1'
ML_SLICE = 100

def preprocess():
    path_test_images = sorted(glob(os.path.join(TEST_IMG_PATH, '*.nii.gz')))
    path_test_seg = sorted(glob(os.path.join(TEST_SEG_PATH, '*.nii.gz')))

    test_files = [{'img': img_name, 'seg': label_name} for img_name, label_name in zip(path_test_images, path_test_seg)]
    
    return test_files

def get_loaders(
    test_files,
    batch_size,
    shuffle,
    num_workers,
):

    test_transform = Compose(
        [
            LoadImaged(keys=['img', 'seg']),
            EnsureChannelFirstd(keys=['img', 'seg']),
            # RandFlipd(keys=['img', 'seg'], prob=0.25),
            # RandRotated(
            #     keys=['img', 'seg'], 
            #     range_x=20*np.pi/180, 
            #     range_y=20*np.pi/180, 
            #     range_z=20*np.pi/180, 
            #     prob=0.25
            # ),
            DivisiblePadd(keys=['img', 'seg'], k=32),
            ToTensord(keys=['img', 'seg']),
        ]
    )

    test_ds = Dataset(data=test_files, transform=test_transform)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return test_loader


def test_fn():
    test_files = preprocess()

    test_loader = get_loaders(
        test_files=test_files,
        batch_size=hp.BATCH_SIZE,
        shuffle=False,
        num_workers=hp.NUM_WORKERS
    )
    
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

    # Load best model
    model.load_state_dict(torch.load(
        os.path.join(MODEL_PATH, 'best_model_e10.pth'),
        map_location='cpu'
    ))

    output_segmentations = []
    model.eval()
    with torch.no_grad():
        test_metric = 0

        for test_step, test_data in enumerate(test_loader):
            test_inputs = test_data['img']
            test_labels = test_data['seg']

            # Add channel to label
            test_labels_2 = 1 - test_labels
            test_labels = torch.cat((test_labels, test_labels_2), axis=1)

            test_inputs, test_labels = (test_inputs.to(hp.DEVICE), test_labels.to(hp.DEVICE))

            test_outputs = model(test_inputs)
            test_outputs = F.softmax(test_outputs)
            output_segmentations.append(
                test_outputs.cpu().detach().numpy()[0, 0, :, :, :]
            ) # make dictionary list?

            # Dice loss
            test_metric += dice_metric(test_outputs, test_labels)

            if (test_step+1) % 10 == 0:
                print('Step: {}/{} \tTest dice loss: {:.4f}'.format(
                    test_step+1, 
                    len(test_loader),
                    test_metric,
                ))

    return (test_metric / len(test_loader)), output_segmentations

def get_bbox(output_segmentations):
    a0_bbox = []
    for seg in output_segmentations:
        seg = torch.from_numpy(seg.astype(np.uint8))
        
        a0_boxes = np.empty((seg.shape[0], 4))
        a0_boxes[:] = np.nan
        for a0_slice in range(seg.shape[0]):
            slice = seg[a0_slice, :, :]
            obj_ids = torch.unique(slice)
            obj_ids = obj_ids[1:] # Filter out background
            mask = slice == obj_ids[:, None, None]
            if mask.shape[0] != 0:
                # boxes = masks_to_boxes(mask)
                a0_boxes[a0_slice, :] = masks_to_boxes(mask)
        a0 = {
            'min_x1': np.nanmin((a0_boxes[:, 0]-5)).clip(0),
            'min_y1': np.nanmin(a0_boxes[:, 1]-5).clip(0),
            'max_x1': np.nanmax(a0_boxes[:, 2]+5).clip(max=seg.shape[1]),
            'max_y1': np.nanmax(a0_boxes[:, 3]+5).clip(max=seg.shape[2]),
        }
        a0_bbox.append(a0)

    return a0_bbox

def visualize(outputs, a0_bbox):  
    test_files = preprocess()
    for i in range(len(test_files)):
        img_path = test_files[i]['img']
        seg_path = test_files[i]['seg']

        img = nib.load(img_path)
        img = img.get_fdata()
        seg = nib.load(seg_path)
        seg = seg.get_fdata()
        output_seg = outputs[i]

        ML_img_slice = img[ML_SLICE, :, :]
        ML_seg_slice = seg[ML_SLICE, :, :]
        ML_pred_slice = output_seg[ML_SLICE, :, :]

        # Find difference of shape due to padding 
        y_diff = (ML_pred_slice.shape[0] - ML_seg_slice.shape[0]) / 2
        x_diff = (ML_pred_slice.shape[1] - ML_seg_slice.shape[1]) / 2

        fig1 = plt.figure(figsize=(5,5))
        plt.imshow(ML_img_slice, cmap='gray')
        fig2 = plt.figure(figsize=(5,5))
        plt.imshow(ML_seg_slice, cmap='gray')
        fig3, ax = plt.subplots()
        ax.imshow(ML_pred_slice, cmap='gray')
        rec = patches.Rectangle(
            (a0_bbox[i]['min_x1'], a0_bbox[i]['min_y1']), # start coord
            a0_bbox[i]['max_x1'] - a0_bbox[i]['min_x1'], # width
            a0_bbox[i]['max_y1'] - a0_bbox[i]['min_y1'], # height
            linewidth=1,
            edgecolor='r',
            facecolor='none',
        )
        ax.add_patch(rec)
        fig4, ax4 = plt.subplots()
        ax4.imshow(ML_img_slice, cmap='gray')
        rec = patches.Rectangle(
            (a0_bbox[i]['min_x1']-x_diff, a0_bbox[i]['min_y1']-y_diff),
            a0_bbox[i]['max_x1'] - a0_bbox[i]['min_x1'],
            a0_bbox[i]['max_y1'] - a0_bbox[i]['min_y1'],
            linewidth=1,
            edgecolor='r',
            facecolor='none',
        )
        ax4.add_patch(rec)
        plt.show()

def get_crop_nifti():
    pass

def main():
    test_metric, output_segmentations = test_fn()
    print(test_metric)
    a0_bbox = get_bbox(output_segmentations)
    visualize(output_segmentations, a0_bbox)

if __name__ == '__main__':
    main()