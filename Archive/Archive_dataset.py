import os
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

# Using PyTorch Dataset
class SpineGenericDataset(Dataset):
    def __init__(self, img_dir, seg_dir, transform=None):
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        # self.transform = transform

        self.img = os.listdir(img_dir)
        self.seg = os.listdir(seg_dir)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img[index])
        seg_path = os.path.join(self.seg_dir, self.seg[index])
        
        image = nib.load(img_path).get_fdata()
        segmentation = nib.load(seg_path).get_fdata()
        # Note: [x, :, :] -> sagittal, [:, x, :] -> AP, [:, :, x] -> SI

        # if self.transform is not None:
        #     augmentations = self.transform(image=image, segmentation=segmentation)
        #     image = augmentations['image']
        #     segmentation = augmentations['seg']

        return image, segmentation
