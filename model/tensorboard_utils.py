import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
# from torchvision.utils import make_grid
from monai.utils import first
from monai.visualize.img2tensorboard import (
    add_animated_gif, 
    plot_2d_or_3d_image,
)

def image_grid(loader, tb_dir):
    '''
    Shows first index of data in dataloader in tensorboard
    :params:
        loader: train or val dataloader
        tb_dir: tensorboard directoru
    '''
    view_data = first(loader)
    plot_2d_or_3d_image(
        writer=SummaryWriter(log_dir=tb_dir), 
        data=view_data['img'],
        step=0,
        frame_dim=-1,
        max_channels=3,
        tag='image'
    )
    plot_2d_or_3d_image(
        writer=SummaryWriter(log_dir=tb_dir), 
        data=view_data['seg'],
        step=0,
        frame_dim=-1,
        max_channels=3,
        tag='label'
    )

