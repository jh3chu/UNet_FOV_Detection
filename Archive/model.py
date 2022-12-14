
'''
https://www.youtube.com/watch?v=IHq1t7NxS8k&ab_channel=AladdinPersson

Differences to U-net paper
- Use padded convolutions ('same' padding) instead of padding the image prior ('valid' padding)
- Larger images require a multi-crop for padding
- Kaggle comp winners used padded convolutions

'''


import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module): 
    # torch.nn.Module is the base class for all nn modules
    # Note: we are creating a child class based on an already defined parent class that we imported
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__() 
            # ??? Why are we passing in DoubleConv into super()???   
            # super(): Used to give access to methods and properties of a parent or sibling class, returns an object that represents the parent class
        
        self.conv = nn.Sequential(
            # nn.Sequential(): A sequential container for for modules (eg. put all sequential 2d convolutions into one container)
            
            # First convolution layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.Conv2d(): Applies a 2d convolution 
                # Note: bias=False as we are using nn.BatchNorm2d() -> bias get cancelled by batch norm
            nn.BatchNorm2d(out_channels),
                # nn.BatchNorm2d(): Applies a batch normailzation by re-centering and re-scaling layers - see https://en.wikipedia.org/wiki/Batch_normalization
            nn.ReLU(inplace=True),
                # nn.ReLU(): Applies the rectified linear unit function (ReLU) as the activation function 
            
            # Second convolution layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        # in_channels: Number of channels input to UNet
        # out_channels: Number of channels output of UNet -> we choose 1 as we are doing binary classification
        # features: The features of the down sampling and up sampling (ie. number of channels in a layer of a level)

        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
            # nn.ModuleList(): Holds submodules in a list -> acts like a python list, but stores nn layers

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            #nn.MaxPool2d(): Max pools in 2D based on kernel size and stride
        
        # Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature)) # Acting as a list
            in_channels = feature
        
        # Up part of UNet (Note: We using transpose convolutions (has some artifacts) -> could also use a bilinear transformation and conv layer after)
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, out_channels=feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, out_channels=feature))

        # Bottom of UNet
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Final UNet conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    

    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        
        # Reverse the skip connections list as we are concatenating the last one to the first upsample
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x) # Even i is nn.ConvTranspose2d()
            skip_connection = skip_connections[i//2] 

            # Resize x in case max pool floor sizing (input 161x161 -> max pool -> output 80x80)
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1) # Concatenate the data
            x = self.ups[i+1](concat_skip) # Odd i is DoubleConv()

        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)

    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


if __name__ == '__main__':
    test()




