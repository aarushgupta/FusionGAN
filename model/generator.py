import torch
import torch.nn as nn


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        # if torch.cuda.is_available():
        	# residual = torch.cuda.FloatTensor(residual)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# Generator Definition

class Generator(nn.Module):
    def __init__(self, block):
        super(Generator, self).__init__()
        
        self.conv1_x = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2))
        
        self.conv2_x = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2))

        self.conv3_x = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.conv1_y = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2))
        
        self.conv2_y = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2))

        self.conv3_y = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))      
            
        # 2 Residual Blocks for Identity Image
        self.block1_x = block(16, 16)
#         downsample_x = nn.Sequential(conv3x3(16, 1, 1), nn.BatchNorm2d(1))
#         self.block2_x = block(16, 1, 1, downsample_x)
        self.block2_x = block(16, 16)

        # 2 Residual Blocks for Shape Image
        self.block1_y = block(16, 16)
#         downsample_y = nn.Sequential(conv3x3(16, 1, 1), nn.BatchNorm2d(1))
#         self.block2_y = block(16, 1, 1, downsample_y)
        self.block2_y = block(16, 16)

        # 2 Residual Blocks for Combined(concat) image
        downsample1_concat = nn.Sequential(conv3x3(32, 16, 1), nn.BatchNorm2d(16))
        self.block1_concat = block(32, 16, 1, downsample1_concat)

        self.block2_concat = block(16, 16)
        
        # Upsampling layers
                
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ConvTranspose2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(32, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True))
        
        
    def forward(self, x, y):
        
        x = self.conv1_x(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.block1_x(x)
        x = self.block2_x(x)
        
        y = self.conv1_y(y)
        y = self.conv2_y(y)
        y = self.conv3_y(y)
        y = self.block1_y(y)
        y = self.block2_y(y)
        
        concat_result = torch.zeros([x.shape[0], x.shape[1] * 2, x.shape[2], x.shape[3]], dtype=x.dtype)
#         print(x.shape, y.shape, concat_result.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                concat_result[i][j] = x[i][j]
                concat_result[i][j + x.shape[1]] = y[i][j]
        if torch.cuda.is_available():
        	concat_result = concat_result.cuda()
        concat_result = self.block1_concat(concat_result)
        concat_result = self.block2_concat(concat_result)
        
        upsampled_1 = self.upsample1(concat_result)
        upsampled_2 = self.upsample2(upsampled_1)
#         print(upsample2.shape)
        return upsampled_2
        