
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader


from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

import pydicom
from dicom_contour.contour import *
from unet_architecture import *

torch.manual_seed(2020)


f_size = 3
pad = int((f_size-1)/2)
padding = (pad, pad, pad)
# chs = [32, 64, 128, 256, 512]
# chs = [32, 64, 128, 256]

class ConvBlock(nn.Module):
    """3D Unet with 4 downsamples/upsamples
    Input Channels: 3 (mask, PTV_mask(modified), CT, spine)
    Filter: 3x3 with stride 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # High momentum due to small batch size
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, f_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=0.99)
        # 1x1 convolution
        self.conv2 = nn.Conv3d(out_channels, out_channels, f_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=0.99)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x
        
class unet_3d(nn.Module):
    def __init__(self, num_downsample=4):
        
        super().__init__()
        # pooling
        self.num_downsample = num_downsample
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        ## Notsure what this does
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # ENCODER
        self.conv00 = ConvBlock(3, 32)
        self.conv10 = ConvBlock(32, 64)
        self.conv20 = ConvBlock(64, 128)
        self.conv30 = ConvBlock(128, 256)
        self.conv40 = ConvBlock(256, 512)  
        
        # DECODER
        self.upconv31 = nn.ConvTranspose3d(512, 256, 2, stride=2)
        self.conv31 = ConvBlock(2*256, 256)
        self.upconv22 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.conv22 = ConvBlock(2*128, 128)
        self.upconv13 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.conv13 = ConvBlock(2*64, 64)
        self.upconv04 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.conv04 = ConvBlock(2*32, 32)
        
        # NOTE: UNSURE ABOUT THIS PART
        # final layers
        self.final04 = nn.Conv3d(32, 1, 1)
    
    def forward(self, x):
        
        # Encoder Path
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool(x00))
        x20 = self.conv20(self.pool(x10))
        
        if self.num_downsample==3:
            x = self.conv30(self.pool(x20))
            
        if self.num_downsample==4:
            x30 = self.conv30(self.pool(x20))
            x = self.conv40(self.pool(x30))
    
            # UPSAMPLING
            x = self.upconv31(x)
            x = self.conv31(torch.cat((x30,x), dim=1))
            
        x = self.upconv22(x)
        x = self.conv22(torch.cat((x20,x),dim=1))
        x = self.upconv13(x)
        x = self.conv13(torch.cat((x10,x),dim=1))
        x = self.upconv04(x)
        x = self.conv04(torch.cat((x00,x),dim=1))
        
        # Outputs
        x  = self.final04(x)
        return x
            
class OrganWeightedLoss(nn.Module):
    """Penalize heavily for PTV and cord masks"""
    
    def __init__(self, loss_func="MSE", alpha=1, beta=1, gamma=1):
        super(OrganWeightedLoss, self).__init__()
        self.alpha = alpha   # PTV
        self.beta = beta     # spine
        self.gamma = gamma   # total output
        
        if loss_func == "L1":
            self.lossFun = nn.L1Loss()
        elif loss_func == "Huber":
            self.lossFun = nn.SmoothL1Loss(beta=1.0)   # play with Deltas
        else:
            self.lossFun = nn.MSELoss()

    def forward(self, X_batch, outputs, Y_batch):
        
        PTV_mask     = X_batch[:,0,:,:,:]
        cord_mask    = X_batch[:,2,:,:,:]
        # healthy_mask = X_batch[:,3,:,:,:]
        
        # NOTE: The general loss includes the CT
        # loss_healthy = self.gamma * self.lossFun(healthy_mask*outputs, healthy_mask *Y_batch)
        loss_PTV     = self.lossFun(PTV_mask*outputs, PTV_mask*Y_batch)
        loss_cord    = self.lossFun(cord_mask*outputs, cord_mask*Y_batch)   # Try torch.sigmoid(outputs)
        loss_general  = self.lossFun(outputs, Y_batch) 
        
        loss_oneBatch =  (self.alpha *loss_PTV) + (self.beta *loss_cord) + (self.gamma*loss_general)
        return loss_oneBatch
    
    
    
def show_result(epoch_idx, ds_val, dose_preds):
    """Show results of all validation sets"""
    for i in range(len(ds_val)):
        true_dose, PTV_mask, PTV_img_arr, cord_mask = ds_val[i]
        true_dose = true_dose.numpy()

        # Normalize
        prediction = dose_preds[epoch_idx][i].detach().cpu().numpy()[0]   

        # Get the max and min of true and predicted dose arrays for display
        true_min, true_max = true_dose.min(), true_dose.max()
        pred_min, pred_max = prediction.min(), prediction.max()
        # prediction = (prediction - pred_min)/ (pred_max - pred_min)

        print(true_min, true_max, pred_min, pred_max)

        for i in range(16):
            plt.figure(figsize=(10, 5))
            plt.subplot(1,2,1)
            plt.imshow(true_dose[i], cmap='coolwarm', vmin=true_min, vmax=true_max+ true_min)     # first slice of 16

            plt.subplot(1,2,2)
            plt.imshow(prediction[i], cmap='coolwarm', vmin=0, vmax=pred_max) # pred_max+pred_min)  

