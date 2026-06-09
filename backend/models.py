import torch
import torch.nn as nn
from torchvision import models
from backend.attention_blocks import CBAM

class ResNet50_CBAM_MultiTask(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50_CBAM_MultiTask, self).__init__()
        # Load pre-trained ResNet50 base
        if pretrained:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            resnet = models.resnet50()
            
        # Extract features for skip connections
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1 # 256 channels, H/4 (56x56)
        self.layer2 = resnet.layer2 # 512 channels, H/8 (28x28)
        self.layer3 = resnet.layer3 # 1024 channels, H/16 (14x14)
        self.layer4 = resnet.layer4 # 2048 channels, H/32 (7x7)
        
        # Bottleneck Attention
        self.cbam = CBAM(2048)
        
        # Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, 2)
        
        # Segmentation Decoder with Skip Connections
        # Upsample 1: 2048 -> 512 (H/16). Concat with layer3 (1024) = 1536 channels.
        self.up1 = nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2)
        self.conv_dec1 = nn.Sequential(
            nn.Conv2d(512 + 1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Upsample 2: 512 -> 256 (H/8). Concat with layer2 (512) = 768 channels.
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_dec2 = nn.Sequential(
            nn.Conv2d(256 + 512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Upsample 3: 256 -> 128 (H/4). Concat with layer1 (256) = 384 channels.
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_dec3 = nn.Sequential(
            nn.Conv2d(128 + 256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Upsample 4: 128 -> 64 (H). Concat with stem is omitted for computational ease.
        # Direct projection from H/4 to H (224x224) using a 4x stride transpose conv.
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4)
        self.conv_dec4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1) # 2 channels: [0] = optic disc, [1] = optic cup
        )

    def forward(self, x):
        # 1. Forward through ResNet backbone to collect skip connections
        s = self.stem(x)
        l1 = self.layer1(s)  # H/4 (56x56), 256 channels
        l2 = self.layer2(l1) # H/8 (28x28), 512 channels
        l3 = self.layer3(l2) # H/16 (14x14), 1024 channels
        l4 = self.layer4(l3) # H/32 (7x7), 2048 channels
        
        # 2. Apply CBAM attention to the deepest features
        refined = self.cbam(l4)
        
        # 3. Classification Head
        pool = self.avgpool(refined)
        flat = torch.flatten(pool, 1)
        class_out = self.classifier(flat)
        
        # 4. Segmentation Decoder
        d1 = self.up1(refined)
        d1 = torch.cat([d1, l3], dim=1)
        d1 = self.conv_dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, l2], dim=1)
        d2 = self.conv_dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([d3, l1], dim=1)
        d3 = self.conv_dec3(d3)
        
        d4 = self.up4(d3)
        seg_out = self.conv_dec4(d4)
        
        return class_out, seg_out


class EfficientNetB0_CBAM_MultiTask(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetB0_CBAM_MultiTask, self).__init__()
        # Load pre-trained EfficientNet B0 base
        if pretrained:
            effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        else:
            effnet = models.efficientnet_b0()
            
        self.features = effnet.features # 9 stages
        
        # Bottleneck Attention
        self.cbam = CBAM(1280)
        
        # Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1280, 2)
        
        # Segmentation Decoder with Skip Connections
        # Upsample 1: 1280 -> 256 (H/16). Concat with stage 5 (112 channels) = 368 channels.
        self.up1 = nn.ConvTranspose2d(1280, 256, kernel_size=2, stride=2)
        self.conv_dec1 = nn.Sequential(
            nn.Conv2d(256 + 112, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Upsample 2: 128 -> 64 (H/8). Concat with stage 3 (40 channels) = 104 channels.
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_dec2 = nn.Sequential(
            nn.Conv2d(64 + 40, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Upsample 3: 64 -> 32 (H/4). Concat with stage 2 (24 channels) = 56 channels.
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_dec3 = nn.Sequential(
            nn.Conv2d(32 + 24, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Upsample 4: 32 -> 16 (H). Upsample from H/4 to H (224x224) using a 4x stride transpose conv.
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=4)
        self.conv_dec4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1) # 2 channels: [0] = optic disc, [1] = optic cup
        )

    def forward(self, x):
        # 1. Forward through EfficientNet backbone and extract skip maps
        x0_1 = self.features[0](x)
        x0_1 = self.features[1](x0_1)
        x2 = self.features[2](x0_1) # H/4 (56x56), 24 channels
        x3 = self.features[3](x2)   # H/8 (28x28), 40 channels
        x4 = self.features[4](x3)
        x5 = self.features[5](x4)   # H/16 (14x14), 112 channels
        x6 = self.features[6](x5)
        x7 = self.features[7](x6)
        x8 = self.features[8](x7)   # H/32 (7x7), 1280 channels
        
        # 2. Apply CBAM attention to bottleneck features
        refined = self.cbam(x8)
        
        # 3. Classification Head
        pool = self.avgpool(refined)
        flat = torch.flatten(pool, 1)
        class_out = self.classifier(flat)
        
        # 4. Segmentation Decoder
        d1 = self.up1(refined)
        d1 = torch.cat([d1, x5], dim=1)
        d1 = self.conv_dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, x3], dim=1)
        d2 = self.conv_dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.conv_dec3(d3)
        
        d4 = self.up4(d3)
        seg_out = self.conv_dec4(d4)
        
        return class_out, seg_out
