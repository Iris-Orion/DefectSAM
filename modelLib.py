import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp
from torchvision import transforms
from pathlib import Path
from torchinfo import summary

# from nets.med_seg_diff_pytorch.med_seg_diff_pytorch import MedSegDiff, Unet

import data_setup

#--------------------Vanilla Unet ---------------------#
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    """ downscale with maxpool then double conv """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Up conv then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            # 选择双线性插值重建
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # 选择转置卷积重建，考虑一下步长的大小
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 =  F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)
        
class UNet(nn.Module):
    def __init__(self, n_channels, num_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = num_classes
        self.bilinear = bilinear
    
        self.input = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.output = (OutConv(64, num_classes))
    
    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.output(x)
        return logits
    
    # need modify here
    def use_checkpointing(self):
        self.input = torch.utils.checkpoint(self.input)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.output = torch.utils.checkpoint(self.output)

def get_vanilla_unet(n_channels = 3, num_classes = 4):
    return UNet(n_channels=n_channels, num_classes=num_classes)

def test_vanilla_unet():
    model = UNet(n_channels=3, num_classes=4)
    input = torch.rand(1, 3, 256, 1600)
    print("---Test vanilla unet---")
    print(f"input shape (B, C, H, W) is {input.shape}")
    output = model(input)
    print(f"output shape: {output.shape}")
    summary(model, input_size=(1, 3, 256, 1600))
#------------------------vallina Unet end-------------------------------------#

#--------------------------AutoEncoder-----------------------------------#
class ConvAutoEncoder(nn.Module):
    """
    this will return an autoencoder model
    """
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=4),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=4),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=5, stride=2, output_padding=1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
#--------------------------AutoEncoder End-----------------------------------#


#--------------------------SMP U-Net Start-----------------------------------#
def get_smp_unet(encoder_name = "resnet34", encoder_weights="imagenet", in_channels=3, classes=4, activation=None):
    smp_resnet34_encoder_unet_model = smp.Unet(
        encoder_name = encoder_name,
        encoder_weights = encoder_weights,
        in_channels = in_channels,
        classes = classes,
        activation = activation
    )
    return smp_resnet34_encoder_unet_model

def test_smp_unet():
    model = get_smp_unet(encoder_name = "resnet34", encoder_weights="imagenet", in_channels=3, classes=4, activation=None)
    input = torch.rand(1, 3, 256, 1600)
    print("---Test smp unet---")
    print(f"input shape (B, C, H, W) is {input.shape}")
    output = model(input)
    print(f"output shape: {output.shape}")
    summary(model, input_size=(1, 3, 256, 1600))
#-------------------SMP U-Net End-----------------------------------#


#--------------------------SMP deeplabv3plus Start-------------------------#
def get_smp_deeplabv3plus(encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=3, classes=4, activation=None):
    """
    骨干网络可以使用 ResNet50, efficient-b0, b7的预训练权重
    """
    smp_deeplabv3plus_model = smp.DeepLabV3Plus(
        encoder_name = encoder_name,                     # 使用 ResNet50 作为骨干网络     # b0, b7的预训练权重都在
        encoder_weights = encoder_weights,               # 使用 ImageNet 预训练权重
        in_channels = in_channels,
        classes = classes,                               # 设置类别数，例如 4 类缺陷
        activation = activation                                # 设置激活函数，适合多标签分类
        )
    return smp_deeplabv3plus_model

def test_smp_deeplabv3plus():
    model = get_smp_deeplabv3plus(encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=3, classes=4, activation=None)
    input = torch.rand(1, 3, 256, 1600)
    print("---Test smp deeplabv3plus---")
    print(f"input shape (B, C, H, W) is {input.shape}")
    model.eval()                                        # 如果不设置为eval会报batch size为1的错误
    output = model(input)
    print(f"output shape: {output.shape}")
    summary(model, input_size=(1, 3, 256, 1600))
#-------------------SMP deeplabv3plus End-----------------------------------#


#---------------------------smp segformer start------------------------------#
def build_segformer_model(
    encoder_name="mit_b0",
    encoder_weights="imagenet",
    in_channels=3,
    classes=4,
    activation=None
):
    """
    构建 SegFormer 分割模型
    
    参数:
        encoder_name: 编码器backbone名称, 可选 ['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5']
        encoder_weights: 预训练权重, 'imagenet' 或 None
        in_channels: 输入通道数
        classes: 分割类别数量
        activation: 激活函数, 例如 'sigmoid' 用于二分类
    返回:
        model: SegFormer 模型实例
    """
    # 创建模型配置
    model = smp.Segformer(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation
    )
    
    return model

def test_smp_segformer():
    model = build_segformer_model(encoder_name="mit_b0", encoder_weights="imagenet", in_channels=3, classes=4, activation=None)
    input = torch.rand(1, 3, 256, 1600)
    print("---Test smp segformer---")
    print(f"input shape (B, C, H, W) is {input.shape}")
    # model.eval()                                        # 如果不设置为eval会报batch size为1的错误
    output = model(input)
    print(f"output shape: {output.shape}")
    pred = torch.sigmoid(output)
    pred = (pred > 0.5).to(torch.float32)  # 加一个阈值
    pred_flat = pred.view(-1)
    print(f"pred_flat shape: {pred_flat.shape}")
    summary(model, input_size=(1, 3, 256, 1600))

#---------------------------smp segformer end--------------------------------#

#---------------------------MedSegDiff--------------------------------#
def get_medsegdiff_model(img_size: int=128, in_put_channels: int = 3, mask_channels: int = 1):
    model = Unet(
    dim = 64,
    image_size = img_size,
    mask_channels = mask_channels,          # segmentation has 1 channel
    input_img_channels = in_put_channels,     # input images have 3 channels
    dim_mults = (1, 2, 4, 8)
    )

    diffusion = MedSegDiff(
        model,
        timesteps = 1000
    ).cuda()
    return diffusion

#---------------------------MedSegDiff--------------------------------#

#------------------transfomrs-----------------------#
def get_transforms():
    train_transforms = transforms.Compose([
            # transforms.Resize()    ?? 是否需要resize
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.ColorJitter(brightness=[0, 0.2], contrast=[0, 0.2]),  ## 这个参数设置的不对，不要用
            transforms.ToTensor(),   # totensor 需要在 normalize之前
            # 来自imagenet的计算结果
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),   # totensor 需要在 normalize之前
        # 来自imagenet的计算结果
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transforms, val_transforms
#------------------transfomrs-----------------------#

#============ 通用的获取模型函数 ============#
def get_model(model_name: str = "smp_deeplabv3plus"):
    """ 根据传入的 model_name 来选择返回对应的模型实例 """
    print(f"using model type: {model_name}")
    model_name = model_name.lower().strip()
    if model_name == "vanilla_unet":
        return UNet(n_channels=3, num_classes=4)
    elif model_name == "smp_unet":
        return get_smp_unet()
    elif model_name == "smp_deeplabv3plus":
        return get_smp_deeplabv3plus()
    else:
        raise ValueError(f"Unknown model_name: {model_name} !")
#============ 通用的获取模型函数 ============#

#------test_smp_on_severstal---------------------#
def test_smp_on_severstal():
    p = Path("data/severstal_steel_defect_detection")
    train_transforms, val_transforms = get_transforms()
    train_df, val_df = data_setup.traindf_preprocess(split_seed=42)
    train_dataloader, val_dataloader = data_setup.create_dataloaders(train_df,
                                                                    val_df,
                                                                    data_path = p,
                                                                    train_transform=train_transforms,
                                                                    val_transform=val_transforms,
                                                                    batch_size=16,
                                                                    num_workers=8)
    print(f"train_dataloader长度: {len(train_dataloader)}, val_dataloader长度: {len(val_dataloader)}")

    model = get_model(model_name="smp_deeplabv3plus")

    img, mask, image_id = next(iter(train_dataloader))
    # input: b, c, h, w   ---- > output
    print(f"input img batch shape: b, c, h, w: {img.shape}")
    print(f"input mask shape: {mask.shape}")
    output = model(img)
    print(f"model output: {output.shape}")
    # print(output)
#------test_smp_on_severstal---------------------#




if __name__ == "__main__":
    # test_vanilla_unet()
    test_smp_unet()
    # test_smp_deeplabv3plus()
    # test_smp_on_severstal()
    # test_smp_segformer()