import os
import torch
import torch.nn as nn
import numpy as np
import skimage
import PIL.Image
import torch.nn.functional as f
import torchvision.transforms as transforms
from skimage import morphology,io
from skimage.color import rgb2gray
from nets.nets_utility import GaussBlur
from nets.guided_filter import GuidedFilter
import cv2


class GACN_Fuse():
    """
    Fusion Class
    """
    def __init__(self):
        # initialize model
        self.device = "cuda:0"
        self.model = GACNFuseNet()
        self.model_path = os.path.join(os.getcwd(), "nets", "parameters", "GACN.pkl")
        self.checkpoint = torch.load(self.model_path, map_location={'cuda:3': 'cuda:0'})
        self.model.load_state_dict(self.checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self.feature_extraction = GACNFeatureExtraction().to(self.device)
        self.decision_path = GACNDecisionPath().to(self.device)
        model_dict_feature_extraction = self.feature_extraction.state_dict()
        model_dict_decision_path = self.decision_path.state_dict()
        pretrained_dict_feature_extraction = {k: v for k, v in self.checkpoint.items() if k in self.feature_extraction.state_dict()}
        pretrained_dict_decision_path = {k: v for k, v in self.checkpoint.items() if k in self.decision_path.state_dict()}
        model_dict_feature_extraction.update(pretrained_dict_feature_extraction)
        model_dict_decision_path.update(pretrained_dict_decision_path)
        self.feature_extraction.load_state_dict(model_dict_feature_extraction)
        self.decision_path.load_state_dict(model_dict_decision_path)
        self.decision_path.eval()
        self.feature_extraction.eval()
        self.mean_value = 0.4532911013165387
        self.std_value = 0.2650597193463966
        self.data_transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([self.mean_value], [self.std_value])
        ])

    def fuse(self, img1, img2):
        """
        Double images fusion
        """
        ndim = img1.ndim
        
        if ndim == 2:
            img1_gray = img1
            img2_gray = img2
        else:
            img1_gray = rgb2gray(img1)
            img2_gray = rgb2gray(img2)
        
        img1_gray_pil = PIL.Image.fromarray(img1_gray)
        img2_gray_pil = PIL.Image.fromarray(img2_gray)
        img1_tensor = self.data_transforms(img1_gray_pil).unsqueeze(0).to(self.device)               
        img2_tensor = self.data_transforms(img2_gray_pil).unsqueeze(0).to(self.device)
        mask, mask_BGF = self.model.forward(img1_tensor, img2_tensor)
        img1_t = self.data_transforms(img1).unsqueeze(0).to(self.device)
        img2_t = self.data_transforms(img2).unsqueeze(0).to(self.device)

        if ndim == 3:
            mask.repeat(1, 3, 1, 1)

        fused = torch.mul(mask_BGF, img1_t) + torch.mul((1 - mask_BGF), img2_t)
        if ndim == 3:
            fused = fused.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        else:
            fused = fused.squeeze(0).squeeze(0).cpu().detach().numpy()
        return fused

    def multi_fuse_calibration(self, path):
        """
        Multi image fusion using Decision calibration fusion strategy
        """
        x = 2400
        y = 1800
        img_list = os.listdir(path)
        first  = 'V017.jpg'
        second = 'V047.jpg'
        for i in range(len(img_list)):
            if img_list[i] == first:
                img_list[i], img_list[0] = img_list[0], img_list[i]
            elif img_list[i] == second:
                img_list[i], img_list[1] = img_list[1], img_list[i]
        if ".ipynb_checkpoints" in img_list:
            img_list.remove(".ipynb_checkpoints")
        num = len(img_list)
        img1 = io.imread(os.path.join(path, img_list[0]))
        ndim = img1.ndim
        img1 = cv2.resize(img1, (x, y))
        img1_t = self.data_transforms(img1).unsqueeze(0)

        if ndim == 2:
            img1_gray = img1
            img_total = torch.zeros((num, 1, img1.shape[0], img1.shape[1]))
        else:
            img1_gray = rgb2gray(img1)
            img_total = torch.zeros((num, 3, img1.shape[0], img1.shape[1]))
        img1_gray_pil = PIL.Image.fromarray(img1_gray)
        img1_tensor = self.data_transforms(img1_gray_pil).unsqueeze(0).to(self.device)
        img_total[0, :, :, :] = img1_t
        mask = torch.zeros((num - 1, 1, img1.shape[0], img1.shape[1])).to(self.device)
        decision_volume = torch.zeros((num, 1, img1.shape[0], img1.shape[1])).to(self.device)

        # Mask generation
        with torch.no_grad():
            cat_1 = self.feature_extraction(img1_tensor.to(self.device))
            f1_sf = self.channel_sf(cat_1)

        for i in range(num - 1):
            print(num, ':', i)
            img2 = io.imread(os.path.join(path, img_list[i + 1]))
            img2 = cv2.resize(img2, (x, y))
            img2_t = self.data_transforms(img2).unsqueeze(0)
            if ndim == 2:
                img2_gray = img2
            else:
                img2_gray = rgb2gray(img2)

            img2_gray_pil = PIL.Image.fromarray(img2_gray)
            img2_tensor = self.data_transforms(img2_gray_pil).unsqueeze(0)

            with torch.no_grad():

                cat_2 = self.feature_extraction(img2_tensor.to(self.device))
                f2_sf = self.channel_sf(cat_2)
                bimap = torch.sigmoid(1000 * (f1_sf - f2_sf))
                _, mask_1 = self.decision_path(bimap, img1_tensor, img2_tensor.to(self.device))
            mask[i, :, :, :] = mask_1
            img_total[i + 1, :, :, :] = img2_t

        # Decision volume generation
        decision_volume[0, :, :, :] = mask[0, :, :, :]
        decision_volume[1, :, :, :] = 1 - mask[0, :, :, :]
        for i in range(1, num - 1):
            decision_volume[i + 1, :, :, :] = decision_volume[0, :, :, :] / (mask[i, :, :, :] + 1e-4) * \
                                              (1 - mask[i, :, :, :])
        _, ind = torch.max(decision_volume, dim=0)
        for i in range(num):
            decision_volume[i, :, :, :] = (ind == i)

        # Fusion
        if ndim == 3:
            decision_volume.repeat(1, 3, 1, 1)
        fused = torch.sum(img_total.to(self.device) * decision_volume, dim=0)
        if ndim == 3:
            fused = fused.permute(1, 2, 0).cpu().detach().numpy()
        else:
            fused = fused.squeeze(0).cpu().detach().numpy()

        return fused

    def multi_fuse_origin(self, path):
        """
        Multi image fusion using one by one serial fusion strategy
        """
        x = 2400
        y = 1800
        img_list = sorted(os.listdir(path))
        if ".ipynb_checkpoints" in img_list:
            img_list.remove(".ipynb_checkpoints")
        num = len(img_list)
        img1 = io.imread(os.path.join(path, img_list[0]))
        img1 = cv2.resize(img1, (x, y))
        img1_t = self.data_transforms(img1).unsqueeze(0).to(self.device)
        ndim = img1.ndim
        if ndim == 2:
            img1_gray = img1
        else:
            img1_gray = rgb2gray(img1)
        img1_gray_pil = PIL.Image.fromarray(img1_gray)
        img1_tensor = self.data_transforms(img1_gray_pil).unsqueeze(0).to(self.device)
        for i in range(num - 1):
            print(num, ':', i)
            img2 = io.imread(os.path.join(path, img_list[i + 1]))
            img2 = cv2.resize(img2, (x, y))
            img2_t = self.data_transforms(img2).unsqueeze(0).to(self.device)
            if ndim == 2:
                img2_gray = img2
            else:
                img2_gray = rgb2gray(img2)

            img2_gray_pil = PIL.Image.fromarray(img2_gray)
            img2_tensor = self.data_transforms(img2_gray_pil).unsqueeze(0)

            with torch.no_grad():

                _, mask_1 = self.model.forward(img1_tensor, img2_tensor.to(self.device))

            if img1.ndim == 3:
                mask_1.repeat(1, 3, 1, 1)

            fused = img1_t * mask_1 + img2_t * (1 - mask_1)
            if img1.ndim == 3:
                R = fused[:, 0, :, :].unsqueeze(1)
                G = fused[:, 1, :, :].unsqueeze(1)
                B = fused[:, 2, :, :].unsqueeze(1)
                img1_tensor = 0.2125 * R + 0.7154 * G + 0.0721 * B
            else:
                img1_tensor = fused

            img1_t = fused

        if img1.ndim == 3:
            fused = fused.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        else:
            fused = fused.squeeze(0).squeeze(0).cpu().detach().numpy()
        return fused
    
    @staticmethod
    def channel_sf(f1, kernel_radius=5):
        """
        Calculate channel sf of deep feature
        """
        device = f1.device
        b, c, h, w = f1.shape
        r_shift_kernel = torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]])\
            .cuda(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
        b_shift_kernel = torch.FloatTensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]])\
            .cuda(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
        f1_r_shift = f.conv2d(f1, r_shift_kernel, padding=1, groups=c)
        f1_b_shift = f.conv2d(f1, b_shift_kernel, padding=1, groups=c)
        f1_grad = torch.pow((f1_r_shift - f1), 2) + torch.pow((f1_b_shift - f1), 2)
        kernel_size = kernel_radius * 2 + 1
        add_kernel = torch.ones((c, 1, kernel_size, kernel_size)).float().cuda(device)
        kernel_padding = kernel_size // 2
        f1_sf = f.conv2d(f1_grad, add_kernel, padding=kernel_padding, groups=c)
        return f1_sf


class GACNFuseNet(nn.Module):
    """
    The Class of SESFuseNet
    """
    def __init__(self):
        super(GACNFuseNet, self).__init__()
        
        # feature_extraction
        self.feature_extraction_conv0 = self.conv_block(1, 16, name="feature_extraction_conv0")
        self.se_0 = CSELayer(16, 8)
        self.feature_extraction_conv1 = self.conv_block(16, 16, name="feature_extraction_conv1")
        self.se_1 = CSELayer(16, 8)
        self.feature_extraction_conv2 = self.conv_block(32, 16, name="feature_extraction_conv2")
        self.se_2 = CSELayer(16, 8)
        self.feature_extraction_conv3 = self.conv_block(48, 16, name="feature_extraction_conv3")
        self.se_3 = CSELayer(16, 8)
        
        # decision_path
        self.se_4 = SSELayer(64)
        self.se_5 = SSELayer(48)
        self.se_6 = SSELayer(32)
        self.se_7 = SSELayer(16)
        self.se_8 = SSELayer(1)
        self.decision_path_conv1 = self.conv_block(64, 48, name="decision_path_conv1")
        self.decision_path_conv2 = self.conv_block(48, 32, name="decision_path_conv2")
        self.decision_path_conv3 = self.conv_block(32, 16, name="decision_path_conv3")
        self.decision_path_conv4 = self.conv_block(16, 1,  name="decision_path_conv4")
        self.guided_filter = GuidedFilter(3, 0.1)
        self.gaussian = GaussBlur(8, 4)
   
    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size=3, relu=True, batchnorm=True, name=None):
        """
        The conv block of common setting: conv -> relu -> bn
        In conv operation, the padding = 1
        :param in_channels: int, the input channels of feature
        :param out_channels: int, the output channels of feature
        :param kernel_size: int, the kernel size of feature
        :param relu: bool, whether use relu
        :param batchnorm: bool, whether use bn
        :param name: str, name of the conv_block
        :return:
        """
        block = torch.nn.Sequential()
        block.add_module(name+"_Conv2d", torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels,
                                                        out_channels=out_channels, padding=kernel_size // 2))
        if relu:
            block.add_module(name+"_ReLu", torch.nn.ReLU())
        if batchnorm:
            block.add_module(name+"_BatchNorm", torch.nn.BatchNorm2d(out_channels))
        return block
    
    @staticmethod
    def concat(f1, f2):
        """
        Concat two feature in channel direction
        """
        return torch.cat((f1, f2), 1)
    
    @staticmethod
    def fusion_channel_sf(f1, f2, kernel_radius=5):
        """
        Perform channel sf fusion two features
        """
        device = f1.device
        b, c, h, w = f1.shape
        r_shift_kernel = torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]])\
            .cuda(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
        b_shift_kernel = torch.FloatTensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]])\
            .cuda(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
        f1_r_shift = f.conv2d(f1, r_shift_kernel, padding=1, groups=c)
        f1_b_shift = f.conv2d(f1, b_shift_kernel, padding=1, groups=c)
        f2_r_shift = f.conv2d(f2, r_shift_kernel, padding=1, groups=c)
        f2_b_shift = f.conv2d(f2, b_shift_kernel, padding=1, groups=c)
        f1_grad = torch.pow((f1_r_shift - f1), 2) + torch.pow((f1_b_shift - f1), 2)
        f2_grad = torch.pow((f2_r_shift - f2), 2) + torch.pow((f2_b_shift - f2), 2)
        kernel_size = kernel_radius * 2 + 1
        add_kernel = torch.ones((c, 1, kernel_size, kernel_size)).float().cuda(device)
        kernel_padding = kernel_size // 2
        f1_sf = f.conv2d(f1_grad, add_kernel, padding=kernel_padding, groups=c)
        f2_sf = f.conv2d(f2_grad, add_kernel, padding=kernel_padding, groups=c)

        # get decision map
        bimap = torch.sigmoid(1000 * (f1_sf - f2_sf))
        return bimap
    
    def forward(self, img1, img2):
        """
        Train or Forward for two images
        :param img1: torch.Tensor
        :param img2: torch.Tensor
        :return: output, torch.Tensor
        """
        # Feature extraction c1
        feature_extraction_conv0_c1 = self.feature_extraction_conv0(img1)
        se_feature_extraction_conv0_c1 = self.se_0(feature_extraction_conv0_c1)
        feature_extraction_conv1_c1 = self.feature_extraction_conv1(se_feature_extraction_conv0_c1)
        se_feature_extraction_conv1_c1 = self.se_1(feature_extraction_conv1_c1)
        se_cat1_c1 = self.concat(se_feature_extraction_conv0_c1, se_feature_extraction_conv1_c1)
        feature_extraction_conv2_c1 = self.feature_extraction_conv2(se_cat1_c1)
        se_feature_extraction_conv2_c1 = self.se_2(feature_extraction_conv2_c1)
        se_cat2_c1 = self.concat(se_cat1_c1, se_feature_extraction_conv2_c1)
        feature_extraction_conv3_c1 = self.feature_extraction_conv3(se_cat2_c1)
        se_feature_extraction_conv3_c1 = self.se_3(feature_extraction_conv3_c1)
        
        # Feature extraction c2
        feature_extraction_conv0_c2 = self.feature_extraction_conv0(img2)
        se_feature_extraction_conv0_c2 = self.se_0(feature_extraction_conv0_c2)
        feature_extraction_conv1_c2 = self.feature_extraction_conv1(se_feature_extraction_conv0_c2)
        se_feature_extraction_conv1_c2 = self.se_1(feature_extraction_conv1_c2)
        se_cat1_c2 = self.concat(se_feature_extraction_conv0_c2, se_feature_extraction_conv1_c2)
        feature_extraction_conv2_c2 = self.feature_extraction_conv2(se_cat1_c2)
        se_feature_extraction_conv2_c2 = self.se_2(feature_extraction_conv2_c2)
        se_cat2_c2 = self.concat(se_cat1_c2, se_feature_extraction_conv2_c2)
        feature_extraction_conv3_c2 = self.feature_extraction_conv3(se_cat2_c2)
        se_feature_extraction_conv3_c2 = self.se_3(feature_extraction_conv3_c2)

        # SF fusion
        cat_1 = torch.cat((se_feature_extraction_conv0_c1, se_feature_extraction_conv1_c1, 
                           se_feature_extraction_conv2_c1, se_feature_extraction_conv3_c1), axis=1)
        cat_2 = torch.cat((se_feature_extraction_conv0_c2, se_feature_extraction_conv1_c2, 
                           se_feature_extraction_conv2_c2, se_feature_extraction_conv3_c2), axis=1)
        fused_cat = self.fusion_channel_sf(cat_1, cat_2, kernel_radius=5)
        se_f = self.se_4(fused_cat)

        # Decision path
        decision_path_conv1 = self.decision_path_conv1(se_f)
        se_decision_path_conv1 = self.se_5(decision_path_conv1)
        decision_path_conv2 = self.decision_path_conv2(se_decision_path_conv1)
        se_decision_path_conv2 = self.se_6(decision_path_conv2)
        decision_path_conv3 = self.decision_path_conv3(se_decision_path_conv2)
        se_decision_path_conv3 = self.se_7(decision_path_conv3)
        decision_path_conv4 = self.decision_path_conv4(se_decision_path_conv3)
        se_decision_path_conv4 = self.se_8(decision_path_conv4)
        
        # Boundary guided filter
        output_origin = torch.sigmoid(1000 * se_decision_path_conv4)
        output_blur = self.gaussian(output_origin)
        zeros = torch.zeros_like(output_blur)
        ones = torch.ones_like(output_blur)
        half = ones / 2
        mask_1 = torch.where(output_blur > 0.8, ones, zeros)
        mask_2 = torch.where(output_blur < 0.1, ones, zeros)
        mask_3 = mask_1 * output_blur + mask_2 * (1 - output_blur)
        boundary_map = 1 - torch.abs(2 * (output_blur * mask_3 + (1 - mask_3) * half) - 1)
        temp_fused = img1 * output_origin + (1 - output_origin) * img2
        output_gf = self.guided_filter(temp_fused, output_origin)
        output_bgf = output_gf * boundary_map + output_origin * (1 - boundary_map)
        return output_origin, output_bgf


class SSELayer(nn.Module):
    def __init__(self, channel):
        super(SSELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=7, bias=False, padding=[3, 3]),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        y = self.fc(x) 
        return x * y


class CSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GACNFeatureExtraction(nn.Module):
    def __init__(self):
        super(GACNFeatureExtraction, self).__init__()
        self.feature_extraction_conv0 = self.conv_block(1, 16, name="feature_extraction_conv0")
        self.se_0 = CSELayer(16, 8)
        self.feature_extraction_conv1 = self.conv_block(16, 16, name="feature_extraction_conv1")
        self.se_1 = CSELayer(16, 8)
        self.feature_extraction_conv2 = self.conv_block(32, 16, name="feature_extraction_conv2")
        self.se_2 = CSELayer(16, 8)
        self.feature_extraction_conv3 = self.conv_block(48, 16, name="feature_extraction_conv3")
        self.se_3 = CSELayer(16, 8)
    
    def forward(self, img):
        # feature_extraction
        feature_extraction_conv0 = self.feature_extraction_conv0(img)
        se_feature_extraction_conv0 = self.se_0(feature_extraction_conv0)
        feature_extraction_conv1 = self.feature_extraction_conv1(se_feature_extraction_conv0)
        se_feature_extraction_conv1 = self.se_1(feature_extraction_conv1)
        se_cat1 = self.concat(se_feature_extraction_conv0, se_feature_extraction_conv1)
        feature_extraction_conv2 = self.feature_extraction_conv2(se_cat1)
        se_feature_extraction_conv2 = self.se_2(feature_extraction_conv2)
        se_cat2 = self.concat(se_cat1, se_feature_extraction_conv2)
        feature_extraction_conv3 = self.feature_extraction_conv3(se_cat2)
        se_feature_extraction_conv3 = self.se_3(feature_extraction_conv3)
        se_cat = torch.cat((se_feature_extraction_conv0, se_feature_extraction_conv1, 
                           se_feature_extraction_conv2, se_feature_extraction_conv3), axis = 1)
        return se_cat
    
    @staticmethod
    def concat(f1, f2):
        """
        Concat two feature in channel direction
        """
        return torch.cat((f1, f2), 1)
    
    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size=3, relu=True, batchnorm=True, name=None):
        """
        The conv block of common setting: conv -> relu -> bn
        In conv operation, the padding = 1
        :param in_channels: int, the input channels of feature
        :param out_channels: int, the output channels of feature
        :param kernel_size: int, the kernel size of feature
        :param relu: bool, whether use relu
        :param batchnorm: bool, whether use bn
        :param name: str, name of the conv_block
        :return:
        """
        block = torch.nn.Sequential()
        block.add_module(name+"_Conv2d", torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels,
                                                         out_channels=out_channels, padding=kernel_size // 2))
        if relu:
            block.add_module(name+"_ReLu", torch.nn.ReLU())
        if batchnorm:
            block.add_module(name+"_BatchNorm", torch.nn.BatchNorm2d(out_channels))
        return block


class GACNDecisionPath(nn.Module):
    def __init__(self):
        super(GACNDecisionPath, self).__init__()
        # decision_path
        self.se_4 = SSELayer(64)
        self.se_5 = SSELayer(48)
        self.se_6 = SSELayer(32)
        self.se_7 = SSELayer(16)
        self.se_8 = SSELayer(1)
        self.decision_path_conv1 = self.conv_block(64, 48, name="decision_path_conv1")
        self.decision_path_conv2 = self.conv_block(48, 32, name="decision_path_conv2")
        self.decision_path_conv3 = self.conv_block(32, 16, name="decision_path_conv3")
        self.decision_path_conv4 = self.conv_block(16, 1,  name="decision_path_conv4")
        self.guided_filter = GuidedFilter(3, 0.1)
        self.gaussian = GaussBlur(8, 4)
        
    def forward(self, fused_cat, img1, img2):
        
        se_f = self.se_4(fused_cat)
        decision_path_conv1 = self.decision_path_conv1(se_f)
        se_decision_path_conv1 = self.se_5(decision_path_conv1)
        decision_path_conv2 = self.decision_path_conv2(se_decision_path_conv1)
        se_decision_path_conv2 = self.se_6(decision_path_conv2)
        decision_path_conv3 = self.decision_path_conv3(se_decision_path_conv2)
        se_decision_path_conv3 = self.se_7(decision_path_conv3)
        decision_path_conv4 = self.decision_path_conv4(se_decision_path_conv3)
        se_decision_path_conv4 = self.se_8(decision_path_conv4)
        
        # Boundary guided filter
        output_origin = torch.sigmoid(1000*(se_decision_path_conv4))    
        output_blur = self.gaussian(output_origin)
        zeros = torch.zeros_like(output_blur)
        ones = torch.ones_like(output_blur)
        half = ones/2
        mask_1 = torch.where(output_blur > 0.8, ones, zeros)
        mask_2 = torch.where(output_blur < 0.1, ones, zeros)
        mask_3 = mask_1 * output_blur+mask_2 * (1 - output_blur)
        boundary_map = 1 - torch.abs(2 * (output_blur * mask_3 + (1 - mask_3) * half)-1)
        temp_fused = img1 * output_origin + (1 - output_origin) * img2
        output_gf = self.guided_filter(temp_fused, output_origin)
        output_bgf = output_gf * boundary_map + output_origin * (1 - boundary_map)
        return output_origin, output_bgf
    
    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size=3, relu=True, batchnorm=True, name=None):
        """
        The conv block of common setting: conv -> relu -> bn
        In conv operation, the padding = 1
        :param in_channels: int, the input channels of feature
        :param out_channels: int, the output channels of feature
        :param kernel_size: int, the kernel size of feature
        :param relu: bool, whether use relu
        :param batchnorm: bool, whether use bn
        :param name: str, name of the conv_block
        :return:
        """
        block = torch.nn.Sequential()
        block.add_module(name+"_Conv2d", torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels,
                                                         out_channels=out_channels, padding=kernel_size // 2))
        if relu:
            block.add_module(name+"_ReLu", torch.nn.ReLU())
        if batchnorm:
            block.add_module(name+"_BatchNorm", torch.nn.BatchNorm2d(out_channels))
        return block
