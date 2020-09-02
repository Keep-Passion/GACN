import os
import random
import cv2
import numpy as np
import PIL.Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from nets.nets_utility import *
import torchvision.transforms.functional as F

class COCODataset(Dataset):
    def __init__(self, data_dir, mask_dir, crop_size = 156, transform = None, need_crop = False, need_rotate = False, need_filp = False):
        
        self._images_basename = os.listdir(data_dir)
        if '.ipynb_checkpoints' in self._images_basename:
            self._images_basename.remove('.ipynb_checkpoints')
            
        self._data_address = [os.path.join(data_dir, item) for item in sorted(self._images_basename)]
        self._mask_address = [os.path.join(mask_dir, item) for item in sorted(self._images_basename)]
        self._crop_size = crop_size
        self._transform = transform
        self._origin_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self._need_rotate = need_rotate
        self._need_crop = need_crop
        self._need_flip = need_filp

    def __len__(self):
        return len(self._data_address)

    def __getitem__(self, idx):
        data = cv2.imread(self._data_address[idx], 0) / 255.0
        data = cv2.resize(data, (256, 256))  
        mask = cv2.imread(self._mask_address[idx], 0) / 255.0
        mask = cv2.resize(mask, (256, 256))  
        
        #random crop
        roi_image_np_1, roi_image_np_2 = self._random_crop(data, mask) 
        #random filp
        roi_image_pil_1, roi_image_pil_2 = self._rand_filp(roi_image_np_1, roi_image_np_2)
        #random rotated
        roi_image_pil_1, roi_image_pil_2 = self._rand_rotate(roi_image_pil_1, roi_image_pil_2)
        
        #transform    
        if self._transform is not None:
            roi_image_tensor_1 = self._transform(roi_image_pil_1)
            roi_image_tensor_2 = self._origin_transform(roi_image_pil_2)
        else:
            roi_image_tensor_1 = self._origin_transform(roi_image_pil_1)
            roi_image_tensor_2 = self._origin_transform(roi_image_pil_2)
        
        roi_image_tensor_2[roi_image_tensor_2<0.5] = 0
        roi_image_tensor_2[roi_image_tensor_2>0.5] = 1
        
        return roi_image_tensor_1, roi_image_tensor_2

    def _rand_filp(self, image_1, image_2):
        """
        random filp
        :param image_1: array, input image
        :param image_2: array, input mask
        :return:
        """
        image_pil_1 = PIL.Image.fromarray(image_1.astype(np.float32))
        image_pil_2 = PIL.Image.fromarray(image_2.astype(np.float32))
        if self._need_flip:

            image_pil_1, image_pil_2 = self._rand_horizontal_flip(image_pil_1, image_pil_2)
            image_pil_1, image_pil_2 = self._rand_vertical_flip(image_pil_1, image_pil_2)

        return image_pil_1,  image_pil_2

    def _rand_rotate(self, image_pil_1, image_pil_2):
        """
        random rotate
        :param image_pil_1: PIL.Image, input image
        :param image_pil_2: PIL.Image, input mask
        :return:
        """
        if self._need_rotate:
            rotate_angle = random.choice([0, 30,60,90,120,150,180,210,240,270,300,330])
            image_pil_1 = image_pil_1.rotate(rotate_angle)
            image_pil_2 = image_pil_2.rotate(rotate_angle)
        return image_pil_1, image_pil_2

    def _rand_horizontal_flip(self, image_pil_1, image_pil_2):
        """
        Randomly flipped horizontally with a probability of 0.5
        :param image_pil_1: PIL.Image, input image
        :param image_pil_2: PIL.Image, input mask
        :return:
        """
        if random.random() < 0.5:
            image_pil_1 = F.hflip(image_pil_1)
            image_pil_2 = F.hflip(image_pil_2)
        return image_pil_1, image_pil_2
    
    def _rand_vertical_flip(self, image_pil_1, image_pil_2):
        """
        Randomly flipped vertically with a probability of 0.5
        :param image_pil_1: PIL.Image, input image
        :param image_pil_2: PIL.Image, input mask
        :return:
        """
        if random.random() < 0.5:
            image_pil_1 = F.vflip(image_pil_1)
            image_pil_2 = F.vflip(image_pil_2)
        return image_pil_1, image_pil_2

    def _random_crop(self, image, mask):
        """
        random crop
        :param image: array, input image
        :param mask: array, input mask
        :return:
        """
        if self._need_crop:
            h, w = image.shape[:2]
            start_row = random.randint(0, h - self._crop_size)
            start_col = random.randint(0, w - self._crop_size)
            image = image[start_row: start_row + self._crop_size, start_col: start_col + self._crop_size]
            mask = mask[start_row: start_row + self._crop_size, start_col: start_col + self._crop_size]
        return image, mask
   
    