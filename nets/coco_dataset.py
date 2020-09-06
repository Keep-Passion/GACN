import PIL.Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from nets.nets_utility import *
import torchvision.transforms.functional as F


class COCODataset(Dataset):
    def __init__(self, data_dir, mask_dir, crop_size=156,
                 transform=None, need_crop=False, need_rotate=False, need_flip=False):
        
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
        self._need_flip = need_flip

    def __len__(self):
        return len(self._data_address)

    def __getitem__(self, idx):
        data = cv2.imread(self._data_address[idx], 0) / 255.0
        data = cv2.resize(data, (256, 256))  
        mask = cv2.imread(self._mask_address[idx], 0) / 255.0
        mask = cv2.resize(mask, (256, 256))  
        
        # random crop
        roi_image_np, roi_mask_np = self._random_crop(data, mask)
        # random flip
        roi_image_pil, roi_mask_pil = self._rand_flip(roi_image_np, roi_mask_np)
        # random rotated
        roi_image_pil, roi_mask_pil = self._rand_rotate(roi_image_pil, roi_mask_pil)
        
        # transform
        if self._transform is not None:
            roi_image_tensor = self._transform(roi_image_pil)
            roi_mask_tensor = self._origin_transform(roi_mask_pil)
        else:
            roi_image_tensor = self._origin_transform(roi_image_pil)
            roi_mask_tensor = self._origin_transform(roi_mask_pil)
        
        roi_mask_tensor[roi_mask_tensor < 0.5] = 0
        roi_mask_tensor[roi_mask_tensor > 0.5] = 1
        
        return roi_image_tensor, roi_mask_tensor

    def _rand_flip(self, image_1, image_2):
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
            rotate_angle = random.choice([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
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


class DataAugment():
    """
    Implement data augment
    """
    def __init__(self, dataloader, random_blur=True, random_erasing=True, random_offset=True, gaussian_noise=True, swap=True, filter_sizes=None, device = 'cpu'):
        self.dataloader = dataloader
        self.random_blur = random_blur
        self.random_erasing = random_erasing
        self.random_offset = random_offset
        self.gaussian_noise = gaussian_noise
        self.swap = swap
        self.filter_sizes = filter_sizes
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            input_mask = data[1]
            # data augment
            # random blur
            if self.random_blur:
                input_img_1, input_img_2 = random_blurred(data[0].to(self.device), data[1].to(self.device))
            else:
                input_img_1, input_img_2 = random_blurred(data[0].to(self.device), data[1].to(self.device), filter_size=self.filter_sizes[i])
            # random erasing
            if self.random_erasing:
                if np.random.rand() > 99:
                    input_img_1, input_img_2 = random_erasing(input_img_1, input_img_2, 6, 15, 20)
            # random offset
            if self.random_offset:
                input_img_1, input_img_2 = random_offset(input_img_1, input_img_2, 2, 2)
            # gaussian noise
            if self.gaussian_noise:
                std = torch.rand(1) * 0.1
                input_img_1, input_img_2 = gaussian_noise(input_img_1, input_img_2, std)

            # swap input order randomly
            if self.swap:
                flag = np.random.rand()
                if flag <= 0.5:
                    input_mask = 1 - input_mask
                    input_img_1, input_img_2 = input_img_2, input_img_1
            yield input_img_1, input_img_2, input_mask
