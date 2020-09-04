import os
import cv2
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch
import random

def training_setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer, learning_rate, epoch):
    """Sets the learning rate to the initial LR decayed by half every 10 epochs until 1e-5"""
    lr = learning_rate * (0.8 ** (epoch // 2))
    #     if not lr < 1e-6:
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def plot_loss(experiment_name, epoch, train_loss_list, val_loss_list):
    clear_output(True)
    print('Epoch %s. train loss: %s. val loss: %s' % (epoch, train_loss_list[-1], val_loss_list[-1]))
    print('Best val loss: %s' % (min(val_loss_list)))
    print('Back up')
    print('train_loss_list: {}'.format(train_loss_list))
    print('val_loss_list: {}'.format(val_loss_list))
    plt.figure()
    plt.plot(train_loss_list, color="r", label="train loss")
    plt.plot(val_loss_list, color="b", label="val loss")
    plt.legend(loc="best")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss " + experiment_name, fontsize=16)
    figure_address = os.path.join(os.path.join(os.getcwd(), 'nets'), 'figures')
    plt.savefig(os.path.join(figure_address, experiment_name + '_loss.png'))
    plt.show()


def plot_iteration_loss(experiment_name, epoach, loss , qg_loss,dice_loss):
    
    plt.figure()
    plt.plot(loss, color="r", label="loss")
    plt.plot(qg_loss, color="g", label="qg_loss")
    plt.plot(dice_loss, color="b", label="dice_loss")
    plt.legend(loc="best")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss " + experiment_name, fontsize=16)
    figure_address = os.path.join(os.path.join(os.getcwd(), 'nets'), 'figures')
    plt.savefig(os.path.join(figure_address, experiment_name + '_' + str(epoach) + '_loss.png'))
    plt.show()


def print_and_log(content, is_out_log_file=True, file_address=None):
    print(content)
    if is_out_log_file:
        f = open(os.path.join(file_address), "a")
        f.write(content)
        f.write("\n")
        f.close()


def get_mean_value(input_dir):
    images_list = [os.path.join(input_dir, item) for item in sorted(os.listdir(input_dir))]
    count = 0
    pixel_sum = 0
    for index, sub_folder in enumerate(images_list):
        image_name = os.path.basename(sub_folder)
        last_image = cv2.imread(os.path.join(sub_folder, image_name + "_1.png"), 0) * 1.0 / 255
        next_image = cv2.imread(os.path.join(sub_folder, image_name + "_2.png"), 0) * 1.0 / 255
        pixel_sum = pixel_sum + np.sum(last_image) + np.sum(next_image)
        count = count + last_image.size + next_image.size
    return pixel_sum / count


def get_std_value(input_dir, mean):
    images_list = [os.path.join(input_dir, item) for item in sorted(os.listdir(input_dir))]
    count = 0
    pixel_sum = 0
    for index, sub_folder in enumerate(images_list):
        image_name = os.path.basename(sub_folder)
        last_image = np.power((cv2.imread(os.path.join(sub_folder, image_name + "_1.png"), 0) * 1.0 / 255) - mean, 2)
        next_image = np.power((cv2.imread(os.path.join(sub_folder, image_name + "_2.png"), 0) * 1.0 / 255) - mean, 2)
        pixel_sum = pixel_sum + np.sum(last_image) + np.sum(next_image)
        count = count + last_image.size + next_image.size
    return np.sqrt(pixel_sum / count)

def random_blured(img, mask, filtersize = 0, if_reverse = False, rate = 0.1, num = 1):
    """
    generate multi-focus image pairs with input origin image
    and ground-truth segmentation mask
    :param img: tensor, the input origin image
    :param mask: tensor, the ground-truth segmentation mask
    :param filtersize: int, the filtersize of the gaussion filter, when set to 0 means use random blur stratage
    :param argument: bool, whether use the random reverse stratge  which randomlly blur some clear area  or make some blur area clear
    :param rate: int, the ratio of reversed area to image area in random reverse stratge
    :param num: int, the quantity of reversed area in random reverse stratge
    :return:
    """
    b,_,w,h = img.shape
    deviation = 2
    if filtersize == 0:
        filtersize = np.random.randint(1,8)

    img_blured= gauss_blur(img,deviation,filtersize)  
    mask[mask!=0]=1
    result_o_blured = mask*img+(1-mask)*img_blured
    result_b_blured = (1-mask)*img+mask*img_blured
    mask2_o = gauss_blur(mask,8,45)

    if(if_reverse):
        thresh = 0.9
        result_b_blured_g = torch.zeros_like(img).to(img.device)
        result_o_blured_g = torch.zeros_like(img).to(img.device)
        mask_pool = torch.nn.AdaptiveAvgPool2d((int(rate*w),int(rate*h)))(mask)
        mask_pool2 = torch.nn.AdaptiveAvgPool2d((int(rate*w),int(rate*h)))(mask2_o)
        for i in range(b):
            mask_pool_sub = mask_pool[i,0,:,:]
            mask_pool_sub2 = mask_pool2[i,0,:,:]
            threshhold = torch.max(mask_pool_sub).item()*thresh
            threshhold2 = np.max(torch.min(mask_pool_sub2).item(),0)
            
            for j in range(num):
                m = (mask_pool_sub == mask_pool_sub[mask_pool_sub>=threshhold][np.random.randint(0,len(mask_pool_sub[mask_pool_sub>=threshhold]))])
                m2 = (mask_pool_sub2 == mask_pool_sub2[mask_pool_sub2<=threshhold2][np.random.randint(0,len(mask_pool_sub2[mask_pool_sub2<=threshhold2]))])
                if(len(m.nonzero())>1):
                    x,y = (m.nonzero())[np.random.randint(0,len(m.nonzero())-1)]*int(1/rate)
                else:
                    x,y = (m.nonzero())[0]*int(1/rate)
                if(len(m2.nonzero())>1):   
                    x2,y2 = (m2.nonzero())[np.random.randint(0,len(m2.nonzero())-1)]*int(1/rate)
                else:
                    x2,y2 = (m2.nonzero())[0]*int(1/rate)
                mask[i,0,x:min(x+int(1/rate),w-1),y:min(y+int(1/rate),h-1)] = 0
                mask2_o[i,0,x2:min(x2+int(1/rate),w-1),y2:min(y2+int(1/rate),h-1)] = 1

            result_b_blured_g[i,0,:,:] = mask[i,0,:,:]*img[i,0,:,:]+(1-mask[i,0,:,:])*img_blured[i,0,:,:]
            result_o_blured_g[i,0,:,:]  = img[i,0,:,:]*(1-mask2_o[i,0,:,:]) + mask2_o[i,0,:,:]*img_blured[i,0,:,:]
    else:
        result_o_blured_g = img*(1-mask2_o) + mask2_o*img_blured
        result_b_blured_g = img*mask +(1-mask)*img_blured
    return result_o_blured_g, result_b_blured_g

class GaussBlur(nn.Module):
    """
     gaussion blur
    """
    def __init__(self, sigma, filtersize):
        super(GaussBlur, self).__init__()
        self.radius = filtersize
        sigma2 = sigma ** 2
        sum_val = 0
        x = torch.tensor(np.arange(-self.radius, self.radius + 1), dtype=torch.float).expand(1, 2 * self.radius + 1)
        y = x.t().expand(2 * self.radius + 1, 2 * self.radius + 1)
        x = x.expand(2 * self.radius + 1, 2 * self.radius + 1)
        self.kernel = torch.exp(-(torch.mul(x, x) + torch.mul(y, y)) / (2 * sigma2))
        self.kernel = self.kernel / torch.sum(self.kernel)
        self.weight = self.kernel.expand(1, 1, 2 * self.radius + 1, 2 * self.radius + 1)

    def forward(self, data):
        _, c, _, _ = data.shape
        self.weight = self.weight.expand(c, 1, 9, 9)
        if str(self.weight.device) != str(data.device):
            self.weight = self.weight.to(data.device)
        blured = F.conv2d(data, self.weight, padding=[self.radius], groups=c)
        return blured

def gauss_blur(data, sigma, filtersize):
    """
     gaussion blur
    :param data: tensor, the input image
    :param sigma: int, the standard deviation of gaussion filter
    :param filtersize: int, the filtersize of the gaussion filter
    :return:
    """
    _, c, _, _ = data.shape
    radius = filtersize
    sigma2 = sigma**2
    sum_val =  0
    x = torch.tensor(np.arange(-radius, radius+1),dtype=torch.float).expand(1,2*radius+1)
    y = x.t().expand(2*radius+1,2*radius+1)
    x = x.expand(2*radius+1,2*radius+1)
    kernel = torch.exp(-(torch.mul(x,x)+torch.mul(y,y))/(2*sigma2))  
    kernel = kernel/torch.sum(kernel)
    weight = nn.Parameter(data=kernel,requires_grad=False).expand(c,1,2*radius+1,2*radius+1).to(data.device)
    blured = F.conv2d(data, weight, padding=[radius], groups = c)
    return blured

def random_erasing(data_1, data_2, num = 1, min = 20, max = 40):
    """
     data augment using random erasing
    :param data_1: tensor, the input image A
    :param data_2: tensor, the input image B
    :param num: int, the quantity of erased area
    :param min: int, the minimum size of the erased area
    :param nax: int, the maximum size of thw erased area
    :return:
    """
    b,c,w,h = data_1.shape
    num = np.random.randint(num/2,num,b)
    final_mask = []
    for i in range(b):
        erasing_mask = np.ones([c,w,h]).astype(np.float32)
        for j in range(num[i]):
            w_1 = np.random.randint(0,w-min-max)
            w_2 = np.random.randint(w_1+min,w_1+min+max)
            h_1 = np.random.randint(0,h-min-max)
            h_2 = np.random.randint(h_1+min,h_1+min+max)
            erasing_mask[:,w_1:w_2,h_1:h_2] = 0.0
        final_mask.append(erasing_mask)
    final_mask_tensor = torch.tensor(final_mask).to(data_1.device)
    if np.random.rand()>0.5:
        data_1_o = torch.mul(data_1, final_mask_tensor)
        data_2_o = data_2
    else:
        data_1_o = data_1
        data_2_o = torch.mul(data_2, final_mask_tensor)
    return data_1_o, data_2_o

def random_offset(data_1, data_2, w_offset, h_offset):
    """
     data augment using random offset. Randomly offset one of the 
     input images in the w and h direction with a probability of 0.5
    :param data_1: tensor, the input image A
    :param data_2: tensor, the input image B
    :param w_offset: int, maximum offset in w direction
    :param h_offset: int, maximum offset in h direction
    :return:
    """
    b,c,w,h = data_1.shape
    result_1 = torch.zeros(b,c,w,h).to(data_1.device)
    result_2 = torch.zeros(b,c,w,h).to(data_1.device)
    for i in range(b):
        if(random.random()<0.5):
            x = random.randint(-w_offset, w_offset)
            y = random.randint(-h_offset, h_offset)
            if(random.random()<0.5):
                result_1[i,:,np.max((0,x)):np.min((w-1,w+x-1)),np.max((0,y)):np.min((h-1,h+y-1))] = data_1[i,:,np.max((0,-x)):np.min((w-1,w-1-x)),np.max((0,-y)):np.min((h-1,h-1-y))]
                result_2[i,:,:,:] = data_2[i,:,:,:] 
            else:
                result_1[i,:,:,:]  = data_1[i,:,:,:] 
                result_2[i,:,np.max((0,x)):np.min((w-1,w+x-1)),np.max((0,y)):np.min((h-1,h+y-1))]  = data_2[i,:,np.max((0,-x)):np.min((w-1,w-1-x)),np.max((0,-y)):np.min((h-1,h-1-y))]
        else:
            result_1[i,:,:,:] = data_1[i,:,:,:] 
            result_2[i,:,:,:] = data_2[i,:,:,:] 
    return result_1,result_2



def gaussion_noise(data_1, data_2, std = 0.05, rate = 0.05):
    """
     data augment using gaussion noise. 
    :param data_1: tensor, the input image A
    :param data_2: tensor, the input image B
    :param std: int, standered deviation of gaussion noise
    :param rate: int, the noise to signal ratio
    :return:
    """
    b,c,w,h = data_1.shape
    std = std.to(data_1.device)
    rate = torch.FloatTensor([rate]).to(data_1.device)
    mean_ = torch.zeros_like(data_1)
    std_ = torch.ones_like(data_1)*std
    gaussion_mask = torch.normal(mean = mean_, std = std_)
    data_1_n = data_1*(1-rate)+gaussion_mask*rate
    data_2_n = data_2*(1-rate)+gaussion_mask*rate
    return data_1_n, data_2_n
    
        
            
    
    
    
