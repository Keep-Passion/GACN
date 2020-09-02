import os
import time
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from nets.coco_dataset import COCODataset
from nets.gacn_net import GACNFuseNet
from nets.nets_utility import *
import torch.nn as nn
from nets.ga_loss import GALoss
import time
import skimage
# parameter for net
# name
experiment_name = 'GACN'
gpu_device = "cuda:0"
# gpu_device_for_parallel = [2, 3]
learning_rate = 1e-4
epochs = 50
batch_size = 16
display_step = 50
shuffle = True
#Sets the lambda_qg increased by half every 10 epochs "
rate = 1.0414


# address
project_addrsss = os.getcwd()
train_dir = os.path.join(project_addrsss, "data", "mydata", "train")
val_dir = os.path.join(project_addrsss, "data", "mydata", "val")
mask_t_dir = os.path.join(project_addrsss, "data", "mydata", "mask_train")
mask_v_dir = os.path.join(project_addrsss, "data", "mydata", "mask_val")
out_dir = os.path.join(project_addrsss, "nets", "out")
log_address = os.path.join(project_addrsss, "nets", "train_record", experiment_name + "_log_file.txt")
is_out_log_file = True
parameter_address = os.path.join(project_addrsss, "nets", "parameters")
print(experiment_name)
# datasets
data_transforms = transforms.Compose([
    transforms.ToTensor(),
])

image_datasets = {}
image_datasets['train_data'] = COCODataset(train_dir, mask_t_dir, transform = data_transforms, need_crop=True, need_rotate=True,need_filp=True)
image_datasets['val_data'] = COCODataset(val_dir,  mask_v_dir, transform = data_transforms)
dataloders = {}
dataloders['train'] = DataLoader(
    image_datasets['train_data'],
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=1)
dataloders['val'] = DataLoader(
    image_datasets['val_data'],
    batch_size=batch_size,
    shuffle=False,
    num_workers=1)
datasets_sizes = {x: len(image_datasets[x]) for x in ['train_data','val_data']}
print_and_log("datasets size: {}".format(datasets_sizes), is_out_log_file, log_address)
# models
training_setup_seed(1)
model = GACNFuseNet()
model.to(gpu_device)
criterion = GALoss().to(gpu_device)
optimizer = optim.Adam(model.parameters(), learning_rate)

# fix the filter size in random blur augment during validation process
filtersizes = np.random.randint(1,8,(np.ceil(datasets_sizes['val_data']/batch_size).astype(np.int)))

def val(epoch):
    model.eval()
    running_loss = 0.0
    qg_loss = 0.00
    dice_loss = 0.00
    with torch.no_grad():
        out = []
        for i, data in enumerate(dataloders['val']):
            gt_mask = data[1].to(gpu_device)
            input_1, input_2 = random_blured(data[0].to(gpu_device), gt_mask, filtersize = filtersizes[i])    
            optimizer.zero_grad()
            mask, mask_BGF = model.forward(input_2, input_1)
            loss, dice, qg = criterion(input_2, input_1, mask, mask_BGF, gt_mask)
            running_loss += loss.item()
            qg_loss +=qg.item()
            dice_loss += dice.item()
            index_1 = np.random.randint(int(len(image_datasets["val_data"])/16) , size=3)
        
    epoch_loss_val = running_loss / datasets_sizes['val_data'] * batch_size
    epoch_qg_val = qg_loss / datasets_sizes['val_data'] * batch_size
    epoch_dice_val = dice_loss / datasets_sizes['val_data'] * batch_size
    return epoch_loss_val, epoch_qg_val, epoch_dice_val

def train(epoch):
    iterations_loss_list = []
    iterations_qg_list = []
    iterations_dice_list = []
    model.train()
    adjust_learning_rate(optimizer, learning_rate, epoch)
    print_and_log('Train Epoch {}/{}:'.format(epoch + 1, epochs), is_out_log_file, log_address)
    running_loss = 0.0
    # Iterate over data.
    out = []
    for i, data in enumerate(dataloders['train']):
        gt_mask = data[1].to(gpu_device)
        ori_img = data[0].to(gpu_device)
        
        # data augment
        # random blur
        input_1, input_2 = random_blured(ori_img, gt_mask)
        # random erasing
        if np.random.rand()>99:
            input_1,input_2 = random_erasing(input_1, input_2, 6, 15, 20)
        # random offset
        input_1,input_2 = random_offset(input_1,input_2,2,2)
        # gaussion noise
        std = torch.rand(1)*0.1
        input_1, input_2 = gaussion_noise(input_1, input_2,std)
        
        # swap input order randomly
        flag = np.random.rand()
        if(flag>0.5):
            output, output_2 = model.forward(input_2,input_1)
            loss, dice, qg = criterion(input_2, input_1, output, output_2, data[1].to(gpu_device))
        else:
            output, output_2 = model.forward(input_1,input_2)
            loss, dice, qg = criterion(input_1, input_2, output, output_2, 1-data[1].to(gpu_device))
        
        # display
        running_loss += loss.item()
        if i % display_step == 0:
            print_and_log('\t{} {}-{}: Loss: {:.4f} ,qg: {:.4f},dice: {:.4f}'.format('train', epoch + 1, i, loss.item(),qg.item(),dice.item()),
                          is_out_log_file, log_address)
            iterations_loss_list.append(loss.item())
            iterations_qg_list.append(qg.item())
            iterations_dice_list.append(dice.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    epoch_loss_train = running_loss / datasets_sizes['train_data'] * batch_size
    plot_iteration_loss(experiment_name, epoch + 1, iterations_loss_list, iterations_qg_list, iterations_dice_list)
    return epoch_loss_train


def main():
    min_loss = 100000000.0
    loss_train = []
    loss_val = []
    since = time.time()
    for epoch in range(epochs):
        epoch_loss_train = train(epoch)
        loss_train.append(epoch_loss_train)
        epoch_loss_val,epoch_qg_val,epoch_dice_val= val(epoch)
        loss_val.append(epoch_loss_val)
        print_and_log('\ttrain Loss: {:.6f}'.format(epoch_loss_train), is_out_log_file, log_address)
        print_and_log('\tvalidation Loss: {:.6f}, Qg: {:.6f}, dice: {:.6f}'.format(epoch_loss_val,epoch_qg_val,epoch_dice_val), is_out_log_file, log_address)

        # deep copy the models
        if epoch_loss_val  < min_loss:
            print(epoch_loss_val)
            min_loss = epoch_loss_val
            best_model_wts = model.state_dict()
            print_and_log("Updating", is_out_log_file, log_address)
            torch.save(best_model_wts,
                       os.path.join(parameter_address, experiment_name +'.pkl'))
        plot_loss(experiment_name, epoch, loss_train, loss_val)
        # save models
        model_wts = model.state_dict()
        torch.save(model_wts,
                   os.path.join(parameter_address, experiment_name + '_'+str(epoch)+'.pkl'))

        time_elapsed = time.time() - since
        print_and_log('Time passed {:.0f}h {:.0f}m {:.0f}s'.
                      format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60), is_out_log_file,
                      log_address)
        print_and_log('-' * 20, is_out_log_file, log_address)
    print_and_log("train loss: {}".format(loss_train), is_out_log_file, log_address)
    print_and_log("val loss: {}".format(loss_val), is_out_log_file, log_address)
    print_and_log("min val loss: {}".format(min(loss_val)), is_out_log_file, log_address)
if __name__ =="__main__":
    main()