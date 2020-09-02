import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class GALoss(nn.Module):
    """
    The Class of GALoss
    """
    def __init__(self):
        super(GALoss, self).__init__()
        self._smooth = 1
        
    def _dice_loss(self, predict, target):
        """
        Compute the dice loss of the prediction decision map and ground-truth label
        :param predict: tensor, the prediction decision map
        :param target: tensor, ground-truth label
        :return:
        """
        target = target.float()
        intersect = torch.sum(predict * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(predict * predict)
        loss = (2 * intersect + self._smooth) / (z_sum + y_sum + self._smooth)
        loss = 1 - loss
        return loss
   
    def _qg_soft(self, img1, img2, fuse, k):
        """
        Compute the Qg for the given two image and the fused image
        :param img1: tensor, input image A
        :param img2: tensor, input image B
        :param fuse: tensor, fused image
        :param k: softening factor 
        :return:
        """
        #1) get the map
        img1_gray = img1
        img2_gray = img2
        buf = 0.000001
        flt1 = torch.FloatTensor(np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1], ])).reshape((1, 1, 3, 3)).cuda(img1.device)
        flt2 = torch.FloatTensor(np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1], ])).reshape((1, 1, 3, 3)).cuda(img1.device)
        fuseX = F.conv2d(fuse, flt1, padding=1)+buf
        fuseY = F.conv2d(fuse, flt2, padding=1)
        fuseG = torch.sqrt(torch.mul(fuseX, fuseX)+torch.mul(fuseY, fuseY))
        buffer = (fuseX == 0)
        buffer = buffer.float()
        buffer = buffer*buf
        fuseX = fuseX+buffer
        fuseA = torch.atan( torch.div(fuseY, fuseX))
        
        img1X = F.conv2d(img1_gray,flt1,padding=1)
        img1Y = F.conv2d(img1_gray,flt2,padding=1)
        img1G = torch.sqrt(torch.mul(img1X, img1X)+torch.mul(img1Y, img1Y))
        buffer = (img1X == 0)
        buffer = buffer.float()
        buffer = buffer*buf
        img1X = img1X+buffer
        img1A = torch.atan(torch.div(img1Y, img1X))

        img2X = F.conv2d(img2_gray,flt1,padding=1)
        img2Y = F.conv2d(img2_gray,flt2,padding=1)
        img2G = torch.sqrt(torch.mul(img2X, img2X)+torch.mul(img2Y, img2Y))
        buffer = (img2X == 0)
        buffer = buffer.float()
        buffer = buffer*buf
        img2X = img2X+buffer
        img2A = torch.atan(torch.div(img2Y, img2X))
        # 2) edge preservation estimation

        buffer = (img1G == 0)
        buffer = buffer.float()
        buffer = buffer*buf
        img1G = img1G+buffer
        buffer1 = torch.div(fuseG, img1G)

        buffer = (fuseG == 0)
        buffer = buffer.float()
        buffer = buffer*buf
        fuseG = fuseG+buffer
        buffer2 = torch.torch.div(img1G, fuseG)

        bimap = torch.sigmoid(-k*(img1G-fuseG))
        bimap_1 = torch.sigmoid(k*(img1A-fuseA))
        Gaf = torch.mul(bimap, buffer2)+torch.mul((1-bimap), buffer1)
        Aaf = torch.abs(torch.abs(img1A-fuseA)-np.pi/2)*2/np.pi

        #-------------------

        buffer = (img2G == 0)
        buffer = buffer.float()
        buffer = buffer*buf
        img2G = img2G+buffer
        buffer1 = torch.div(fuseG, img2G)

        buffer = (fuseG == 0)
        buffer = buffer.float()
        buffer = buffer*buf
        fuseG = fuseG+buffer
        buffer2 = torch.div(img2G, fuseG)

        #bimap = torch.sigmoid(-k*(img2G-fuseG))
        bimap = torch.sigmoid(-k*(img2G-fuseG))
        bimap_2 = torch.sigmoid(k*(img2A-fuseA))
        Gbf = torch.mul(bimap, buffer2)+torch.mul((1-bimap), buffer1)
        Abf = torch.abs(torch.abs(img2A-fuseA)-np.pi/2)*2/np.pi

        #some parameter
        gama1 = 1
        gama2 = 1
        k1 = -10 
        k2 = -20
        delta1 = 0.5 
        delta2 = 0.75

        Qg_AF = torch.div(gama1, (1+torch.exp(k1*(Gaf-delta1))))
        Qalpha_AF = torch.div(gama2, (1+torch.exp(k2*(Aaf-delta2))))
        Qaf = torch.mul(Qg_AF, Qalpha_AF)

        Qg_BF = torch.div(gama1, (1+torch.exp(k1*(Gbf-delta1))))
        Qalpha_BF = torch.div(gama2, (1+torch.exp(k2*(Abf-delta2))))
        Qbf = torch.mul(Qg_BF, Qalpha_BF)

        # 3) compute the weighting matrix
        L=1
        Wa=torch.pow(img1G, L)
        Wb=torch.pow(img2G, L)
        res=torch.mean(torch.div(torch.mul(Qaf, Wa)+torch.mul(Qbf, Wb), (Wa+Wb)))

        return res
        
    def forward(self, img1, img2, mask, mask_BGF, gt_mask, k = 10e4):
        """
        Compute the GALoss
        :param img1: tensor, input image A
        :param img2: tensor, input image B
        :param mask: tensor, the prediction decision map without bounary guider filter
        :param mask_BGF: tensor, the prediction decision map with bounary guider filter
        :param gt_mask: tensor, the ground-truth decision map 
        :param k: the softening factor of loss_qg
        :return:
        """
        fused = torch.mul(mask_BGF,img1)+torch.mul((1-mask_BGF),img2)
        loss_qg = 1 - self._qg_soft(img1, img2, fused, k)
        loss_dice = self._dice_loss(mask, gt_mask)

        return loss_dice+loss_qg, loss_dice, loss_qg
        