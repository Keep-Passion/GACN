import numpy as np
import cv2
import torch
import torch.nn.functional as F


def evaluate_by_Qg(img1, img2, fuse):
    #flt1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    #flt2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    #1) get the map
    
    #fuseX = filter2(flt1,fuse,'SAME')
    flt1 = torch.from_numpy( np.reshape(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float64),(1,1,3,3)))
    flt2 = torch.from_numpy( np.reshape(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float64),(1,1,3,3)))
    
    h,w = fuse.shape
    fuse = torch.from_numpy(np.reshape(fuse,(1,1,h,w)).astype(np.float64))
    img1 = torch.from_numpy(np.reshape(img1,(1,1,h,w)).astype(np.float64))
    img2 = torch.from_numpy(np.reshape(img2,(1,1,h,w)).astype(np.float64))
    fuseX = F.conv2d(fuse,flt1, padding=1)
    fuseY = F.conv2d(fuse,flt2, padding=1)
    fuseG = torch.sqrt(torch.mul(fuseX, fuseX)+torch.mul(fuseY, fuseY))
    buffer = (fuseX == 0)
    buffer = buffer.double()
    buffer = buffer*0.00001
    fuseX = fuseX+buffer
    fuseA = torch.atan(torch.div(fuseY, fuseX))

    img1X = F.conv2d(img1,flt1,padding=1)
    img1Y = F.conv2d(img1,flt2,padding=1)
    img1G = torch.sqrt(torch.mul(img1X, img1X)+torch.mul(img1Y, img1Y))
    buffer = (img1X == 0)
    buffer = buffer.double()
    buffer = buffer*0.00001
    img1X = img1X+buffer
    img1A = torch.atan(torch.div(img1Y, img1X))

    img2X = F.conv2d(img2,flt1,padding=1)
    img2Y = F.conv2d(img2,flt2,padding=1)
    img2G = torch.sqrt(torch.mul(img2X, img2X)+torch.mul(img2Y, img2Y))
    buffer = (img2X == 0)
    buffer = buffer.double()
    buffer = buffer*0.00001
    img2X = img2X+buffer
    img2A = torch.atan(torch.div(img2Y, img2X))

    # 2) edge preservation estimation

    bimap = img1G>fuseG
    bimap = bimap.double()
    buffer = (img1G == 0)
    buffer = buffer.double()
    buffer = buffer*0.00001
    img1G = img1G+buffer
    buffer1 = torch.div(fuseG, img1G)

    buffer = (fuseG == 0)
    buffer = buffer.double()
    buffer = buffer*0.00001
    fuseG = fuseG+buffer
    buffer2 = torch.div(img1G, fuseG)

    Gaf = torch.mul(bimap, buffer1)+torch.mul((1-bimap), buffer2)
    Aaf = torch.abs(torch.abs(img1A-fuseA)-np.pi/2)*2/np.pi

    #-------------------

    bimap = img2G > fuseG
    bimap = bimap.double()
    buffer = (img2G == 0)
    buffer = buffer.double()
    buffer = buffer*0.00001
    img2G = img2G+buffer
    buffer1 = torch.div(fuseG, img2G)

    buffer = (fuseG == 0)
    buffer = buffer.double()
    buffer = buffer*0.00001
    fuseG = fuseG+buffer
    buffer2 = torch.div(img2G, fuseG)

    Gbf = torch.mul(bimap, buffer1)+torch.mul((1-bimap), buffer2)
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
    L = 1

    Wa = torch.pow(img1G, L)
    Wb = torch.pow(img2G, L)

    # res=np.sum(np.sum(Qaf.*Wa+Qbf.*Wb))/np.sum(np.sum(Wa+Wb))
    res = torch.mean(torch.div(torch.mul(Qaf, Wa)+torch.mul(Qbf, Wb), (Wa+Wb)))
    return res.item()


if __name__ == "__main__":
    a = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]
    b = [[1, 2, 3], [3, 4, 2], [3, 4, 1]]
    a = np.array(a)
    b = np.array(b)
    c = filter2(b, a, 'SAME')
    img1 = cv2.imread('C:/Users/yinxiang/Desktop/picture/data/color_dsift_flower_1.png', 0)
    img2 = cv2.imread('C:/Users/yinxiang/Desktop/picture/data/color_dsift_flower_2.png', 0)
    fused = cv2.imread('C:/Users/yinxiang/Desktop/picture/result/color_dsift_flower.png', 0)
    '''
    cv2.imwrite('C:/Users/yinxiang/Desktop/picture/data/color_dsift_flower_1_gray.png',img1)
    cv2.imwrite('C:/Users/yinxiang/Desktop/picture/data/color_dsift_flower_2_gray.png',img2)
    cv2.imwrite('C:/Users/yinxiang/Desktop/picture/result/color_dsift_flower_gray.png',fused)
    '''
    b = evaluate_by_Qg(img1, img2, fused)

    print(np.shape(b), torch.mul(np.array([3, 3]), [2, 1]))
