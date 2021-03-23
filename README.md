# GACN
End-to-End Learning for Simultaneously GeneratingDecision Map and Multi-Focus Image Fusion Result

## Abstract
The general aim of multi-focus image fusion is to gather focused regions of different images to generate a unique all-in-focus fused image. Deep learning based methods become the mainstream of image fusion by virtue of its powerful feature representation ability. However, most of the existing deep learning structures failed to balance fusing quality and end-to-end implementation convenience. End-to-end decoder design often leads to poor performance because of non-linear mapping. On the other hands, generating an intermediate decision map can achieve better quality for fused image, but relies on the rectification with empirical post-processing parameter choices. In this work, to handle the requirements of both output image quality and comprehensive simplicity of structure implementation, we propose a cascade network to simultaneously generate decision map and fusing result with an end-to-end training procedure. It avoids utilizing empirical post-processing methods in the inference stage. To improve output fused image quality, we introduce a gradient aware loss function to preserve gradient information in output fused image. In addition, we design a decision calibration method to decrease the time consumption in the application of multiple images fusion. Extensive experiments are conducted to compare with 16 different state-of-the-art multi-focus image fusion structures with 6 assessment metrics. We implement overall ablation studies additionally to test the impact of different modules in our network structure. The results prove that our designed structure can generally ameliorate the output fused image quality for multi-focus images, while implementation efficiency increases for over 25\%.

![avatar](/paper/network.png)
## Branches Introduction
We provide the training and testing method of GACN in this branch.  

## Requirements
- Pytorch = 1.2.0
- Python = 3.6
- torchvision = 0.5.0
- numpy = 1.17.0
- opencv-python
- scikit-image
- pillow
- matplotlib
- jupyter notebook

## Usage
```bash
# Clone our code
git clone https://github.com/Keep-Passion/GACN.git
cd GACN

# Replicate our image method on fusing multi-focus images
python main.py

# Train GACN 
python train_net.py

```
## Visualization
![avatar](/paper/visualization.png)

## Citation
Still in submission. And the pre-print version can be found at [the paper](https://arxiv.org/abs/2010.08751)

## Acknowledgement
The authors acknowledge the financial support from the National Key Research and Development 
Program of China (No. 2016YFB0700500). Besides, the dataset of multiple image fusion is driven from the 
[website](https://mp.weixin.qq.com/s?__biz=MzU1ODE1NTQ0Mg==&mid=2247488091&idx=1&sn=648aa20f0d2dd599f194392aaba37dcc&chksm=fc2b8186cb5c08905480d20bfc2cbcf1946b6bdb746d7e38dfe68061a10b6b545eb50e2c3cf7&mpshare=1&scene=1&srcid=072957pnJNaD0fSfk5xWCVAt&sharer_sharetime=1596024724190&sharer_shareid=5ebb4ebec0efa12c73f7f111bfa30973&key=590f90317dcde6d793130d527e1a34d7d813595dc6ec22d294606b7bc26e32c68321a95f0b68fcbbf46ddbdba56f614487090a062e0a1c02e25268307f0bea1825346b7863d7ffffdd3e1877ad5be11817d7562c23737d8af7f9919cab373f3e7ba0091cf7b765a48ba86c2442915db53b38bd27c1cda54db54a5932369d01e7&ascene=1&uin=MTcxMjE3OTc0MA%3D%3D&devicetype=Windows+10+x64&version=62090529&lang=zh_CN&exportkey=AQnWDf40U6oLM5pjz69RWcI%3D&pass_ticket=XdlQqtchEHnePeyXepJjmagwSDi3IJPG9i1G1%2FiJSXWjsCgRP8rk56O3qRxINEIZ&wx_header=0) provided by [Zhuhai Boming Vision Technology Co., Ltd](http://bomming.com/index.html).
