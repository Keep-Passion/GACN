# GACN
Gradient Aware Cascade Network for Multi-Focus Image Fusion

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
Still in submission.