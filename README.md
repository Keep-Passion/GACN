# GACN
Gradient Aware Cascade Network for Multi Focus Image Fusion

## Abstract
Muti-focus image fusion is the task of extraction of focused regions from different images to create one all-in-focus fused image. And by virtue of high ability of feature representation, deep learning based methods have become the mainstream of image fusion. However, due to nonlinear mapping in decoder structure, most end-to-end deep learning methods cannot precisely reconstruct fusing result, which leads to poor performance in fusion evaluation. Then, some researchers resort to generate an intermediate result, termed decision map, to decide which pixel should appear in fusion result. Although highly fusing performance of these decision map based methods, they need some post-processing methods with empirical parameters to rectify the decision map, which cannot easily apply to different scenes of image fusion. In this work, by a combination of advantages of the above two classes, we propose a cascade network to simultaneously generate decision map and fusing result with end-to-end training procedure, which avoids many empirical post-processing methods in inference stage. Besides we present a gradient aware loss function to optimize the above network and generate promising fusing results. Also, we present a decision calibration method to decrease time consumption in the application of multi-images fusion. Experiments have shown that our method achieves comparable fusion performance against existing state-of-the-art multi-focus image fusion methods in objective and subjective assessments.

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

# Or run code part by part in notebook
jupyter notebook main.ipynb

# Train GACN with our multi-focus image dataset
python train_net.py

# Or train GACN part by part in notebook
jupyter notebook train_net.ipynb

```

## Citation
Still in submission.