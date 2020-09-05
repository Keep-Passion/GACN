import os
from skimage import io
from nets.gacn_net import GACN_Fuse


def image_fusion(input_dir:  str, output_dir: str):
    """
    Double images fusion
    :param input_dir:  str, input dir with all images stores in one folder
    :param output_dir: str, output dir with all fused images
    :return:
    """
    gacn = GACN_Fuse()
    images_name = sorted(list({item[:-6] for item in os.listdir(input_dir)}))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for image_name in images_name:
        if not (image_name.startswith('.') or image_name == ''):
            print("Fusing {}".format(image_name))
            img1 = io.imread(os.path.join(input_dir, image_name + "_1.png"))
            img2 = io.imread(os.path.join(input_dir, image_name + "_2.png"))
            fused = gacn.fuse(img1, img2)
            io.imsave(os.path.join(output_dir, image_name + ".png"), fused)


def multi_images_fusion(input_dir: str, output_dir: str, strategy: str="calibration"):
    """
    Multiple images fusion
    :param input_dir: str, input dir with all images stores in one folder
    :param output_dir: str, output dir with all fused images
    :param strategy: str, fusion strategy:
                    "calibration" means using decision calibration fusion strategy
                    "origin" means using one by one serial fusion strategy
    :return:
    """  
    gacn = GACN_Fuse()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    images_path = os.listdir(input_dir)
    if ".ipynb_checkpoints" in images_path:
        images_path.remove(".ipynb_checkpoints")
    for images in images_path:
        final_path = os.path.join(input_dir, images)
        if strategy == "calibration":
            fused = gacn.multi_fuse_calibration(final_path)
        elif strategy == "origin":
            fused = gacn.multi_fuse_origin(final_path)
        else:
            raise NameError("illegal fusion strategy")
        io.imsave(os.path.join(output_dir, images+".png"), fused)


if __name__ == "__main__":
    
    # Double images fusion
    input_dir = os.path.join(os.getcwd(), "data", "multi_focus")
    output_dir = os.path.join(os.getcwd(), "data", "result")
    image_fusion(input_dir, output_dir)
    
    # Multi image fusion using Decision calibration fusion strategy
    input_dir_multi = os.path.join(os.getcwd(), "data", "material")
    output_dir_multi = os.path.join(os.getcwd(), "data", "result")
    multi_images_fusion(input_dir_multi, output_dir_multi, "calibration")