from config import RESULTS_PATH, DATA_PATH
import matplotlib.pyplot as plt
from PIL import Image
from data_management import change_im_range
import os.path
from os.path import join
import random
import argparse
from networks import DeepDeepUNet_v2_big
import torch
from config import RESULTS_PATH, TMP_REP, WEIGHTS_PATH
from forward_radial_dist import get_fwd_op
from torchvision.transforms import Compose, CenterCrop, ToTensor, ToPILImage, Resize
from data_transforms import Apply_Fct_To_Input
from OCR_evaluation import evaluateImage

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    str_dev = "GPU"
else:
    device = torch.device("cpu")
    str_dev = "CPU"

# ----- network specific configurations -----
def get_deblurrer(step, checkpoint_path, device=device):
    """
    :param step: blurryness step
    :param checkpoint_path: (to be formated by step), eg: "checkpoints/version_2/checkpoint_v2_step{:02d}"
    :param device: device to perform deblurring on
    :return: deblurrer
    """
    Blur_OP = get_fwd_op(step).to(device)
    # todo: change network archtiecture
    network = DeepDeepUNet_v2_big
    network_params = {
        "in_channels": 1,
        "drop_factor": 0.0,
        "base_features": 32,
        "out_channels": 1,
        "num_groups": 32,
        "do_center_crop": True
    }

    # if torch.cuda.is_available():
    #     checkpoint = torch.load(checkpoint_path.format(step))
    # else:
    #
    checkpoint = torch.load(checkpoint_path.format(step), map_location=device)

    deblur_net = network(**network_params).to(device)
    deblur_net.load_state_dict(checkpoint)

    pretransform = Apply_Fct_To_Input(Blur_OP.undistort)
    def _deblurrer(input_tensor):
        img_tensor = input_tensor.to(device)
        img_tensor = pretransform(img_tensor)
        return deblur_net.forward(torch.unsqueeze(img_tensor, dim=0))
    return _deblurrer


def get_tensor_from_path(image_path):
    _pre_trans = Compose([ToTensor()])
    with Image.open(image_path, 'r') as file:
        img = _pre_trans(file.point(lambda i: i * (1. / 256)).convert('L'))
    return img


def get_pred(image_path, *deblurrer):
    inp = get_tensor_from_path(image_path)
    pred = torch.zeros_like(inp)
    for debl in deblurrer:
        if isinstance(debl, tuple):
            pred += debl[0](inp).detach().squeeze().cpu()
        else:
            pred += debl(inp).detach().squeeze().cpu()
    num_deblr = len(deblurrer)
    pred = (1./num_deblr) * pred
    return pred



def get_OCR_score(image_path, text_targ_path, *deblurrer):
    deblurred_img_path = save_png_result(image_path, TMP_REP, *deblurrer)
    return evaluateImage(deblurred_img_path, text_targ_path)


def create_fig(image_path, text=None, *deblurrer):
    def _implot(sub, im, vmin=None, vmax=None):
        p = sub.imshow(im.squeeze().detach().cpu(), vmin=vmin, vmax=vmax)
        return p

    inp = get_tensor_from_path(image_path)
    # pred = deblurrer(inp)

    pred = get_pred(image_path, *deblurrer)

    fig, subs = plt.subplots(2, clear=True, num=1, figsize=(20, 20))

    p0 = _implot(subs[0], inp)
    subs[0].set_title("input")
    plt.colorbar(p0, ax=subs[0])

    p1 = _implot(subs[1], pred)
    subs[1].set_title("prediction" if text is None else text)
    plt.colorbar(p1, ax=subs[1])
    return fig


def save_png_result(image_path, target_folder, *deblurrer):
    # inp = get_tensor_from_path(image_path)
    # pred = deblurrer(inp)
    # pred_img = pred.detach().squeeze().cpu()
    #
    pred_img = get_pred(image_path, *deblurrer)
    pred_img = ToPILImage(mode='L')(pred_img)
    filename, extension = os.path.splitext(os.path.basename(image_path))
    save_path = os.path.join(target_folder, filename + ".png")
    pred_img.save(save_path, format="png")
    return save_path


def show_plot_with_OCR(image_path, text_targ_path, *deblurrer):
    OCR_score = get_OCR_score(image_path, text_targ_path, *deblurrer)
    fig = create_fig(image_path, OCR_score, *deblurrer)
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deblur all images in the given directory')
    parser.add_argument('input_folder', type=str,
                        help='(string) Folder where the input image files are located')
    parser.add_argument('output_folder', type=str,
                        help='(string) Folder where the input image files are located')
    parser.add_argument('step', type=int,
                        help='(int) Blur category number. Values between 0 and 19')
    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    step = args.step
    files = os.listdir(input_folder)

    # todo: specify checkpoint location, depending on config import

    checkpoint_path_1 = os.path.join(WEIGHTS_PATH, "step_{:02d}".format(step), "UNet_weights_step_{:02d}_v1.pt")

    deb_1 = get_deblurrer(step, checkpoint_path_1, device=device)
    for i, file in enumerate(files):
        save_png_result(os.path.join(input_folder, file), output_folder, deb_1)
        print("file " + str(i + 1) + "/" + str(len(files)) + " deblurred on " + str_dev)
