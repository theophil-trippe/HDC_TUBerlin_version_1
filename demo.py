from main import get_deblurrer, get_tensor_from_path
from config import WEIGHTS_PATH
from torchvision.transforms import ToPILImage
from torch import cat
import argparse
from os.path import join
import torch


parser = argparse.ArgumentParser(description='Demo run, deblurring one image test image for each step')
parser.add_argument('save_path', type=str,
                    help='(string) Folder where the output image is located')

args = parser.parse_args()
save_path = args.save_path

def checkpoint_path(step):
    return join(WEIGHTS_PATH, "step_{:02d}".format(step), "UNet_weights_step_{:02d}_v1.pt".format(step))

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
    return inp, pred

rows = []
for step in range(20):
    img_path = join(WEIGHTS_PATH, "step_{:02d}".format(step), "focusStep_{}_verdanaRef_size_30_sample_0100.tif".format(step))
    deblu = get_deblurrer(step, checkpoint_path(step))
    inp, pred = get_pred(img_path, deblu)
    rows.append(cat((inp, pred), dim=2))


example_img = ToPILImage(mode='L')(cat(tuple(rows), dim=1))
example_img.save(join(save_path, 'deblur_demo.png'), format="png")
