from main import get_deblurrer, get_inp_pred
from config import WEIGHTS_PATH
from torchvision.transforms import ToPILImage
from torch import cat
import argparse
from os.path import join


parser = argparse.ArgumentParser(description='Demo run, deblurring one image test image for each step')
parser.add_argument('save_path', type=str,
                    help='(string) Folder where the output image is located')

args = parser.parse_args()
save_path = args.save_path

def checkpoint_path(step):
    return join(WEIGHTS_PATH, "step_{:02d}".format(step), "UNet_weights_step_{:02d}_v1.pt".format(step))

rows = []
for step in range(20):
    img_path = join(WEIGHTS_PATH, "focusStep_0_verdanaRef_size_30_sample_0100.tif")
    deblu = get_deblurrer(step, checkpoint_path(step))
    inp, pred = get_inp_pred(img_path, deblu)
    rows.append(cat((inp, pred), dim=2))


example_img = ToPILImage(mode='L')(cat(tuple(rows), dim=1))
example_img.save(join(save_path, 'deblur_demo.png'), format="png")
