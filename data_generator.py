import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
# import cv2
import string
import random
from config import VERDANA_PATH, TIMES_PATH, FONT_PATHS
from torchvision.transforms import ToTensor, Compose, CenterCrop, ToPILImage
import torch


def generate_text(num_lines=3, line_len_min=10, line_len_max=10):
    """
    :param num_lines: number of lines to be returned
    :param line_len_min: minimal length each line should have
    :param line_len_max: maximal lentgh each line should have
    :return: list of length 'num_lines' containing random strings
    """
    alphabet = [string.ascii_lowercase + ' ', string.ascii_uppercase + ' ']
    alphabet_len = len(alphabet[0])

    characters = 3 * string.ascii_letters + \
                 2 * string.digits + \
                 ' +!#&()*+-/:;=>?\^'
    num_characters = len(characters)

    out = []
    for l in range(num_lines):
        length = random.randint(line_len_min, line_len_max)
        str_l = ''
        for c in range(length):
            char_id = random.randint(0, num_characters-1)  # determines which character to choose
            str_l = str_l + characters[char_id]
        out.append(str_l)

    return out


def generate_background(size=(300, 200), intensity=220):
    """
    :param size: width x height of image to be generated
    :param intensity: background intensity
    :return: returns monochromatic grayscale PIL.Image
    """
    return Image.new('L', size, color=intensity)

# def translate_cv(cv_img, x_shift, y_shift):
#     rows, cols = cv_img.shape
#     M = np.float32([[1, 0, x_shift],
#                     [0, 1, y_shift]])
#     out = cv2.warpAffine(cv_img, M, (cols,rows))
#     return out
#
# def rotate_image(image, angle):
#   image_center = tuple(np.array(image.shape[1::-1]) / 2)
#   rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#   result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
#   return result

def add_text(img, text=['Example'], positions=[(0, 0)], font_path=VERDANA_PATH, font_size=35, font_colour=0):
    """
    :param img: input image (PIL.Image)
    :param text: list containing the strings to be added
    :param positions: list containing the positions where the text should be placed on the input image
    :param font_path: path pointing to the location of the font to be used
    :param font_size:
    :param font_colour:
    :return: input image with text added onto it
    """
    font = ImageFont.truetype(font_path, font_size, layout_engine=ImageFont.LAYOUT_RAQM)

    assert len(text) == len(positions)
    d = ImageDraw.Draw(img)
    for l in range(len(text)):
        d.text(positions[l], text[l], font=font, fill=font_colour)
    return img, text

def add_gaussian_noise(img, mean=0, var=10):
    """
    :param img: input image (PIL.Image)
    :param mean: should stay 0 unless image should be brightened up or darkened
    :param var: determines amount of noise to be added by this funciton
    :return: input image with random gaussian noise added
    """
    gaussian = np.random.normal(mean, var**0.5, (img.height, img.width))
    return Image.fromarray(np.clip(img + gaussian, 0, 255).astype(np.uint8))

def gen_and_add_text_HDC_style(img, font_size=35, font_path=VERDANA_PATH, line_spacing=5, font_colour=20):
    """
    :param img: input image (PIL.Image)
    :param font_size: integer
    :param font_path: pick one of the font paths specified in config
    :param line_spacing: space between lines
    :param font_colour: uint8 integer determining the font colour
    :return: randomly generates three string-lines and adds them to the input image in HDC-style
    """
    cwd = os.getcwd()
    font = ImageFont.truetype(font_path, font_size)
    text = generate_text(3, 10, 10)
    d = ImageDraw.Draw(img)
    positions = []
    text_sizes = []
    for i in range(3):
        w, h = d.textsize(text[i], font=font)
        text_sizes.append((w, h))
        W, H = (img.width - w) / 2, (img.height - h) / 2
        if len(positions) == 1:
            H = H + line_spacing + font_size
        if len(positions) == 2:
            H = H - (line_spacing + font_size)
        positions.append((W, H))
    text = [text[2], text[0], text[1]]
    positions = [positions[2], positions[0], positions[1]]
    return add_text(img, text, positions, font_path, font_size, font_colour)




# ---- public -----
def change_im_range(img_tens,
                    new_min,
                    new_max):
    old_min = torch.min(img_tens).item()
    old_max = torch.max(img_tens).item()
    ret = ((img_tens - old_min) * ((new_max - new_min) / (old_max - old_min))) + new_min
    return ret

def generate_shifted_sample_old(forward_op,
                            step,
                    device=None,
                    random_position=False,
                    font_paths=FONT_PATHS,
                    save_dataset=False,):
    font_index = random.randint(0, len(font_paths) - 1)

    img = generate_background(size=(600, 400), intensity=220)

    if random_position:
        img, text = gen_and_add_text_HDC_style(img, font_size=70, font_path=font_paths[font_index])
    else:
        img, text = gen_and_add_text_HDC_style(img, font_size=70, font_path=font_paths[font_index])

    img = img.resize((2360, 1460), resample=Image.NEAREST)
    sharp = ToTensor()(img)
    sharp = change_im_range(sharp, new_min=-0.75, new_max=0.0)
    if device is not None:
        sharp.to(device)
    return forward_op(sharp, step, device), sharp, text


def generate_shifted_sample(forward_op,
                    random_position=False,
                    font_paths=FONT_PATHS):
    font_index = random.randint(0, len(font_paths) - 1)

    img = generate_background(size=(600, 400), intensity=220)

    if random_position:
        img, text = gen_and_add_text_HDC_style(img, font_size=70, font_path=font_paths[font_index])
    else:
        img, text = gen_and_add_text_HDC_style(img, font_size=70, font_path=font_paths[font_index])

    img = img.resize((2360, 1460), resample=Image.NEAREST)
    sharp = ToTensor()(img)
    sharp = change_im_range(sharp, new_min=-0.75, new_max=0.0)
    return forward_op(sharp), sharp, text

def add_relative_noise(img_tens, mean=0, variance=0, ):
    dims = tuple(img_tens.shape)
    if img_tens.is_cuda:
        dev =  img_tens.get_device()
        abs_noise = torch.tensor(np.random.normal(mean + 1, variance ** 0.5, dims), device=torch.device('cuda:' + str(dev)))
    else:
        abs_noise = torch.tensor(np.random.normal(mean + 1, variance ** 0.5, dims))
    return img_tens * abs_noise
