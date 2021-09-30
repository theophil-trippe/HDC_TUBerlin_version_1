from os.path import join
import random
from torchvision.transforms import ToTensor, CenterCrop, Resize, ToPILImage, Compose, RandomResizedCrop, RandomAffine, RandomCrop
from PIL import Image, ImageOps
from config import DATA_PATH, VERDANA_PATH, TIMES_PATH, FONT_PATHS
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from data_generator import generate_shifted_sample, change_im_range, add_relative_noise
import matplotlib.pyplot as plt
from data_transforms import *
# for testing purpose only
import operators
import os


# ---- default background: ----
def _psf_shift(step):
    return (torch.cat(PSFDataset(step)[2][0:-1], 0))


# ----- base datasets -----

# creates a torch.dataset for the font data
class HDCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        step,
        subset,
        val_split=10,
        font=None,
        num_fold=0,
        font_id=None,
        transform=None,
        device=None,
        input_transform=None,
        shift_bg=None,
        add_bg=None  # dummy parameter doesnt do anything
    ):
        self.type = "HDC"
        self.transform = transform
        self.input_transform = input_transform
        self.device = device
        self.subset = subset
        if shift_bg is not None:
            self.bg = shift_bg(step)
        else:
            self.bg = None
        # create validation split, s.t. 2*val_split will be in the validation set
        sample_ids = list(range(98))
        random.seed(val_split + num_fold)  # that way, the randomization will always be identical for a fixed val_split
        random.shuffle(sample_ids)
        if subset == "val":
            self.sample_ids = [str(i+1) for i in sample_ids[0:val_split - 1]]
        elif subset == "train":
            self.sample_ids = [str(i+1) for i in sample_ids[val_split - 1:]]
        else:
            print("ERROR: subset need to be one of 'val' or 'train' ")

        # set the font_id in order to set the right filenames later on
        font_names = {'Verdana': 'verdanaRef',
                      'Times': 'timesR'}
        if font == 'Verdana':
            del font_names['Times']
        elif font == 'Times':
            del font_names['Verdana']
        if font_id is not None and font is not None:
            font_names[font] = font_id

        self.blury_img_paths = []
        self.sharp_img_paths = []
        self.text_target_paths = []
        # choose directory according to step and font
        for font in font_names:
            path = join(DATA_PATH, 'step' + str(step), font.capitalize())
            font_id = font_names[font]
            # load data files
            def sample_name(sample_id):
                return 'focusStep_' + str(step) + '_' + font_id + '_size_30_sample_' + str(sample_id).zfill(4)
            blury_img_paths = [join(path, 'CAM02', sample_name(sample_id) + '.tif') for sample_id in self.sample_ids]
            sharp_img_paths = [join(path, 'CAM01', sample_name(sample_id) + '.tif') for sample_id in self.sample_ids]
            text_target_paths = [join(path, 'CAM01', sample_name(sample_id) + '.txt') for sample_id in self.sample_ids]

            self.blury_img_paths.extend(blury_img_paths)
            self.sharp_img_paths.extend(sharp_img_paths)
            self.text_target_paths.extend(text_target_paths)

        assert len(self.blury_img_paths) == len(self.sharp_img_paths)
        assert len(self.blury_img_paths) == len(self.text_target_paths)

    def __len__(self):
        return len(self.blury_img_paths)

    def __getitem__(self, idx):
        # create the text target.
        with open(self.text_target_paths[idx], 'r') as f:
            text_target = f.readlines()
        # if self.subset == 'val':  # skip this step if not for validation set, to speed up DataLoading
            text_target = [text.rstrip() for text in text_target]

        # load sample from storage
        def load(paths, IDX):
            # pre_trans = Compose([CenterCrop((1456, 2352)), ToTensor()])
            pre_trans = Compose([ToTensor()])

            with Image.open(paths[IDX], 'r') as file:
                img = pre_trans(file.point(lambda i: i * (1. / 256)).convert('L'))
            return img

        out = [load(self.blury_img_paths, idx), # loads feature
               load(self.sharp_img_paths, idx)] # loads target

        # move to device and apply transformations
        if self.device is not None:
            out = [x.to(self.device) for x in out]

        if self.bg is not None:
            out = list(torch.chunk(Shift(self.bg)(torch.cat((out[0], out[1]), 0)),
                                   2))  # torch.cat -> transform -> torch.chunk necessary for random transforms

        blur_min = torch.min(out[0]).item()
        blur_max = torch.max(out[0]).item()
        shar_min = torch.min(out[1]).item()
        shar_max = torch.max(out[1]).item()

        out[0] = change_im_range(out[0], new_min=0.0, new_max=1.0)
        out[1] = change_im_range(out[1], new_min=0.0, new_max=1.0)

        if self.input_transform is not None:
            out[0] = self.input_transform(out[0])

        if self.transform is not None:
            out = [self.transform(x) for x in out]

        out[0] = change_im_range(out[0], new_min=blur_min, new_max=blur_max)
        out[1] = change_im_range(out[1], new_min=shar_min, new_max=shar_max)

        return tuple(out) + (text_target,)

# creates a torch.dataset for the PSF and LSF data (3 samples per step)
class PSFDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        step,
        transform=None,
        input_transform=None,
        device=None,
        shift_bg=None,
        add_bg=None  # dummy parameter doesnt do anything
    ):
        self.type = "PSF"
        self.transform = transform
        self.input_transform = input_transform
        self.device = device
        self.step = step
        self.sample_ids = ['LSF_X', 'LSF_Y', 'PSF']
        if shift_bg is not None:
            self.bg = shift_bg(step)
        else:
            self.bg = None
        font = 'Times'
        # choose directory according to step and (either) font
        path = join(DATA_PATH, 'step' + str(step), font)

        # load file names with location
        self.feature_paths = [join(path, 'CAM02', 'focusStep_' + str(step) + '_' + idx + '.tif') for idx in self.sample_ids]
        self.target_paths = [join(path, 'CAM01', 'focusStep_' + str(step) + '_' + idx + '.tif') for idx in self.sample_ids]

        assert len(self.feature_paths) == len(self.target_paths)

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):

        # load sample from storage
        def load(paths, IDX):
            # pre_trans = Compose([CenterCrop((1456, 2352)), ToTensor()])
            pre_trans= Compose([ToTensor()])
            with Image.open(paths[IDX], 'r') as file:
                img = pre_trans(file.point(lambda i: i * (1. / 256)).convert('L'))
            return img

        out = [load(self.feature_paths, idx),  # loads feature
               load(self.target_paths, idx)]  # loads target
        # print(img.mode, img.getextrema())

        # move to device and apply transformations
        if self.device is not None:
            out = [x.to(self.device) for x in out]

        if self.bg is not None:
            out = list(torch.chunk(Shift(self.bg)(torch.cat((out[0], out[1]), 0)),
                                   2))  # torch.cat -> transform -> torch.chunk necessary for random transforms

        blur_min = torch.min(out[0]).item()
        blur_max = torch.max(out[0]).item()
        shar_min = torch.min(out[1]).item()
        shar_max = torch.max(out[1]).item()

        out[0] = change_im_range(out[0], new_min=0.0, new_max=1.0)
        out[1] = change_im_range(out[1], new_min=0.0, new_max=1.0)

        if self.input_transform is not None:
            out[0] = self.input_transform(out[0])

        if self.transform is not None:
            out = [self.transform(x) for x in out]

        out[0] = change_im_range(out[0], new_min=blur_min, new_max=blur_max)
        out[1] = change_im_range(out[1], new_min=shar_min, new_max=shar_max)

        return tuple(out) + (self.sample_ids[idx],)

class Synthesized_Dataset(torch.utils.data.Dataset):
    """
    Generates a dataset containing synthesized data.
    WARNING: two calls of getitem produce seperate samples
    """
    def __init__(
            self,
            step,
            subset,
            forward_op,
            length=15,
            font_paths=None,
            transform=None,
            input_transform=None,
            device=None,
            sample_generator=generate_shifted_sample,
            timing_mode=False,
            add_bg=None,
            shift_bg=None,  # dummy parameter doesnt do anything
            rel_noise_var=0
    ):
        """
        :param step: blurriness
        :param forward: function taking three parameter: input, step, device
                        and returning a simulation of the forward operator
        :param font_paths: list containing the paths to all fonts to be used
        :param length: legth of dataset (needed to define __len__)
        :param transform: transforms to be applied on the samples
        :param device: device to locate the samples on
        """

        self.transform = transform
        self.input_transform = input_transform
        self.device = device
        self.step = step
        self.length = length
        self.sample_generator = sample_generator
        self.timing_mode = timing_mode
        self.subset = subset
        if font_paths:
            self.font_paths = font_paths
        else:
            self.font_paths = FONT_PATHS
        self.type = "Sim"
        self.forward_op = forward_op
        if add_bg is not None:
            self.bg = - add_bg(step)
        else:
            self.bg = None
        self.rel_noise_var = rel_noise_var

    # def _fwd_op(self, inp):
    #     return self.forward_op(inp, self.step, self.device)

    def __len__(self):
        if self.subset == "val":
            return 10
        else:
            return self.length

    def __getitem__(self, IDX):
        with torch.no_grad():
            # blurry, sharp, txt = self.sample_generator(forward_op=self.forward_op, step=self.step, device=self.device, font_paths=self.font_paths)
            blurry, sharp, txt = self.sample_generator(forward_op=self.forward_op, font_paths=self.font_paths)
            blurry.detach()
            out = [blurry, sharp]

            if self.device is not None:
                out = [x.to(self.device) for x in out]

            if self.bg is not None:
                tmp = torch.cat((out[0], out[1]), 0)
                ret = Shift(self.bg)(tmp)
                out = list(torch.chunk(ret, 2))

            blur_min = torch.min(out[0]).item()
            blur_max = torch.max(out[0]).item()
            shar_min = torch.min(out[1]).item()
            shar_max = torch.max(out[1]).item()

            out[0] = change_im_range(out[0], new_min=0.0, new_max=1.0)
            out[1] = change_im_range(out[1], new_min=0.0, new_max=1.0)

            if self.input_transform is not None:
                out[0] = self.input_transform(out[0])

            if self.transform is not None:
                out = [self.transform(x) for x in out]

            out[0] = change_im_range(out[0], new_min=blur_min, new_max=blur_max)
            out[1] = change_im_range(out[1], new_min=shar_min, new_max=shar_max)

            if self.rel_noise_var != 0:
                out[0] = add_relative_noise(out[0], variance=self.rel_noise_var).float()

            if not self.timing_mode:
                return tuple(out) + (txt,)
            if self.timing_mode:
                return IDX


# Todo: Change version attribute to subset attribute
class Sanity_Dataset(Dataset):
    def __init__(self,
                 step,
                 forward_op,
                 version='stock',
                 transform=None,
                 input_transform=None,
                 device=None,
                 add_bg=None,
                 shift_bg=None,  # dummy parameter doesnt do anything
                 randomize=False
                 ):
        self.version = version
        self.forward_op = forward_op
        self.step = step
        self.type = "San"
        self.transform = transform
        self.input_transform = input_transform
        self.device = device
        if add_bg is not None:
            self.bg = - add_bg(step)
        else:
            self.bg = None
        self.randomize = randomize

    def __len__(self):
        if self.version == 'stock':
            return 1
        elif self.version == 'train':
            return 1489
        elif self.version == 'new_train':
            return 3550
        else:
            return 0


    def __getitem__(self, item):
        if self.version == 'stock':
            path = join(DATA_PATH, "sanity_check", "sanity.jpg")
            with Image.open(path, 'r') as file:
                img = file.convert('L')
                img = ImageOps.invert(img)
                img = img.resize((2360, 1460))
                sharp = ToTensor()(img)
                sharp = change_im_range(sharp, new_min=-0.79, new_max=0.0)

        elif self.version == "train":
            dir_path = join(DATA_PATH, "sanity_check", "RGB")
            files = os.listdir(dir_path)

            if self.randomize:
                img_ids = random.sample(range(0, len(self)), 9)
            else:
                img_ids = range(9 * item, 9 * (item+1))

            sharp = []
            for i in img_ids:
                with Image.open(join(dir_path, files[i]), 'r') as file:
                    img = file.convert('L')
                    img_tens = ToTensor()(img)
                    sharp.append(img_tens)
            sharp = [torch.cat(tuple(tuple(sharp[i::3])), dim=1) for i in range(3)]
            sharp = torch.cat(tuple(sharp), dim=2)
            sharp = RandomCrop((1460, 2360))(ToPILImage(mode="L")(sharp))
            sharp = ToTensor()(ImageOps.invert(sharp))
            sharp = change_im_range(sharp, new_min=-0.79, new_max=0.0)

        elif self.version == "new_train":
            idx = item
            path = join(DATA_PATH, "sanity_check_2")
            if item < 800:
                path = join(path, "DIV2K_train_HR", "{:04d}.png").format(item + 1)
            elif item < 900:
                path = join(path, "DIV2K_val_HR", "{:04d}.png").format(item - 800 + 1)
            else:
                path = join(path, "Flickr2K", "*{:06d}.png").format(item + 1)
            with Image.open(path, 'r') as file:
                sharp = ToTensor()(file).convert('L')

        else:
            sharp = None

        blurry = self.forward_op(sharp)

        if self.device is not None:
            blurry = blurry.to(self.device)
            sharp = sharp.to(self.device)

        out = [blurry, sharp]

        if self.bg is not None:
            out = list(torch.chunk(Shift(self.bg)(torch.cat((out[0], out[1]), 0)),
                                   2))  # torch.cat -> transform -> torch.chunk necessary for random transforms

        blur_min = torch.min(out[0]).item()
        blur_max = torch.max(out[0]).item()
        shar_min = torch.min(out[1]).item()
        shar_max = torch.max(out[1]).item()

        out[0] = change_im_range(out[0], new_min=0.0, new_max=1.0)
        out[1] = change_im_range(out[1], new_min=0.0, new_max=1.0)

        if self.input_transform is not None:
            out[0] = self.input_transform(out[0])

        if self.transform is not None:
            out = [self.transform(x) for x in out]
            # out = list(torch.chunk(self.transform(torch.cat((out[0], out[1]), 0)),
            #                        2))  # torch.cat -> transform -> torch.chunk necessary for random transforms

        out[0] = change_im_range(out[0], new_min=blur_min, new_max=blur_max)
        out[1] = change_im_range(out[1], new_min=shar_min, new_max=shar_max)

        return tuple(out) + (3*[self.version],)


# ----- combined datasets -----

# combines HDC and PSF Datasets
def HDC_Dataset_BlurOp2(subset, indices, shifted=True, background=_psf_shift, **kwargs):
    if shifted:
        kwargs["add_bg"] = None
        kwargs["shift_bg"] = background
    else:
        kwargs["shift_bg"] = None
        kwargs["add_bg"] = background
    if subset == 'train':
        return torch.utils.data.Subset(HDCDataset(val_split=10, subset='train', **kwargs), indices)
    elif subset == 'val':
        return HDCDataset(val_split=10, subset='val', **kwargs)
    else:
        print("ERROR: Subset needs to be either 'train' or 'val'")
        return None


def Validation_Set_all(forward_op, shifted, background=_psf_shift, data_generator=generate_shifted_sample, sim_val_len=10, rel_noise_var=0, **kwargs):
    if shifted:
        kwargs["add_bg"] = None
        kwargs["shift_bg"] = background
    else:
        kwargs["shift_bg"] = None
        kwargs["add_bg"] = background
    HDC = HDCDataset(subset='val', val_split=10, **kwargs)
    Sim = Synthesized_Dataset(subset='train', length=sim_val_len, forward_op=forward_op, sample_generator=data_generator, rel_noise_var=rel_noise_var, **kwargs)
    San = Sanity_Dataset(forward_op=forward_op, **kwargs)
    PSF = PSFDataset(**kwargs)
    return ConcatDataset([HDC, Sim, San, PSF])


def Mixed_train_set(forward_op, shifted, hdc_sim_san, background=_psf_shift, data_generator=generate_shifted_sample, rel_noise_var=0, randomize_san=True, sim_len=100000, **kwargs):
    assert isinstance(hdc_sim_san, (list, tuple))  # please provide a tuple of length 3
    assert len(hdc_sim_san) == 3  # please provide a tuple of length 3
    if not hdc_sim_san[0] <= 178:  # at most 180 samples are available from the HDC dataset, 4 are reserved for testing
        hdc_sim_san[0] = 178
    assert hdc_sim_san[2] <= 1489  # at most 1489 samples are available from the sanity dataset
    if shifted:
        kwargs["add_bg"] = None
        kwargs["shift_bg"] = background
    else:
        kwargs["shift_bg"] = None
        kwargs["add_bg"] = background
    HDC = torch.utils.data.Subset(HDCDataset(subset='train', val_split=10, **kwargs), list(range(hdc_sim_san[0])))
    Sim = torch.utils.data.Subset(Synthesized_Dataset(subset='train', length=sim_len, forward_op=forward_op, sample_generator=data_generator, rel_noise_var=rel_noise_var, **kwargs), list(range(hdc_sim_san[1])))
    San = torch.utils.data.Subset(Sanity_Dataset(forward_op=forward_op, version="train", randomize=randomize_san, **kwargs), list(range(hdc_sim_san[2])))
    return ConcatDataset([HDC, Sim, San])
