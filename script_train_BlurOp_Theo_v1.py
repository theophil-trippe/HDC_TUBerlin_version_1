import torch
from data_management import PSFDataset, Shift, HDC_Dataset_BlurOp2
from networks import BlurOp
import argparse
from train_module import PSF_train
from torchvision.transforms import RandomResizedCrop, ToTensor, Compose, ToPILImage, RandomCrop, CenterCrop

network = BlurOp
dataset = lambda subset, **kwargs: HDC_Dataset_BlurOp2(subset, list(range(0, 180)), **kwargs)
psf_shift = lambda step: ToTensor()(ToPILImage()(torch.cat(PSFDataset(step)[2][0:-1], 0)))


target_folder = "BlurOp/Final"


# ----- network configuration -----
network_params = {
    # "init_kernel": network_init.conv2d.weight,
    "kernel_size": 701,
    "inp_size": 2360,
    "rd_params": torch.tensor([2e-1, 2e-1], requires_grad=True),
    "rd_fac": 1e-3
}

def loss(pred, tar):
    return 1e-4 * (pred - tar).abs().pow(2).sum() / pred.shape[0]

train_phases = 1

train_params = {
    "num_epochs": 100,
    "batch_size": 1,
    "save_epochs": 1,
    "optimizer": torch.optim.Adam,
    "optimizer_params": {"lr": 2e-6, "eps": 1e-5},
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 1, "gamma": 1.0},
    "acc_steps": 1,
    "train_loader_params": {"shuffle": True, "num_workers": 0},
    "val_loader_params": {"shuffle": False, "num_workers": 0},
}


# -----

# steps_to_train = [level, 9, 10, 11, 12]
steps_to_train = list(range(20))
# steps_to_train = [13, 14, 15, 16, 17, 18, 19]
# steps_to_train = [0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Network with for the given step')
    parser.add_argument('TASK_ID', type=int,
                        help='an integer to set the step for the network to be trained for.')
    args = parser.parse_args()
    task_id = args.TASK_ID
    # task_id = 1
    step = steps_to_train[task_id - 1]

    # ----- data configuration -----

    # train_transforms = Compose([ToPILImage(), ToTensor(), Shift(psf_shift(step))])
    # val_transforms = Compose([ToPILImage(), ToTensor(), Shift(psf_shift(step))])

    train_data_params = {
        # 'transform': train_transforms
    }
    val_data_params = {
        # 'transform': val_transforms
    }

    # network_params["bg_gt"] = psf_shift(step)[1, ...].unsqueeze(0).unsqueeze(0)
    # network_params["bg_blurred"] = psf_shift(step)[0, ...].unsqueeze(0).unsqueeze(0)

    PSF_trainer = PSF_train(network,
                            dataset,
                            target_folder,
                            network_params,
                            train_params,
                            train_data_params,
                            val_data_params,
                            loss_func=loss,
                            train_phases=train_phases,
                            steps_to_train=steps_to_train)
    PSF_trainer(task_id)
