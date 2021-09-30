import torch.cuda
from networks import BlurOp
from data_transforms import *
from os.path import join
from config import RESULTS_PATH, WEIGHTS_PATH


def get_fwd_op(step, kernel_size=701):
    network = BlurOp
    # model_weights_part = join("results/BlurOp/v1/step_{:02d}".format(step), "kernel_size_{:03d}".format(kernel_size),
    #                           "model_weights_epoch{:03d}.pt".format(epoch))
    model_weights_part = join(WEIGHTS_PATH, "step_{:02d}".format(step),
                              "forward_weights_step_{:02d}.pt".format(step))
    if torch.cuda.is_available():
        weights = torch.load(model_weights_part)
    else:
        weights = torch.load(model_weights_part, map_location=torch.device("cpu"))
    network_params = {
        "kernel_size": kernel_size,
        "inp_size": 2360,
        "rd_fac": 1e-3
    }

    fwd_net = network(**network_params)
    fwd_net.load_state_dict(weights)
    fwd_net.freeze()
    return fwd_net
