import os

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader

from fft_conv import FFTConv2d, fft_conv_nd

import operators
from operators import l2_error
from torchvision.transforms import ToTensor, Compose, ToPILImage, Resize
from piq import ssim


# ---- utils ----
def change_im_range(img_tens,
                    new_min,
                    new_max):
    old_min = torch.min(img_tens).item()
    old_max = torch.max(img_tens).item()
    ret = ((img_tens - old_min) * ((new_max - new_min) / (old_max - old_min))) + new_min
    return ret


# ----- ----- Abstract Base Network ----- -----
class InvNet(torch.nn.Module, metaclass=ABCMeta):
    """ Abstract base class for networks solving linear inverse problems.

    The network is intended for the denoising of a direct inversion of a 2D
    signal from (noisy) linear measurements. The measurement model

        y = Ax + noise

    can be used to obtain an approximate reconstruction x_ from y using, e.g.,
    the pseudo-inverse of A. The task of the network is either to directly
    obtain x from y or denoise and improve this first inversion x_ towards x.

    """

    def __init__(self):
        super(InvNet, self).__init__()

    @abstractmethod
    def forward(self, z):
        """
        Applies the network to a batch of inputs z, either y or x_ or both.
        """
        pass

    def freeze(self):
        """ Freeze all model weights, i.e. prohibit further updates. """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """ Unfreeze all model weights, i.e. allow further updates. """
        for param in self.parameters():
            param.requires_grad = True

    @property
    def device(self):
        return next(self.parameters()).device

    def _train_step(
        self,
        batch_idx,
        batch,
        loss_func,
        optimizer,
        scaler,
        batch_size,
        acc_steps,
    ):
        with torch.cuda.amp.autocast(enabled=self.mixed_prec):
            if len(batch) == 2:
                inp, tar = batch
                inp = inp.to(self.device)
                tar = tar.to(self.device)
                pred = self.forward(inp)
            else:
                inp, tar, txt_targ = batch
                inp = inp.to(self.device)
                tar = tar.to(self.device)
                pred = self.forward(inp)
            loss = loss_func(pred, tar) / acc_steps
        scaler.scale(loss).backward()
        if (batch_idx // batch_size + 1) % acc_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        return loss * acc_steps, inp, tar, txt_targ, pred

    def _val_step(self, batch_idx, batch, loss_func, do_OCR=True, compute_SSIM=False, add_bg_for_val=None):
        if len(batch) == 2:
            inp, tar = batch
            inp = inp.to(self.device)
            tar = tar.to(self.device)
            pred = self.forward(inp)
        else:
            inp, tar, txt_targ = batch
            inp = inp.to(self.device)
            tar = tar.to(self.device)
            pred = self.forward(inp)
        if add_bg_for_val is not None:
            tar += add_bg_for_val
            pred += add_bg_for_val
        loss = loss_func(pred, tar)
        if do_OCR:
            OCR_score = operators.OCR_score(pred, txt_targ)
        else:
            OCR_score = [0]
        if compute_SSIM:
            min_pr = torch.min(pred)
            max_pr = torch.max(pred).item()
            max_tr = torch.max(tar).item()
            min_tr = torch.min(tar)
            min_tot = min(min_pr.item(), min_tr.item())
            ssim_l = ssim(pred - min_tot, tar - min_tot, data_range=max(1.0, max_pr, max_tr) - min_tot)
        else:
            ssim_l = [0]
        return loss, OCR_score, inp, tar, txt_targ, pred, ssim_l

    def _on_epoch_end(
        self,
        epoch,
        save_epochs,
        save_path,
        logging,
        loss,
        inp,
        tar,
        txt_targ,
        pred,
        v_loss,
        v_inp,
        v_tar,
        v_txt_targ,
        v_pred,
        val_data,
        rel_err_val,
        OCR_score,
        SSIM_score,
        optimizer,
        multi_val=False,
        compute_SSIM=False
    ):

        self._print_info()

        if not multi_val:
            logging_tail = {
                    "loss": loss.item(),
                    "val_loss": v_loss.item(),
                    "rel_l2_error": l2_error(
                        pred, tar, relative=True, squared=False
                    )[0].item(),
                    "val_rel_l2_error": rel_err_val,
                    "OCR_score": operators.OCR_score(
                        pred, txt_targ)[0],  #  OCR_score for training is computed only on last batch of each epoch
                    "val_OCR_score": OCR_score,
                }
        else:
            logging_tail = {
                    "loss": loss.item(),
                    "val_loss": v_loss["HDC"].item(),
                    "val_sim_loss": v_loss["Sim"].item(),
                    "val_san_loss": v_loss["San"].item(),
                    "rel_l2_error": l2_error(
                        pred, tar, relative=True, squared=False
                    )[0].item(),
                    "val_rel_l2_error": rel_err_val,
                    "OCR_score": operators.OCR_score(
                        pred, txt_targ)[0],  # OCR_score for training is computed only on last batch of each epoch
                    "val_OCR_score": OCR_score["HDC"],
                    "val_sim_OCR_score": OCR_score["Sim"]
                }
            if compute_SSIM:
                min_pr = torch.min(pred)
                max_pr = torch.max(pred).item()
                max_tr = torch.max(tar).item()
                min_tar = torch.min(tar)
                min_tot = min(min_pr.item(), min_tar.item())
                logging_tail["SSIM"] = ssim(pred - min_tot, tar - min_tot, data_range=max(1.0, max_pr, max_tr) - min_tot).item()
                logging_tail["val_SSIM"] = SSIM_score["HDC"]
                logging_tail["val_sim_SSIM"] = SSIM_score["Sim"]
                logging_tail["val_san_SSIM"] = SSIM_score["San"]

        logging = logging.append(logging_tail, ignore_index=True, sort=False)
        
        print(logging.tail(1))

        if (epoch + 1) % save_epochs == 0:
            fig = self._create_figure(
                logging, loss, inp, tar, pred, v_loss, v_inp, v_tar, v_txt_targ, v_pred, multi_val=multi_val, compute_SSIM=compute_SSIM
            )

            os.makedirs(save_path, exist_ok=True)

            if multi_val:
                san_fig = self._create_figure_sanity(v_inp["San"], v_pred["San"], v_tar["San"])
                san_fig.savefig(
                    os.path.join(save_path, "plot_sanity_epoch{:03d}.png".format(epoch + 1)),
                    bbox_inches="tight",
                )


            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'logging': logging,
            }, os.path.join(
                save_path, "model_checkpoint_epoch{:03d}.tar".format(epoch + 1)
                )
            )

            logging.to_pickle(
                os.path.join(
                    save_path, "losses_epoch{:03d}.pkl".format(epoch + 1)
                ),
            )
            fig.savefig(
                os.path.join(save_path, "plot_epoch{:03d}.png".format(epoch + 1)),
                bbox_inches="tight",
            )

        return logging

    def _create_figure(self,
                       logging,
                       loss,
                       inp,
                       tar,
                       pred,
                       v_loss,
                       v_inp,
                       v_tar,
                       v_txt_targ,
                       v_pred,
                       multi_val=False,
                       compute_SSIM=False):
        if not multi_val:
            return self._create_figure_uni_val(logging,
                                               loss,
                                               inp,
                                               tar,
                                               pred,
                                               v_loss,
                                               v_inp,
                                               v_tar,
                                               v_txt_targ,
                                               v_pred)
        else:
            return self._create_figure_multi_val(logging,
                                               loss,
                                               inp,
                                               tar,
                                               pred,
                                               v_loss,
                                               v_inp,
                                               v_tar,
                                               v_txt_targ,
                                               v_pred)

    def _create_figure_uni_val(self,
                       logging,
                       loss,
                       inp,
                       tar,
                       pred,
                       v_loss,
                       v_inp,
                       v_tar,
                       v_txt_targ,
                       v_pred,
                       multi_val=False,
    ):
        def _implot(sub, im):
            if im.shape[-3] == 2:  # complex image
                p = sub.imshow(
                    torch.sqrt(im.pow(2).sum(-3))[0, :, :].detach().cpu()
                )
            else:  # real image
                p = sub.imshow(im[0, 0, :, :].detach().cpu().type(torch.FloatTensor))
            return p

        fig, subs = plt.subplots(2, 3, clear=True, num=1, figsize=(15, 10))

        # training and validation loss
        subs[0, 0].set_title("losses")
        subs[0, 0].semilogy(logging["loss"], label="train")
        subs[0, 0].semilogy(logging["val_loss"], label="val")
        subs[0, 0].legend()

        # training and validation challenge-loss
        subs[0, 1].set_title("challenge metrics")
        subs[0, 1].plot(logging["OCR_score"], label="train")  # semilogy changes the y-scale from linear to logarithmic
        subs[0, 1].plot(logging["val_OCR_score"], label="val")
        subs[0, 1].legend()

        # validation input
        p10 = _implot(subs[1, 0], v_inp)
        subs[1, 0].set_title("val input")
        plt.colorbar(p10, ax=subs[1, 0])

        # validation output
        p11 = _implot(subs[1, 1], v_pred)
        subs[1, 1].set_title(
            "val prediction:\n OCR_score = \n "
            "{}".format(logging["val_OCR_score"].iloc[-1])
        )
        plt.colorbar(p11, ax=subs[1, 1])

        # validation difference
        p12 = _implot(subs[1, 2], v_pred - v_tar)
        subs[1, 2].set_title(
            "val diff: x0 - x_pred \n val_chall="
            "{:1.2e}".format(logging["val_OCR_score"].iloc[-1])
        )
        plt.colorbar(p12, ax=subs[1, 2])

        # training output
        p02 = _implot(subs[0, 2], pred)
        subs[0, 2].set_title(
            "train prediction:"
        )
        plt.colorbar(p02, ax=subs[0, 2])

        return fig

    def _create_figure_multi_val(self,
                       logging,
                       loss,
                       inp,
                       tar,
                       pred,
                       v_loss,
                       v_inp,
                       v_tar,
                       v_txt_targ,
                       v_pred,
                       compute_SSIM=False
    ):
        def _implot(sub, im, vmin=None, vmax=None):
            p = sub.imshow(torch.squeeze(im).detach().cpu().type(torch.FloatTensor), vmin=vmin, vmax=vmax)
            return p

        fig, subs = plt.subplots(5, 4, clear=True, num=1, figsize=(15, 15))
        # plots of figures
        # training and validation loss
        subs[0, 0].set_title("losses")
        subs[0, 0].semilogy(logging["loss"], label="train")
        subs[0, 0].semilogy(logging["val_loss"], label="hdc_val")
        subs[0, 0].semilogy(logging["val_sim_loss"], label="sim_val")
        subs[0, 0].semilogy(logging["val_san_loss"], label="san_val")
        subs[0, 0].legend()

        # training and validation challenge-loss
        subs[0, 1].set_title("OCR scores")
        subs[0, 1].plot(logging["OCR_score"], label="train")
        subs[0, 1].plot(logging["val_OCR_score"], label="hdc_val")
        subs[0, 1].plot(logging["val_sim_OCR_score"], label="sim_val")
        subs[0, 1].legend()

        # [0, 2] reserved for SSIM
        if compute_SSIM:
            subs[0, 2].set_title("SSIM loss")
            subs[0, 2].plot(logging["SSIM"], label="train")
            subs[0, 2].plot(logging["val_SSIM"], label="hdc_val")
            subs[0, 2].plot(logging["val_sim_SSIM"], label="sim_val")
            subs[0, 2].plot(logging["val_san_SSIM"], label="san_val")
            subs[0, 2].legend()

        # training prediction
        p03 = _implot(subs[0, 3], pred)
        subs[0, 3].set_title(
            "train prediction:"
        )
        plt.colorbar(p03, ax=subs[0, 3])

        # hdc data plots:
        # validation input
        p10 = _implot(subs[1, 0], v_inp["HDC"])
        subs[1, 0].set_title("val input (hdc data)")
        plt.colorbar(p10, ax=subs[1, 0])

        # validation output
        p11 = _implot(subs[1, 1], v_pred["HDC"])
        subs[1, 1].set_title(
            "val prediction:\n avg OCR score = \n "
            "{}".format(logging["val_OCR_score"].iloc[-1])
        )
        plt.colorbar(p11, ax=subs[1, 1])

        # validation target
        p12 = _implot(subs[1, 2], v_tar["HDC"])
        subs[1, 2].set_title("val target")
        plt.colorbar(p12, ax=subs[1, 2])

        # validation difference
        p13 = _implot(subs[1, 3], v_pred["HDC"] - v_tar["HDC"])
        subs[1, 3].set_title(
            "val diff: x_targ - x_pred \n avg l2 loss="
            "{:1.2e}".format(logging["val_loss"].iloc[-1])
        )
        plt.colorbar(p13, ax=subs[1, 3])


        # sim data plots:
        # validation input
        p20 = _implot(subs[2, 0], v_inp["Sim"])
        subs[2, 0].set_title("val input (simulated)")
        plt.colorbar(p20, ax=subs[2, 0])

        # validation output
        p21 = _implot(subs[2, 1], v_pred["Sim"])
        subs[2, 1].set_title(
            "val prediction:\n avg OCR score = \n "
            "{}".format(logging["val_sim_OCR_score"].iloc[-1])
        )
        plt.colorbar(p21, ax=subs[2, 1])

        # validation target
        p22 = _implot(subs[2, 2], v_tar["Sim"])
        subs[2, 2].set_title("val target")
        plt.colorbar(p22, ax=subs[2, 2])

        # validation difference
        p23 = _implot(subs[2, 3], v_pred["Sim"] - v_tar["Sim"])
        subs[2, 3].set_title(
            "val diff: x_targ - x_pred \n avg l2 loss="
            "{:1.2e}".format(logging["val_sim_loss"].iloc[-1])
        )
        plt.colorbar(p23, ax=subs[2, 3])


        # Sanity data plots:
        # validation input
        p30 = _implot(subs[3, 0], v_inp["San"])
        subs[3, 0].set_title("val input (sanity)")
        plt.colorbar(p30, ax=subs[3, 0])

        # validation output
        p31 = _implot(subs[3, 1], v_pred["San"])
        subs[3, 1].set_title("val prediction")
        plt.colorbar(p31, ax=subs[3, 1])

        # validation target
        p32 = _implot(subs[3, 2], v_tar["San"])
        subs[3, 2].set_title("val target")
        plt.colorbar(p32, ax=subs[3, 2])

        # validation difference
        p33 = _implot(subs[3, 3], v_pred["San"] - v_tar["San"])
        subs[3, 3].set_title("val diff: x_targ - x_pred")
        plt.colorbar(p33, ax=subs[3, 3])

        # Sanity data plots:
        # validation input
        p40 = _implot(subs[4, 0], v_inp["PSF"], vmin=0.0, vmax=1.0)
        subs[4, 0].set_title("val input (PSF)")
        plt.colorbar(p40, ax=subs[4, 0])

        # validation output
        p41 = _implot(subs[4, 1], v_pred["PSF"], vmin=0.0, vmax=1.0)
        subs[4, 1].set_title("val prediction")
        plt.colorbar(p41, ax=subs[4, 1])

        # validation target
        p42 = _implot(subs[4, 2], v_tar["PSF"], vmin=0.0, vmax=1.0)
        subs[4, 2].set_title("val target")
        plt.colorbar(p42, ax=subs[4, 2])

        # validation difference
        p43 = _implot(subs[4, 3], v_pred["PSF"] - v_tar["PSF"])
        subs[4, 3].set_title("val diff: x_targ - x_pred")
        plt.colorbar(p43, ax=subs[4, 3])

        return fig

    def _create_figure_sanity(self, v_inp, v_pred, v_tar):

        def _implot_res(sub, im, vmin=-1., vmax=1.):
            # resize = Compose([ToPILImage(), Resize((1460, 2360)), ToTensor()])
            im_min = torch.min(im).item()
            im_max = torch.max(im).item()
            tmp_im = change_im_range(im, new_min=0.0, new_max=1.0)
            if im.shape[-3] == 2:  # complex image
                p = sub.imshow(
                    torch.sqrt(im.pow(2).sum(-3))[0, :, :].detach().cpu()
                )
            else:  # real image
                # tmp_im = resize(im[0, 0, :, :].detach().cpu().type(torch.FloatTensor)).squeeze()
                tmp_im = im.squeeze()
                tmp_im = change_im_range(tmp_im, new_min=im_min, new_max=im_max)
                p = sub.imshow(tmp_im.detach().cpu().squeeze(), cmap="Greys", vmin=vmin, vmax=vmax)
            return p

        fig_san, subs_san = plt.subplots(2, 3, clear=True, num=2, figsize=(30, 20))

        p00 = _implot_res(subs_san[0, 0], v_inp)
        subs_san[0, 0].set_title(
            "Sanity prediction")
        plt.colorbar(p00, ax=subs_san[0, 0])

        p01 = _implot_res(subs_san[0, 1], v_pred)
        subs_san[0, 1].set_title(
            "Sanity prediction")
        plt.colorbar(p01, ax=subs_san[0, 1])

        p02 = _implot_res(subs_san[0, 2], v_tar)
        subs_san[0, 2].set_title(
            "Sanity target")
        plt.colorbar(p02, ax=subs_san[0, 2])



        return fig_san


    def _add_to_progress_bar(self, dict):
        """ Can be overwritten by child classes to add to progress bar. """
        return dict

    def _on_train_end(self, save_path, logging, epoch, optimizer):
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            self.state_dict(), os.path.join(save_path, "model_weights.pt")
        )
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'logging': logging,
        }, os.path.join(save_path, "model_checkpoint.tar"))
        logging.to_pickle(os.path.join(save_path, "losses.pkl"),)

    def _print_info(self):
        """ Can be overwritten by child classes to print at epoch end. """
        pass

    def train_on(
        self,
        train_data,
        val_data,
        num_epochs,
        batch_size,
        loss_func,
        save_path,
        save_epochs=50,
        optimizer=torch.optim.Adam,
        optimizer_params={"lr": 2e-4, "eps": 1e-3},
        scheduler=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 1, "gamma": 1.0},
        acc_steps=1,
        train_transform=None,
        val_transform=None,
        train_loader_params={"shuffle": True},
        val_loader_params={"shuffle": False},
        mixed_prec=False,
        checkpoint_path=None,
        finetune=False,
        multi_val=False,
        compute_SSIM=False,
        add_bg_for_val=None,
    ):
        self.mixed_prec = mixed_prec
        scaler = torch.cuda.amp.GradScaler(enabled=mixed_prec)
        optimizer = optimizer(self.parameters(), **optimizer_params)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint['model_state_dict'])  # recover weights
            if not finetune:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = scheduler(optimizer, **scheduler_params)

        if multi_val:
            assert(isinstance(val_data, ConcatDataset))
            val_sets = {}
            for val_set in val_data.datasets:
                if hasattr(val_set, 'type'):
                    val_sets[val_set.type] = val_set
                else:
                    print("ERROR: Val_dataset should be concat of base datasets")
                    assert False

        if not multi_val and isinstance(val_data, ConcatDataset):
            print("WARNING: Set 'multi_val' to True if 4x4grid plot w/ different val evaluations is wanted.")

        train_loader_params = dict(train_loader_params)
        val_loader_params = dict(val_loader_params)
        if "sampler" in train_loader_params:
            train_loader_params["sampler"] = train_loader_params["sampler"](
                train_data
            )
        if "sampler" in val_loader_params:
            val_loader_params["sampler"] = val_loader_params["sampler"](
                val_data
            )

        data_load_train = DataLoader(
            train_data, batch_size, **train_loader_params
        )
        if not multi_val:
            data_load_val = DataLoader(val_data,
                                       batch_size,
                                       **val_loader_params
                                       )
            logging = pd.DataFrame(
                columns=["loss", "val_loss", "rel_l2_error", "val_rel_l2_error"]
            )
        else:
            data_load_val = {}
            for key in val_sets:
                data_load_val[key] = DataLoader(val_sets[key],
                                                batch_size,
                                                **val_loader_params
                                                )
            logging = pd.DataFrame(
                columns=["loss", "val_loss", "val_sim_loss",
                         "rel_l2_error", "val_rel_l2_error"])
        if checkpoint_path:
            logging = checkpoint['logging']
            last_epoch = checkpoint['epoch']
        if checkpoint_path is None:
            last_epoch = 0
        for epoch in range(last_epoch, num_epochs):
            self.train()  # make sure we are in train mode
            t = tqdm(
                enumerate(data_load_train),
                desc="epoch {} / {}".format(epoch + 1, num_epochs),
                total=-(-len(train_data) // batch_size),
                disable="SGE_TASK_ID" in os.environ,
            )
            optimizer.zero_grad()
            loss = 0.0
            for i, batch in t:
                loss_b, inp, tar, txt_targ, pred = self._train_step(
                    i,
                    batch,
                    loss_func,
                    optimizer,
                    scaler,
                    batch_size,
                    acc_steps,
                )
                t.set_postfix(
                    **self._add_to_progress_bar({"loss": loss_b.item()})
                )
                loss += loss_b
            loss /= i + 1

            with torch.no_grad():
                self.eval()  # make sure we are in eval mode
                scheduler.step()

                if not multi_val:
                    v_loss = 0.0
                    rel_err_val = 0.0
                    OCR_score = 0.0
                    SSIM_score = 0.0
                    for i, v_batch in enumerate(data_load_val):
                        v_loss_b, v_OCR_score, v_inp, v_tar, v_txt_targ, v_pred, _ = self._val_step(
                            i, v_batch, loss_func, add_bg_for_val=add_bg_for_val
                        )
                        OCR_score += v_OCR_score[0]

                        v_loss += v_loss_b
                    v_loss /= i + 1
                    # rel_err_val /= i + 1
                    OCR_score /= i + 1

                else:
                    val_inp = {}
                    val_tar = {}
                    val_txt_targ = {}
                    val_pred = {}

                    v_loss = {"HDC": 0.0, "Sim": 0.0, "San": 0.0, "PSF": 0.0}
                    rel_err_val = 0.0
                    OCR_score = {"HDC": 0.0, "Sim": 0.0, "San": 0, "PSF": 0.0}
                    SSIM_score = {"HDC": 0.0, "Sim": 0.0, "San": 0.0, "PSF": 0.0}
                    for key in val_sets:
                        for i, v_batch in enumerate(data_load_val[key]):
                            do_OCR = not (key == "San" or key == "PSF")
                            v_loss_b, v_OCR_score, val_inp[key], val_tar[key], val_txt_targ[key], val_pred[key], v_ssim = self._val_step(i,
                                                                                                     v_batch,
                                                                                                     loss_func,
                                                                                                     do_OCR=do_OCR,
                                                                                                     compute_SSIM=compute_SSIM,
                                                                                                     add_bg_for_val=add_bg_for_val)

                            OCR_score[key] += v_OCR_score[0]
                            if compute_SSIM:
                                SSIM_score[key] += v_ssim.item()
                            v_loss[key] += v_loss_b
                        v_loss[key] /= i + 1
                        # rel_err_val /= i + 1
                        OCR_score[key] /= i + 1
                        if compute_SSIM:
                            SSIM_score[key] /= i + 1
                    v_inp = val_inp
                    v_tar = val_tar
                    v_txt_targ = val_txt_targ
                    v_pred = val_pred


                logging = self._on_epoch_end(
                    epoch,
                    save_epochs,
                    save_path,
                    logging,
                    loss,
                    inp,
                    tar,
                    txt_targ,
                    pred,
                    v_loss,
                    v_inp,
                    v_tar,
                    v_txt_targ,
                    v_pred,
                    val_data,
                    rel_err_val,
                    OCR_score,
                    SSIM_score,
                    optimizer,
                    multi_val=multi_val,
                    compute_SSIM=compute_SSIM
                )

        self._on_train_end(save_path, logging, epoch, optimizer)
        return logging


# ----- ----- U-Net ----- -----
class DeepDeepUNet_v2_big(InvNet):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 base_features=32,
                 drop_factor=0.0,
                 do_center_crop=False,
                 intro_pooling=2,
                 num_groups=32):
        super(DeepDeepUNet_v2_big, self).__init__()

        self.do_center_crop = do_center_crop
        kernel_size = 3 if do_center_crop else 2

        self.pool0 = torch.nn.AvgPool2d(kernel_size=intro_pooling, stride=intro_pooling)

        self.encoder1 = self._conv_block(
            in_channels,
            base_features,
            num_groups,
            drop_factor=drop_factor,
            block_name="encoding_1",
        )
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self._conv_block(
            base_features,
            base_features * 2,
            num_groups,
            drop_factor=drop_factor,
            block_name="encoding_2",
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self._conv_block(
            base_features * 2,
            base_features * 4,
            num_groups,
            drop_factor=drop_factor,
            block_name="encoding_3",
        )
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self._conv_block(
            base_features * 4,
            base_features * 8,
            num_groups,
            drop_factor=drop_factor,
            block_name="encoding_4",
        )
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder5 = self._conv_block(
            base_features * 8,
            base_features * 16,
            num_groups,
            drop_factor=drop_factor,
            block_name="encoding_5",
        )
        self.pool5 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder6 = self._conv_block(
            base_features * 16,
            base_features * 32,
            num_groups,
            drop_factor=drop_factor,
            block_name="encoding_6",
        )
        self.pool6 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._conv_block(
            base_features * 32,
            base_features * 64,
            num_groups,
            drop_factor=drop_factor,
            block_name="bottleneck",
        )

        self.upconv6 = torch.nn.ConvTranspose2d(
            base_features * 64,
            base_features * 32,
            kernel_size=kernel_size,
            stride=2,
        )
        self.decoder6 = self._conv_block(
            base_features * 64,
            base_features * 32,
            num_groups,
            drop_factor=drop_factor,
            block_name="decoding_6",
        )

        self.upconv5 = torch.nn.ConvTranspose2d(
            base_features * 32,
            base_features * 16,
            kernel_size=kernel_size,
            stride=2,
        )
        self.decoder5 = self._conv_block(
            base_features * 32,
            base_features * 16,
            num_groups,
            drop_factor=drop_factor,
            block_name="decoding_5",
        )

        self.upconv4 = torch.nn.ConvTranspose2d(
            base_features * 16,
            base_features * 8,
            kernel_size=kernel_size,
            stride=2,
        )
        self.decoder4 = self._conv_block(
            base_features * 16,
            base_features * 8,
            num_groups,
            drop_factor=drop_factor,
            block_name="decoding_4",
        )
        self.upconv3 = torch.nn.ConvTranspose2d(
            base_features * 8,
            base_features * 4,
            kernel_size=kernel_size,
            stride=2,
        )
        self.decoder3 = self._conv_block(
            base_features * 8,
            base_features * 4,
            num_groups,
            drop_factor=drop_factor,
            block_name="decoding_3",
        )
        self.upconv2 = torch.nn.ConvTranspose2d(
            base_features * 4,
            base_features * 2,
            kernel_size=kernel_size,
            stride=2,
        )
        self.decoder2 = self._conv_block(
            base_features * 4,
            base_features * 2,
            num_groups,
            drop_factor=drop_factor,
            block_name="decoding_2",
        )
        self.upconv1 = torch.nn.ConvTranspose2d(
            base_features * 2, base_features, kernel_size=kernel_size, stride=2
        )
        self.decoder1 = self._conv_block(
            base_features * 2,
            base_features,
            num_groups,
            drop_factor=drop_factor,
            block_name="decoding_1",
        )

        self.upconv0 = torch.nn.ConvTranspose2d(
            base_features,
            out_channels,
            kernel_size=kernel_size,
            stride=intro_pooling,
        )

        self.outconv = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
        )


    def forward(self, x):
        # print('x', x.shape)
        enc1 = self.encoder1(self.pool0(x))
        # print('enc1', enc1.shape)
        enc2 = self.encoder2(self.pool1(enc1))
        # print('enc2', enc2.shape)
        enc3 = self.encoder3(self.pool2(enc2))
        # print('enc3', enc3.shape)
        enc4 = self.encoder4(self.pool3(enc3))

        enc5 = self.encoder5(self.pool4(enc4))

        enc6 = self.encoder6(self.pool5(enc5))

        # print('enc4', enc4.shape)
        bottleneck = self.bottleneck(self.pool6(enc6))
        # print('bottleneck', bottleneck.shape)

        dec6 = self.upconv6(bottleneck)
        # print('dec4', dec4.shape)
        dec6 = self._center_crop(dec6, enc6.shape[-2], enc6.shape[-1])
        # print('dec4_crop', dec4.shape)
        dec6 = torch.cat((dec6, enc6), dim=1)
        dec6 = self.decoder6(dec6)


        dec5 = self.upconv5(dec6)
        # print('dec4', dec4.shape)
        dec5 = self._center_crop(dec5, enc5.shape[-2], enc5.shape[-1])
        # print('dec4_crop', dec4.shape)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)


        dec4 = self.upconv4(dec5)
        # print('dec4', dec4.shape)
        dec4 = self._center_crop(dec4, enc4.shape[-2], enc4.shape[-1])
        # print('dec4_crop', dec4.shape)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self._center_crop(dec3, enc3.shape[-2], enc3.shape[-1])
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self._center_crop(dec2, enc2.shape[-2], enc2.shape[-1])
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self._center_crop(dec1, enc1.shape[-2], enc1.shape[-1])
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        dec0 = self.upconv0(dec1)
        dec0 = self._center_crop(dec0, x.shape[-2], x.shape[-1])
        # dec0 = torch.cat((dec0, x), dim=1)
        # dec0 = self.decoder1(dec0)
        return self.outconv(dec0)


    def _conv_block(
        self, in_channels, out_channels, num_groups, drop_factor, block_name
    ):
        return torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        block_name + "conv1",
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (
                        block_name + "bn_1",
                        torch.nn.GroupNorm(num_groups, out_channels),
                    ),
                    (block_name + "relu1", torch.nn.ReLU(True)),
                    (block_name + "dr1", torch.nn.Dropout(p=drop_factor)),
                    (
                        block_name + "conv2",
                        torch.nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (
                        block_name + "bn_2",
                        torch.nn.GroupNorm(num_groups, out_channels),
                    ),
                    (block_name + "relu2", torch.nn.ReLU(True)),
                    (block_name + "dr2", torch.nn.Dropout(p=drop_factor)),
                ]
            )
        )

    def _center_crop(self, layer, max_height, max_width):
        if self.do_center_crop:
            _, _, h, w = layer.size()
            xy1 = (w - max_width) // 2
            xy2 = (h - max_height) // 2
            return layer[
                :, :, xy2 : (xy2 + max_height), xy1 : (xy1 + max_width)
            ]
        else:
            return layer


# ----- ----- Forward Operator ----- ------
class BlurOp(InvNet):
    def __init__(
        self,
        inp_size,
        init_kernel=None,
        kernel_size=701,
        rd_params=None,
        rd_fac=1e-3,
    ):
        super(BlurOp, self).__init__()
        assert kernel_size % 2 == 1     #  kernel size must be uneven in order to simplify the passing of the padding argument

        self.conv2d = FFTConv2d(in_channels=1,
                                out_channels=1,
                                kernel_size=kernel_size,
                                padding=kernel_size // 2,
                                padding_mode='reflect', #'replicate',
                                bias=False,)

        if init_kernel is not None:
            self.conv2d.weight.data = init_kernel.expand(self.conv2d.weight.shape).clone()
        else:
            torch.nn.init.zeros_(self.conv2d.weight)

        if rd_params is None:
            rd_params = torch.zeros(2, requires_grad=False)

        self.rd_fac = torch.nn.Parameter(torch.tensor(rd_fac), requires_grad=False)
        self.rd_params = torch.nn.Parameter(rd_fac * rd_params, requires_grad=rd_params.requires_grad)

        dom_x = torch.linspace(-1.0, 1.0, inp_size)
        dom_y = torch.linspace(-1.0, 1.0, inp_size)
        self.grid_x, self.grid_y = torch.meshgrid(dom_x, dom_y)
        self.grid_x = torch.nn.Parameter(self.grid_x.unsqueeze(0).clone(), requires_grad=False)
        self.grid_y = torch.nn.Parameter(self.grid_y.unsqueeze(0).clone(), requires_grad=False)
        self.rad = torch.nn.Parameter((self.grid_x ** 2 + self.grid_y ** 2).sqrt(), requires_grad=False)


    def undistort(self, x):
        rad_fac_x = 1 + self.rd_params[0] / self.rd_fac * self.rad ** 2 + \
                    self.rd_params[1] / self.rd_fac * self.rad ** 4
        rad_fac_y = 1 + self.rd_params[0] / self.rd_fac * self.rad ** 2 + \
                    self.rd_params[1] / self.rd_fac * self.rad ** 4
        rad_grid_x = self.grid_x / rad_fac_x
        rad_grid_y = self.grid_y / rad_fac_y
        rad_grid = torch.cat([rad_grid_x.unsqueeze(-1), rad_grid_y.unsqueeze(-1)], -1)

        size_diff = x.shape[-1] - x.shape[-2]
        assert size_diff >= 0
        out_padded = torch.nn.functional.pad(x, (0, 0, size_diff // 2, size_diff // 2), "reflect")
        out_distorted = torch.nn.functional.grid_sample(out_padded, rad_grid.transpose(1, 2), padding_mode="reflection")
        out_distorted = out_distorted[..., size_diff // 2:(x.shape[-1] - size_diff // 2), :]

        return out_distorted

    def distort(self, x):
        rad_fac_x = 1 + self.rd_params[0] / self.rd_fac * self.rad ** 2 + \
                    self.rd_params[1] / self.rd_fac * self.rad ** 4
        rad_fac_y = 1 + self.rd_params[0] / self.rd_fac * self.rad ** 2 + \
                    self.rd_params[1] / self.rd_fac * self.rad ** 4
        rad_grid_x = self.grid_x * rad_fac_x  # .expand(x.shape)
        rad_grid_y = self.grid_y * rad_fac_y
        rad_grid = torch.cat([rad_grid_x.unsqueeze(-1), rad_grid_y.unsqueeze(-1)], -1)

        size_diff = x.shape[-1] - x.shape[-2]
        assert size_diff >= 0
        out_padded = torch.nn.functional.pad(x, (0, 0, size_diff // 2, size_diff // 2), "reflect")
        out_distorted = torch.nn.functional.grid_sample(out_padded, rad_grid.transpose(1, 2), padding_mode="reflection")
        out_distorted = out_distorted[..., size_diff // 2:(x.shape[-1] - size_diff // 2), :]
        return out_distorted

    def forward(self, x):
        out = self.conv2d(x)
        return self.distort(out)

    def _train_step(
        self,
        batch_idx,
        batch,
        loss_func,
        optimizer,
        scaler,
        batch_size,
        acc_steps,
    ):
        with torch.cuda.amp.autocast(enabled=self.mixed_prec):
            if len(batch) == 2:
                inp, tar = batch
                inp = inp.to(self.device)
                tar = tar.to(self.device)
                pred = self.forward(inp)
            else:
                tar, inp, txt_targ = batch
                inp = inp.to(self.device)
                tar = tar.to(self.device)
                pred = self.forward(inp)
            loss = loss_func(pred, tar) / acc_steps
        scaler.scale(loss).backward()
        if (batch_idx // batch_size + 1) % acc_steps == 0:
            scaler.step(optimizer)
            scaler.update()

            # kernel non negativity
            self.conv2d.weight.data = self.conv2d.weight.data.clamp_min(0.0)

            optimizer.zero_grad()
        return loss * acc_steps, inp, tar, txt_targ, pred

    def _val_step(self, batch_idx, batch, loss_func):
        if len(batch) == 2:
            inp, tar = batch
            inp = inp.to(self.device)
            tar = tar.to(self.device)
            pred = self.forward(inp)
        else:
            tar, inp, txt_targ = batch   #  inp and tar switch places here
            inp = inp.to(self.device)
            tar = tar.to(self.device)
            pred = self.forward(inp)
        loss = loss_func(pred, tar)
        OCR_score = 0
        return loss, OCR_score, inp, tar, txt_targ, pred

    def _on_epoch_end(
            self,
            epoch,
            save_epochs,
            save_path,
            logging,
            loss,
            inp,
            tar,
            txt_targ,
            pred,
            v_loss,
            v_inp,
            v_tar,
            v_txt_targ,
            v_pred,
            val_data,
            rel_err_val,
            OCR_score,
            optimizer
    ):

        self._print_info()

        logging = logging.append(
            {
                "loss": loss.item(),
                "val_loss": v_loss.item(),
                "rel_l2_error": l2_error(
                    pred, tar, relative=True, squared=False
                )[0].item(),
                "val_rel_l2_error": rel_err_val,
            },
            ignore_index=True,
            sort=False,
        )

        print(logging.tail(1))

        if (epoch + 1) % save_epochs == 0:
            fig = self._create_figure(
                logging, loss, inp, tar, pred, v_loss, v_inp, v_tar, v_txt_targ, v_pred
            )

            os.makedirs(save_path, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'logging': logging,
            }, os.path.join(
                save_path, "model_checkpoint_epoch{:03d}.tar".format(epoch + 1)
                )
            )
            torch.save(
                self.state_dict(),
                os.path.join(
                    save_path, "model_weights_epoch{:03d}.pt".format(epoch + 1)
                ),
            )
            logging.to_pickle(
                os.path.join(
                    save_path, "losses_epoch{:03d}.pkl".format(epoch + 1)
                ),
            )
            fig.savefig(
                os.path.join(save_path, "plot_epoch{:03d}.png".format(epoch + 1)),
                bbox_inches="tight",
            )

        return logging

    def _create_figure(
            self, logging, loss, inp, tar, pred, v_loss, v_inp, v_tar, v_txt_targ, v_pred
    ):
        def _implot(sub, im, vmin=None, vmax=None):
            if im.shape[-3] == 2:  # complex image
                p = sub.imshow(
                    torch.sqrt(im.pow(2).sum(-3))[0, :, :].detach().cpu(), vmin=vmin, vmax=vmax
                )
            else:  # real image
                p = sub.imshow(im[0, 0, :, :].detach().cpu(), vmin=vmin, vmax=vmax)
            return p

        fig, subs = plt.subplots(2, 4, clear=True, num=1, figsize=(20, 20))

        # training and validation loss
        subs[0, 0].set_title("losses")
        subs[0, 0].semilogy(logging["loss"], label="train")
        subs[0, 0].semilogy(logging["val_loss"], label="val")
        subs[0, 0].legend()

        p = _implot(subs[0, 1], pred, vmin=tar.min().item(), vmax=tar.max().item())
        subs[0, 1].set_title("train prediction")
        plt.colorbar(p, ax=subs[0, 1])

        p = _implot(subs[0, 2], tar)
        subs[0, 2].set_title("train target")
        plt.colorbar(p, ax=subs[0, 2])


        p = _implot(subs[0, 3], pred - tar)
        subs[0, 3].set_title(
            "train diff: x0 - x_pred \n rel l2 error = "
            "{:1.2e}".format(logging["rel_l2_error"].iloc[-1])
        )
        plt.colorbar(p, ax=subs[0, 3])

        subs[1, 0].set_title("kernel")
        p = subs[1, 0].matshow(self.conv2d.weight[0,0,...].squeeze().detach().cpu())
        plt.colorbar(p, ax=subs[1, 0])

        # validation output
        p = _implot(subs[1, 1], v_pred, vmin=v_tar.min().item(), vmax=v_tar.max().item())
        subs[1, 1].set_title("val prediction")
        plt.colorbar(p, ax=subs[1, 1])

        # validation output
        p = _implot(subs[1, 2], v_tar)
        subs[1, 2].set_title("val target")
        plt.colorbar(p, ax=subs[1, 2])

        # validation target
        p = _implot(subs[1, 3], v_pred - v_tar)
        subs[1, 3].set_title("val diff")
        plt.colorbar(p, ax=subs[1, 3])

        return fig

    def train_on(
            self,
            train_data,
            val_data,
            num_epochs,
            batch_size,
            loss_func,
            save_path,
            save_epochs=50,
            optimizer=torch.optim.Adam,
            optimizer_params={"lr": 2e-4, "eps": 1e-3},
            scheduler=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"step_size": 1, "gamma": 1.0},
            acc_steps=1,
            train_transform=None,
            val_transform=None,
            train_loader_params={"shuffle": True},
            val_loader_params={"shuffle": False},
            mixed_prec=False,
            loss_penalty=None,
            checkpoint_path=None
    ):
        self.mixed_prec = mixed_prec
        scaler = torch.cuda.amp.GradScaler(enabled=mixed_prec)
        optimizer = optimizer(self.parameters(), **optimizer_params)
        if checkpoint_path:  # recover weights and state of optimizer
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = scheduler(optimizer, **scheduler_params)

        if loss_penalty is not None:
            loss_func_tmp = loss_func
            loss_func = lambda in1,in2 : loss_func_tmp(in1,in2) + loss_penalty(self)

        train_loader_params = dict(train_loader_params)
        val_loader_params = dict(val_loader_params)
        if "sampler" in train_loader_params:
            train_loader_params["sampler"] = train_loader_params["sampler"](
                train_data
            )
        if "sampler" in val_loader_params:
            val_loader_params["sampler"] = val_loader_params["sampler"](
                val_data
            )

        data_load_train = torch.utils.data.DataLoader(
            train_data, batch_size, **train_loader_params
        )
        data_load_val = torch.utils.data.DataLoader(
            val_data, batch_size, **val_loader_params
        )

        logging = pd.DataFrame(
            columns=["loss", "val_loss", "rel_l2_error", "val_rel_l2_error"]
        )
        if checkpoint_path:
            logging = checkpoint['logging']
            last_epoch = checkpoint['epoch']
        if checkpoint_path is None:
            last_epoch = 0

        for epoch in range(last_epoch, num_epochs):
            self.train()  # make sure we are in train mode
            t = tqdm(
                enumerate(data_load_train),
                desc="epoch {} / {}".format(epoch + 1, num_epochs),
                total=-(-len(train_data) // batch_size),
                disable="SGE_TASK_ID" in os.environ,
            )
            optimizer.zero_grad()
            loss = 0.0
            for i, batch in t:
                loss_b, inp, tar, txt_targ, pred = self._train_step(
                    i,
                    batch,
                    loss_func,
                    optimizer,
                    scaler,
                    batch_size,
                    acc_steps,
                )
                t.set_postfix(
                    **self._add_to_progress_bar({"loss": loss_b.item()})
                )
                loss += loss_b
            loss /= i + 1

            with torch.no_grad():
                self.eval()  # make sure we are in eval mode
                scheduler.step()
                v_loss = 0.0
                rel_err_val = 0.0
                OCR_score = 0.0
                for i, v_batch in enumerate(data_load_val):
                    v_loss_b, v_OCR_score, v_inp, v_tar, v_txt_targ, v_pred = self._val_step(
                        i, v_batch, loss_func
                    )
                    rel_err_val += l2_error(
                        v_pred, v_tar, relative=True, squared=False
                    )[0].item()

                    # chall_err_val += l2_error(
                    #     v_pred, v_tar, relative=False, squared=False
                    # )[0].item() / np.sqrt(v_pred.shape[-1] * v_pred.shape[-2])
                    v_loss += v_loss_b
                v_loss /= i + 1
                rel_err_val /= i + 1
                # chall_err_val /= i + 1

                logging = self._on_epoch_end(
                    epoch,
                    save_epochs,
                    save_path,
                    logging,
                    loss,
                    inp,
                    tar,
                    txt_targ,
                    pred,
                    v_loss,
                    v_inp,
                    v_tar,
                    v_txt_targ,
                    v_pred,
                    val_data,
                    rel_err_val,
                    OCR_score,
                    optimizer
                )

        self._on_train_end(save_path, logging, epoch, optimizer)
        return logging
