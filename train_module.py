import os
import shutil
import matplotlib as mpl
import torch
from networks import InvNet
# ----- load configuration -----
import config  # isort:skip


_mseloss = torch.nn.MSELoss(reduction="sum")
def mse_rel_loss(pred, tar):
    return _mseloss(pred, tar) / pred.shape[0]


class train(object):
    def __init__(self,
                 network,
                 dataset,
                 target_folder,
                 network_params,
                 train_params,
                 train_data_params=None,
                 val_data_params=None,
                 loss_func=mse_rel_loss,
                 steps_to_train=None,
                 train_phases=1,
                 checkpoint_name=None,
                 init_path=None,
                 finetune_path=None,
                 add_bg_for_val=None
                 ):
        if train_data_params is None:
            self.train_data_params = {"val_split": 10}
        else:
            self.train_data_params = train_data_params

        if val_data_params is None:
            self.val_data_params = {"val_split": 10}
        else:
            self.val_data_params = val_data_params

        if steps_to_train is None:
            self.steps_to_train = list(range(20))
        else:
            self.steps_to_train = steps_to_train

        assert(issubclass(network, InvNet))
        self.network = network
        self.dataset = dataset
        self.network_params = network_params
        self.train_params = train_params
        assert("loss_func" not in train_params)  # the loss func is to be handed over seperately
        self.train_params["loss_func"] = loss_func
        assert ("save_path" not in train_params)  # the loss func is to be handed over seperately
        super_save_path = os.path.join(config.RESULTS_PATH, target_folder)
        self.train_params["save_path"] = super_save_path
        self.train_phases = train_phases
        self.train_params["save_path"] = super_save_path
        self.path_adapted = False
        self.checkpoint_name = checkpoint_name
        self.init_path = init_path
        self.hyperparams = {"network": str(network),
                            "network_params": network_params,
                            "train_params": train_params}
        self.finetune_path = finetune_path
        self.add_bg_for_val = add_bg_for_val


    def _change_save_path(self, step, train_phase, **kwargs):
        os.makedirs(self.train_params["save_path"], exist_ok=True)
        if not self.path_adapted:
            self.train_params["save_path"] = os.path.join(self.train_params["save_path"], "step_" + str(step).zfill(2))
            os.makedirs(self.train_params["save_path"], exist_ok=True)
            if self.train_phases > 1:
                self.train_params["save_path"] = os.path.join(self.train_params["save_path"], "phase_" + str(train_phase))
                os.makedirs(self.train_params["save_path"], exist_ok=True)
            self.path_adapted = True
            return self.train_params["save_path"]
        else:
            raise(Exception("The path has already been adapted, _change_save_path should be called only once on each instance."))

    def _save_hyperparameters(self, step, train_phase, path, **kwargs):
        os.makedirs(self.train_params["save_path"], exist_ok=True)  # create superdirectory, where directories for steps
        self.hyperparams["train_params"]["save_path"] = self.train_params["save_path"]

        torch.save(self.hyperparams, os.path.join(path, 'hyperparameters.pkl'))

        with open(
                os.path.join(path, "hyperparameters.txt"), "w"
        ) as file:
            file.write("Step : " + str(step) + "\n")
            file.write("train_phase" + ": " + str(train_phase) + "/" + str(self.train_phases) + "\n")
            for key, value in self.network_params.items():
                file.write(key + ": " + str(value) + "\n")
            # for key, value in it_net_params.items():
            #     file.write(key + ": " + str(value) + "\n")
            for key, value in self.train_params.items():
                file.write(key + ": " + str(value) + "\n")
            for key, value in self.train_data_params.items():
                file.write("[train_data] " + key + ": " + str(value) + "\n")
            for key, value in self.val_data_params.items():
                file.write("[val_data] " + key + ": " + str(value) + "\n")
        pass

    def _current_step_and_train_phase(self, task_id, mode='steps_first'):
        if mode == 'steps_first':
            train_phase = (task_id - 1) // len(self.steps_to_train)
            step = self.steps_to_train[(task_id - 1) % len(self.steps_to_train)]
        elif mode == 'tasks_first':
            step = self.steps_to_train[(task_id - 1) // self.train_phases]
            train_phase = (task_id - 1) % self.train_phases
        else:
            raise(Exception("ERROR: mode needs to be either 'steps_first' or 'tasks_first'."))
        return step, train_phase

    def _copy_plots_for_quick_access(self, save_logging=True):
        """
        copies the last epoch plots and hyperparameters to quick access folder and returns path to target location
        :return:
        """
        destination = self.train_params["save_path"].split('/')
        destination[0] = "results_quick"
        destination_path = os.path.join(*destination)
        os.makedirs(destination_path, exist_ok=True)
        last_epoch = self.train_params["num_epochs"]
        # ----- copy hyperparameters -----
        shutil.copyfile(
            os.path.join(
                self.train_params["save_path"], "hyperparameters.txt"
            ),
            os.path.join(
                destination_path, "hyperparameters.txt"
            ),
        )

        # ---- copy last epoch plots to destination ----
        shutil.copyfile(
            os.path.join(
                self.train_params["save_path"], "plot_epoch{:03d}.png".format(last_epoch)
            ),
            os.path.join(
                destination_path, "plot_epoch{:03d}.png".format(last_epoch)
            ),
        )

        # ---- copy logging if logging = True ----
        if save_logging:
            shutil.copyfile(
                os.path.join(
                    self.train_params["save_path"], "losses.pkl".format(last_epoch)
                ),
                os.path.join(
                    destination_path, "losses.pkl".format(last_epoch)
                ),
            )
        return destination_path

    def _save_best_epoch(self, step, logging):
        # ---- save last epoch for quick access and get quick access destination_path -----
        quick_access_path = self._copy_plots_for_quick_access()

        # ------ optimal OCR score: -----
        optimal_OCR_epoch = logging["val_OCR_score"].argmin() + 1
        shutil.copyfile(
            os.path.join(
                self.train_params["save_path"], "model_checkpoint_epoch{:03d}.tar".format(optimal_OCR_epoch)
            ),
            os.path.join(self.train_params["save_path"], "model_checkpoint_optimal_OCR.tar"),
        )
        shutil.copyfile(
            os.path.join(
                self.train_params["save_path"], "plot_epoch{:03d}.png".format(optimal_OCR_epoch)
            ),
            os.path.join(
                self.train_params["save_path"], "plot_epoch_optimal_OCR(_{:03d}).png".format(optimal_OCR_epoch)
            ),
        )

        # ----- copy optimal_OCR_score plots also for quick access -----
        shutil.copyfile(
            os.path.join(
                self.train_params["save_path"], "plot_epoch{:03d}.png".format(optimal_OCR_epoch)
            ),
            os.path.join(
                quick_access_path, "plot_epoch_optimal_OCR(_{:03d}).png".format(optimal_OCR_epoch)
            ),
        )


        # ----- optimal l2 error: -----
        optimal_l2_epoch = logging["val_loss"].argmin() + 1
        shutil.copyfile(
            os.path.join(
                self.train_params["save_path"], "model_checkpoint_epoch{:03d}.tar".format(optimal_l2_epoch)
            ),
            os.path.join(self.train_params["save_path"], "model_checkpoint_optimal_l2.tar"),
        )
        shutil.copyfile(
            os.path.join(
                self.train_params["save_path"], "plot_epoch{:03d}.png".format(optimal_l2_epoch)
            ),
            os.path.join(
                self.train_params["save_path"], "plot_epoch_optimal_l2.png(_{:03d}).png".format(optimal_l2_epoch)
            ),
        )

        # ----- copy optimal_l2_error plots also for quick access -----

        shutil.copyfile(
            os.path.join(
                self.train_params["save_path"], "plot_epoch{:03d}.png".format(optimal_l2_epoch)
            ),
            os.path.join(
                quick_access_path, "plot_epoch_optimal_l2.png(_{:03d}).png".format(optimal_l2_epoch)
            ),
        )
        pass


    def __call__(self, task_id):

        # ----- global configuration -----
        mpl.use("agg")
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)

        # ----- task ID mapping to the step to be trained in the specified train_phase.  -----
        # ----- assumption: task_id's run through the steps specified in steps_to_train exactly train_phases times. ----
        step, train_phase = self._current_step_and_train_phase(task_id)

        # ------ save hyperparameters -------
        path = self._change_save_path(step, train_phase + 1)
        self._save_hyperparameters(step, train_phase + 1, path)  # self.train_data["save_path"] changed inside

        # ----- set init weights, if initialisation is to be performed -----
        if self.init_path:
            self.network_params['init_kernel'] = self.init_path.format(step)

        # ----- load checkpoint if training is to be resumed -----
        if self.checkpoint_name:
            checkpoint_path = os.path.join(self.train_params["save_path"], self.checkpoint_name)
            self.train_params['checkpoint_path'] = checkpoint_path

        # ------ construct network ------
        network = self.network(**self.network_params).to(device)

        # ------ create data generators ------
        train_data = self.dataset(subset="train", step=step, **self.train_data_params)
        val_data = self.dataset(subset="val", step=step, **self.val_data_params)

        # ------ identify exact training parameters for this run -----
        train_params_cur = {}
        for key, value in self.train_params.items():
            train_params_cur[key] = (
                value[train_phase] if isinstance(value, (tuple, list)) else value
            )

        if self.finetune_path is not None:
            train_params_cur["finetune"] = True
            train_params_cur["checkpoint_path"] = self.finetune_path.format(step)

        if self.add_bg_for_val is not None:
            train_params_cur["add_bg_for_val"] = self.add_bg_for_val(step)

        # ----- print info about this training run -----
        print("Step : {}, Training Phase : {}".format(step, train_phase))
        for key, value in train_params_cur.items():
            print(key + ": " + str(value))

        # ------ train -----
        logging = network.train_on(train_data, val_data, **train_params_cur)

        # ----- pick best weights and save them separately----
        self._save_best_epoch(step, logging)
        pass


class PSF_train(train):
    def __init__(self,
                 network,
                 dataset,
                 target_folder,
                 network_params,
                 train_params,
                 train_data_params=None,
                 val_data_params=None,
                 loss_func=mse_rel_loss,
                 steps_to_train=None,
                 train_phases=1,
                 checkpoint_name=None,
                 init_path=None,
                 more_paths=None
                 ):
        super().__init__(network,
                 dataset,
                 target_folder,
                 network_params,
                 train_params,
                 train_data_params,
                 val_data_params,
                 loss_func,
                 steps_to_train,
                 train_phases,
                 checkpoint_name,
                 init_path,
                         )
        self.more_paths = more_paths


    def _change_save_path(self, step, train_phase, **kwargs):
        _ = super()._change_save_path(step, train_phase, **kwargs)
        self.train_params["save_path"] = os.path.join(
            self.train_params["save_path"],
            "kernel_size_" + str(self.network_params["kernel_size"]).zfill(3))
        if self.more_paths:
            self.train_params["save_path"] = os.path.join(
                self.train_params["save_path"],
                self.more_paths)
        return self.train_params["save_path"]

    def _save_best_epoch(self, step, logging):

        # ---- save last epoch for quick access and get quick access destination_path -----
        quick_access_path = self._copy_plots_for_quick_access()


        # ----- optimal l2 error: -----
        optimal_l2_epoch = logging["val_loss"].argmin() + 1
        shutil.copyfile(
            os.path.join(
                self.train_params["save_path"], "model_checkpoint_epoch{:03d}.tar".format(optimal_l2_epoch)
            ),
            os.path.join(self.train_params["save_path"], "model_checkpoint_optimal_l2.tar"),
        )
        shutil.copyfile(
            os.path.join(
                self.train_params["save_path"], "plot_epoch{:03d}.png".format(optimal_l2_epoch)
            ),
            os.path.join(
                self.train_params["save_path"], "plot_epoch_optimal_l2.png(_{}).png".format(optimal_l2_epoch)
            ),
        )
        shutil.copyfile(
            os.path.join(
                self.train_params["save_path"], "plot_epoch{:03d}_kernel.png".format(optimal_l2_epoch)
            ),
            os.path.join(
                self.train_params["save_path"], "plot_epoch_optimal_l2_kernel.png(_{}).png".format(optimal_l2_epoch)
            ),
        )

        # ----- copy optimal_ls_error plots also for quick access -----

        shutil.copyfile(
            os.path.join(
                self.train_params["save_path"], "plot_epoch{:03d}.png".format(optimal_l2_epoch)
            ),
            os.path.join(
                quick_access_path, "plot_epoch_optimal_l2.png(_{:03d}).png".format(optimal_l2_epoch)
            ),
        )
        shutil.copyfile(
            os.path.join(
                self.train_params["save_path"], "plot_epoch{:03d.}_kernel.png".format(optimal_l2_epoch)
            ),
            os.path.join(
                quick_access_path, "plot_epoch_optimal_l2_kernel.png(_{:03d}).png".format(optimal_l2_epoch)
            ),
        )

    def _copy_plots_for_quick_access(self, save_logging=True):
        """
        copies the last epoch plots and hyperparameters to quick access folder and returns path to target location
        :return:
        """
        destination_path = super()._copy_plots_for_quick_access(save_logging)

        last_epoch = self.train_params["num_epochs"]

        shutil.copyfile(
            os.path.join(
                self.train_params["save_path"], "plot_epoch{:03d}_kernel.png".format(last_epoch)
            ),
            os.path.join(
                destination_path, "plot_epoch{:03d}_kernel.png".format(last_epoch)
            ),
        )
        return destination_path

    def __call__(self, task_id):

        # ----- global configuration -----
        mpl.use("agg")
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)

        # ----- task ID mapping to the step to be trained in the specified train_phase.  -----
        # ----- assumption: task_id's run through the steps specified in steps_to_train exactly train_phases times. ----
        step, train_phase = self._current_step_and_train_phase(task_id)

        # ------ save hyperparameters -------
        path = self._change_save_path(step, train_phase + 1)
        self._save_hyperparameters(step, train_phase + 1, path)  # self.train_data["save_path"] changed inside

        # ----- set init weights, if initialisation is to be performed -----
        if self.init_path:
            self.network_params['init_kernel'] = self._init_weights(step)

        # ----- load checkpoint if training is to be resumed -----
        if self.checkpoint_name:
            checkpoint_path = os.path.join(self.train_params["save_path"], self.checkpoint_name)
            self.train_params['checkpoint_path'] = checkpoint_path

        # ------ construct network ------
        network = self.network(**self.network_params).to(device)

        # ------ create data generators ------
        train_data = self.dataset(subset="train", step=step, **self.train_data_params)
        val_data = self.dataset(subset="val", step=step, **self.val_data_params)

        # ------ identify exact training parameters for this run -----
        train_params_cur = {}
        for key, value in self.train_params.items():
            train_params_cur[key] = (
                value[train_phase] if isinstance(value, (tuple, list)) else value
            )

        # ----- print info about this training run -----
        print("Step : {}, Training Phase : {}".format(step, train_phase))
        for key, value in train_params_cur.items():
            print(key + ": " + str(value))

        # ------ train -----
        logging = network.train_on(train_data, val_data, **train_params_cur)

        # ----- pick best weights and save them separately----
        self._save_best_epoch(step, logging)
        pass