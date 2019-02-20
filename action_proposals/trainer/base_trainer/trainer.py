from typing import Tuple, List, Callable, Iterable

from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from action_proposals.utils import load_yml
from action_proposals.utils import log
import argparse
import torch
from torch.nn import Module
import os

from action_proposals.utils import mkdir_p


class Trainer:
    r"""
    Base class of trainer. You should implement required methods, then you can call the :code:`train()` method,
    and the training process will be auto conducted.

    You must provide epochs, continue_run, save_root in your cfg files.
    """

    def __init__(self, trainer_cfg):
        self._trainer_cfg = trainer_cfg
        self.epochs: int = self._trainer_cfg.epochs

        self._data_loaders = self._get_dataloaders()
        self._models: Iterable[Module] = self._build_models()
        self._optimizers: Iterable[Optimizer] = self._get_optimizers()

    def _build_models(self) -> Iterable[Module]:
        r"""Implement this method to set the models. Trainer will save and load the models.

        :return:
        """
        raise NotImplementedError


    def train(self):
        current_epoch = 0
        if self._trainer_cfg.continue_train:
            current_epoch = self.load_state(self._trainer_cfg.save_root) + 1

        for epoch in range(current_epoch, self._trainer_cfg.epochs):
            self._train_one_epoch(epoch)

    def save_state(self, epoch: int, root='/tmp/', prefix='model'):
        r"""Save state. This method will save self.model and self.optimizer automatically. If you want to save other
        models, please pass it as optional arguments, and pass it using same order when you use self.load_state()

        :param epoch:
        :param root:
        :param prefix:
        :param other_objects: other object which implement `state_dict` method to save.
        :return:
        """
        save_dict = {"models": {},
                     "optimizers": {}}
        mkdir_p(root)
        for i, model in enumerate(self._models):
            model_state = model.state_dict()
            save_dict['models'][i] = model_state

        for i, optimizer in enumerate(self._optimizers):
            optimizer_state = optimizer.state_dict()
            save_dict['optimizers'][i] = optimizer_state

        model_name = '{}_{}.pth'.format(prefix, epoch)
        model_path = os.path.join(root, model_name)

        torch.save(save_dict, model_path)
        model_info_path = os.path.join(root, '{}_model_info.txt'.format(prefix))
        with open(model_info_path, 'w') as f:
            f.write(model_path)
            f.write('\n')
            f.write(str(epoch))

    def load_state(self, root='/tmp/', prefix='model') -> int:
        """

        :param root:
        :param prefix:
        :param other_object_:
        :return: epoch
        """
        # Get the last epoch model.

        model_info_path = os.path.join(root, '{}_model_info.txt'.format(prefix))
        with open(model_info_path) as f:
            model_path = f.readline().strip()
            epoch = int(f.readline().strip())

        state_dicts = torch.load(model_path)

        for i, model in enumerate(self._models):
            model.load_state_dict(state_dicts["models"][i])
        for i, optimizer in enumerate(self._optimizers):
            optimizer.load_state_dict(state_dicts["optimizers"][i])

        log.log(log.WARN, "Successfully load model from {}".format(model_path))

        return epoch

    def _get_dataloaders(self) -> Iterable[DataLoader]:
        r"""Get dataloader. This method will be called after argument parsing in `self.__init__()` .
        Do the data augmentation and construct dataloader in this method.

        :return: Train dataloader and validation dataloader.
        """
        raise NotImplementedError

    def _train_one_epoch(self, epoch: int):
        r"""
        Train one epoch. This method will be called periodically in self.train().

        :param data_loaders: train data loader and validation data loader.
        :param epoch: epoch
        :return: None
        """
        raise NotImplementedError

    def _get_optimizers(self) -> Iterable[Optimizer]:
        r"""
        Construct your optimizer. This method will be called when `self.train` was called.

        :return: optimizer.
        """
        raise NotImplementedError
