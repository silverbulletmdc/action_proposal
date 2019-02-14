from typing import Tuple, List, Callable

from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from action_proposals.utils import cover_args_by_yml
import argparse
import torch
import os

from action_proposals.utils import mkdir_p


class Trainer:
    r"""
    Base class of trainer. You should implement required methods, then you can call the :code:`train()` method,
    and the training process will be auto conducted.
    """

    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._add_config()
        self.cfg = self._parser.parse_args()
        if self.cfg.yml_cfg_file:
            cover_args_by_yml(self.cfg, self.cfg.yml_cfg_file)
        self.epochs: int = self.cfg.epochs

        self.data_loaders = self._get_dataloaders()
        self.optimizer: Optimizer = None
        self.model: torch.nn.Module = None

    def build_model(self) -> torch.nn.Module:
        r"""Implement this method to set the model.

        :return:
        """
        raise NotImplementedError

    def train(self):
        self.optimizer = self._get_optimizer()
        current_epoch = 0
        if self.cfg.continue_train:
            current_epoch = self.load_state(self.cfg.save_root) + 1

        for epoch in range(current_epoch, self.cfg.epochs):
            self._train_one_epoch(self.data_loaders, epoch, self.optimizer)

    def save_state(self, epoch: int, root='/tmp/', prefix='model', *other_objects):
        r"""Save state. This method will save self.model and self.optimizer automatically. If you want to save other
        models, please pass it as optional arguments, and pass it using same order when you use self.load_state()

        :param epoch:
        :param root:
        :param prefix:
        :param other_objects: other object which implement `state_dict` method to save.
        :return:
        """

        mkdir_p(root)
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict()

        model_name = '{}_{}.pth'.format(prefix, epoch)
        model_path = os.path.join(root, model_name)

        save_dict = {'model_state': model_state, 'optimizer_state': optimizer_state}

        for i, object_ in enumerate(other_objects):
            save_dict[i] = object_.state_dict()

        torch.save(save_dict, model_path)
        model_info_path = os.path.join(root, '{}_model_info.txt'.format(prefix))
        with open(model_info_path, 'w') as f:
            f.write(model_path)
            f.write('\n')
            f.write(str(epoch))

    def load_state(self, root='/tmp/', prefix='model', *other_object_) -> int:
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

        self.model.load_state_dict(state_dicts["model_state"])
        self.optimizer.load_state_dict(state_dicts["optimizer_state"])
        for i, object_ in enumerate(other_object_):
            object_.load_state_dict(state_dicts[i])
        return epoch

    def _add_config(self):
        self._parser.add_argument("--epochs", type=int, default=4)
        self._parser.add_argument("--save_root", type=str, default="/tmp")
        self._parser.add_argument("--continue_train", type=bool, default=False)
        self._parser.add_argument("--yml_cfg_file", type=str, default=None)
        self._add_user_config()

    def _add_user_config(self):
        r"""
        Implement this method to add your custom configures. This method will be called when `self.__init__` was called.
        You should add config by

        :code:`self.parser.add_argument("--arg_name", type=int, default=4)`

        and using it with :code:`self.cfg.arg_name`

        :return: None
        """
        raise NotImplementedError

    def _get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        r"""Get dataloader. This method will be called after argument parsing in `self.__init__()` .
        Do the data augmentation and construct dataloader in this method.

        :return: Train dataloader and validation dataloader.
        """
        raise NotImplementedError

    def _train_one_epoch(self, data_loaders: Tuple[DataLoader, DataLoader], epoch: int, optimizer: Optimizer):
        r"""
        Train one epoch. This method will be called periodically in self.train().

        :param data_loaders: train data loader and validation data loader.
        :param epoch: epoch
        :return: None
        """
        raise NotImplementedError

    def _get_optimizer(self) -> Optimizer:
        r"""
        Construct your optimizer. This method will be called when `self.train` was called.

        :return: optimizer.
        """
        raise NotImplementedError
