from action_proposals.models.bsn import Tem, TemLoss
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
import argparse


class Trainer:

    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._add_config()
        self.cfg = self._parser.parse_args()
        self.epochs: int = self.cfg.epochs
        self.batch_size: int = self.cfg.epochs

        self.data_loader = self._get_dataloader()
        self.optimizer:Optimizer = None
    def train(self):
        self.optimizer = self._get_optimizer()
        for epoch in range(self.epochs):
            self._train_one_epoch(self.data_loader, epoch, self.optimizer)

    def _add_config(self):
        self._parser.add_argument("--epochs", type=int, default=4)
        self._parser.add_argument("--batch_size", type=int, default=16)
        self._add_user_config()

    def _add_user_config(self):
        r"""
        Implement this method to add your custom configures. You should add config by

        :code:`self.parser.add_argument("--arg_name", type=int, default=4)`

        and using it with :code:`self.cfg.arg_name`

        :return: None
        """
        raise NotImplementedError

    def _get_dataloader(self) -> DataLoader:
        r"""Get dataloader. Do the data augmentation and construct dataloader in this method.

        :return: Dataset.
        """
        raise NotImplementedError

    def _train_one_epoch(self, data_loader: DataLoader, epoch: int, optimizer: Optimizer):
        r"""
        Train one epoch.

        :param data_loader: data loader
        :param epoch: epoch
        :return: None
        """
        raise NotImplementedError

    def _get_optimizer(self) -> Optimizer:
        raise NotImplementedError