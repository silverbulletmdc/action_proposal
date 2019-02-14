from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
from action_proposals.trainer import Trainer
from action_proposals.dataset.activitynet_dataset import ActivityNetDataset

from action_proposals.models.bsn import Tem, TemLoss
from action_proposals.utils import Statistic, cover_args_by_yml


class TemTrainer(Trainer):

    def _get_optimizer(self) -> Optimizer:
        optimizer = Adam(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        return optimizer

    def __init__(self):
        super(TemTrainer, self).__init__()
        self.model = Tem(self.cfg.input_features)
        self.loss = TemLoss()
        self.model = self.model.cuda()

    def _add_user_config(self):
        self._parser.add_argument("--input_features", type=int, default=400)
        self._parser.add_argument("--csv_path", type=str, default="")
        self._parser.add_argument("--json_path", type=str, default="")
        self._parser.add_argument("--class_name_path", type=str, default="")
        self._parser.add_argument("--video_info_new_csv_path", type=str, default="")
        self._parser.add_argument("--learning_rate", type=float, default=1e-3)
        self._parser.add_argument("--batch_size", type=int, default=16)
        self._parser.add_argument("--weight_decay", type=float, default=1e-3)
        self._parser.add_argument("--num_workers", type=int, default=16)


    def _get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        self.train_dataset = ActivityNetDataset.get_ltw_feature_dataset(self.cfg.csv_path, self.cfg.json_path,
                                                                        self.cfg.video_info_new_csv_path,
                                                                        self.cfg.class_name_path, 'training')

        self.val_dataset = ActivityNetDataset.get_ltw_feature_dataset(self.cfg.csv_path, self.cfg.json_path,
                                                                      self.cfg.video_info_new_csv_path,
                                                                      self.cfg.class_name_path, 'validation')

        train_data_loader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True,
                                       num_workers=self.cfg.num_workers)
        val_data_loader = DataLoader(self.val_dataset, batch_size=self.cfg.batch_size, shuffle=False,
                                     num_workers=self.cfg.num_workers)

        return train_data_loader, val_data_loader

    def _train_one_epoch(self, data_loaders: Tuple[DataLoader, DataLoader], epoch: int, optimizer: Optimizer):
        statistic = Statistic()
        train_dataloader, val_dataloader = data_loaders

        # train model
        self.model.train()
        for idx, (batch_feature, batch_proposals) in enumerate(train_dataloader):
            batch_feature = batch_feature.cuda()
            batch_proposals = batch_proposals.cuda()
            self.optimizer.zero_grad()
            batch_pred = self.model(batch_feature)
            loss_start, loss_action, loss_end = self.loss(batch_pred, batch_proposals)
            loss = (loss_start + 2 * loss_action + loss_end).mean()
            statistic.update('train_loss', loss.item())
            loss.backward()
            optimizer.step(None)

            # if idx % 100 == 0:
            #     print("epoch {}, iter {}: loss {}".format(epoch, idx, loss))

        # validate model
        self.model.eval()
        for idx, (batch_feature, batch_proposals) in enumerate(val_dataloader):
            batch_feature = batch_feature.cuda()
            batch_proposals = batch_proposals.cuda()

            batch_pred = self.model(batch_feature)
            loss_start, loss_action, loss_end = self.loss(batch_pred, batch_proposals)
            loss: torch.Tensor = (loss_start + 2 * loss_action + loss_end).mean()
            statistic.update('val_loss', loss.item())

        print("epoch {}: {}".format(epoch, statistic.format()))
        self.save_state(epoch, self.cfg.save_root)


if __name__ == '__main__':
    bsn_trainer = TemTrainer()
    bsn_trainer.train()
