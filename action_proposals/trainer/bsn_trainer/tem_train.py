from action_proposals.utils import log
log.log_level = log.WARN

import time
from typing import Tuple, List, Iterable
import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
from action_proposals.trainer import Trainer
from action_proposals.dataset.activitynet_dataset import ActivityNetDataset

from action_proposals.models.bsn import Tem, TemLoss
from action_proposals.utils import Statistic, load_yml
from torch.nn import Module

class TemTrainer(Trainer):

    def __init__(self, yml_cfg_file):
        super(TemTrainer, self).__init__(yml_cfg_file)
        self.model.cuda()
        self.loss.cuda()

    def _get_optimizers(self) -> Iterable[Optimizer]:
        self.optimizer = Adam(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        return [self.optimizer, ]

    def _build_models(self) -> Iterable[Module]:
        self.model = Tem(self.cfg.input_features)
        self.loss = TemLoss()
        return [self.model, self.loss]

    def _get_dataloaders(self) -> Iterable[DataLoader]:
        anet = self.cfg.anet_dataset
        
        self.train_dataset = ActivityNetDataset.get_ltw_feature_dataset(anet.csv_path, anet.json_path,
                                                                        anet.video_info_new_csv_path,
                                                                        anet.class_name_path, 'training')

        self.val_dataset = ActivityNetDataset.get_ltw_feature_dataset(anet.csv_path, anet.json_path,
                                                                      anet.video_info_new_csv_path,
                                                                      anet.class_name_path, 'validation')

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True,
                                       num_workers=self.cfg.num_workers)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.cfg.batch_size, shuffle=False,
                                     num_workers=self.cfg.num_workers)

        return [self.train_dataloader, self.val_dataloader]

    def _train_one_epoch(self, epoch: int):
        t = time.time()
        statistic = Statistic()

        # train model
        self.model.train()

        if epoch == 10:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.cfg.learning_rate / 10

        for idx, (batch_feature, batch_proposals) in enumerate(self.train_dataloader):
            batch_feature = batch_feature.cuda()
            batch_proposals = batch_proposals.cuda()
            self.optimizer.zero_grad()
            batch_pred = self.model(batch_feature)
            loss_start, loss_action, loss_end = self.loss(batch_pred, batch_proposals)
            loss = (loss_start + 2 * loss_action + loss_end).mean()
            statistic.update('train_loss', loss.item())
            loss.backward()
            self.optimizer.step(None)

            # if idx % 100 == 0:
            #     print("epoch {}, iter {}: loss {}".format(epoch, idx, loss))

        # validate model
        self.model.eval()
        for idx, (batch_feature, batch_proposals) in enumerate(self.val_dataloader):
            batch_feature = batch_feature.cuda()
            batch_proposals = batch_proposals.cuda()

            batch_pred = self.model(batch_feature)
            loss_start, loss_action, loss_end = self.loss(batch_pred, batch_proposals)
            loss: torch.Tensor = (loss_start + 2 * loss_action + loss_end).mean()
            statistic.update('val_loss', loss.item())

        print("[{:.2f}s]: epoch {}: {}".format(time.time()-t, epoch, statistic.format()))
        self.save_state(epoch, self.cfg.save_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--yml_cfg_file", type=str, default="./cfgs/tem.yml", help="The config file path.")
    args = parser.parse_args()
    bsn_trainer = TemTrainer(args.yml_cfg_file)
    bsn_trainer.train()
