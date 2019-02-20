import os
import argparse
import time
from typing import Tuple, List, Iterable

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
from torch.nn import Module

from action_proposals.trainer import Trainer
from action_proposals.dataset.activitynet_dataset import ActivityNetDataset
from action_proposals.utils import log, mkdir_p
from action_proposals.models.bsn import Tem, TemLoss
from action_proposals.utils import Statistic, load_yml

log.log_level = log.INFO


def ltw_collate_fn(datas):
    features = [item[0] for item in datas]
    proposals = [item[1] for item in datas]
    video_records = [item[2] for item in datas]

    return torch.stack(features), torch.stack(proposals), video_records


class TemTrainer(Trainer):

    def __init__(self, cfg, trainer_cfg):
        self.cfg = cfg
        super(TemTrainer, self).__init__(trainer_cfg)
        self.model.cuda()
        self.loss.cuda()

    def _get_optimizers(self) -> Iterable[Optimizer]:
        self.optimizer = Adam(self.model.parameters(), lr=self.cfg.tem.learning_rate, weight_decay=self.cfg.tem.weight_decay)
        return [self.optimizer, ]

    def _build_models(self) -> Iterable[Module]:
        self.model = Tem(self.cfg.tem.input_features)
        self.loss = TemLoss()
        return [self.model, self.loss]

    def _get_dataloaders(self) -> Iterable[DataLoader]:
        if self.cfg.dataset == "activitynet":
            anet = self.cfg.anet

            self.train_dataset = ActivityNetDataset.get_ltw_feature_dataset(anet.csv_path, anet.json_path,
                                                                            anet.video_info_new_csv_path,
                                                                            anet.class_name_path, 'training')

            self.val_dataset = ActivityNetDataset.get_ltw_feature_dataset(anet.csv_path, anet.json_path,
                                                                          anet.video_info_new_csv_path,
                                                                          anet.class_name_path, 'validation')

            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.cfg.tem.batch_size, shuffle=True,
                                               num_workers=self.cfg.tem.num_workers, collate_fn=ltw_collate_fn)
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.cfg.tem.batch_size, shuffle=False,
                                             num_workers=self.cfg.tem.num_workers, collate_fn=ltw_collate_fn)

            return [self.train_dataloader, self.val_dataloader]

        else:
            log.log_error("Haven't support {} dataset.".format(self.cfg.dataset))
            exit(-1)

    def _train_one_epoch(self, epoch: int):
        t = time.time()
        statistic = Statistic()

        # train model
        self.model.train()

        if epoch == 10:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.cfg.tem.learning_rate / 10
            log.log_info("The learning rate is adjusted in the beginning of epoch {}".format(epoch))

        for idx, (batch_feature, batch_proposals, video_records) in enumerate(self.train_dataloader):
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
        for idx, (batch_feature, batch_proposals, video_records) in enumerate(self.val_dataloader):
            batch_feature = batch_feature.cuda()
            batch_proposals = batch_proposals.cuda()

            batch_pred = self.model(batch_feature)
            loss_start, loss_action, loss_end = self.loss(batch_pred, batch_proposals)
            loss: torch.Tensor = (loss_start + 2 * loss_action + loss_end).mean()
            statistic.update('val_loss', loss.item())

        log.log_info("[{:.2f}s]: epoch {}: {}".format(time.time() - t, epoch, statistic.format()))
        self.save_state(epoch, self.cfg.tem.save_root)

    def test(self):
        mkdir_p(self.cfg.tem.tem_csv_dir)
        self.load_state(self.cfg.tem.save_root)
        self.model.eval()
        for dataloader_id, dataloader in enumerate([self.train_dataloader, self.val_dataloader]):
            for idx, (batch_feature, batch_proposals, video_records) in enumerate(dataloader):
                features = batch_feature.cuda()
                pred_scores: torch.Tensor = self.model(features)
                pred_scores: np.ndarray = pred_scores.detach().cpu().numpy()

                for i, video_record in enumerate(video_records):
                    latent_df = pd.DataFrame(columns=["action", "start", "end", "xmin", "xmax"])
                    pred_score = pred_scores[i].T  # [L, 3]
                    latent_df["start"] = pred_score[:, 0]
                    latent_df["action"] = pred_score[:, 1]
                    latent_df["end"] = pred_score[:, 2]
                    latent_df["xmin"] = np.arange(0, 1, 1 / pred_score.shape[0])
                    latent_df["xmax"] = np.arange(1 / pred_score.shape[0], 1 + 1 / pred_score.shape[0],
                                                  1 / pred_score.shape[0])
                    latent_df.to_csv(os.path.join(self.cfg.tem.tem_csv_dir, "{}.csv".format(video_record.video_name)),
                                     index=False)
                log.log_info("[{}/2 dataset] Handled {}/{} videos.".format(dataloader_id + 1, idx * self.cfg.tem.batch_size,
                                                                           len(dataloader.dataset)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--yml_cfg_file", type=str, default="./cfgs/bsn.yml", help="The config file path.")
    parser.add_argument("--run_type", type=str, default="train", help="train or test.")
    args = parser.parse_args()
    cfg = load_yml(args.yml_cfg_file)
    bsn_trainer = TemTrainer(cfg, cfg.tem)
    if args.run_type == "train":
        log.log_info("Start to train TEM.")
        bsn_trainer.train()
    elif args.run_type == "test":
        log.log_info("Start to test TEM.")
        bsn_trainer.test()
    else:
        log.log_error("You can only use `train` or `test` as run_type. Found `{}`.".format(args.run_type))
