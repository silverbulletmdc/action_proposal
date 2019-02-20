import argparse
import os
import time
from typing import Tuple, Iterable

import pandas as pd
import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader

from action_proposals.trainer import Trainer
from action_proposals.utils import load_yml, Statistic, log
from action_proposals.dataset import ActivityNetDataset, VideoRecord, VideoRecordHandler
from action_proposals.models.bsn import Pem, PemLoss


class PemDatasetHandler(VideoRecordHandler):
    def __init__(self, proposal_dir, feature_dir, subset="training"):
        self._proposal_dir = proposal_dir
        self._feature_dir = feature_dir
        self.feature_dict = {}
        self._subset = subset

    def __call__(self, video_record: VideoRecord):
        if video_record.video_name in self.feature_dict:
            return self.feature_dict[video_record.video_name]

        else:
            proposals_path = os.path.join(self._proposal_dir, '{}.csv'.format(video_record.video_name))
            feature_path = os.path.join(self._feature_dir, '{}.npy'.format(video_record.video_name))
            prop_df = pd.read_csv(proposals_path)
            feature = np.load(feature_path)

            if self._subset == "training":
                return torch.tensor(prop_df.values[:500], dtype=torch.float32), \
                       torch.tensor(feature[:500],dtype=torch.float32), video_record
            elif self._subset == "validation":
                return torch.tensor(prop_df.values[:1000], dtype=torch.float32), \
                       torch.tensor(feature[:1000], dtype=torch.float32), video_record


def pem_collate_cb(data):
    props = [item[0] for item in data]
    features = [item[1] for item in data]
    lengths = [item.shape[0] for item in props]

    props = torch.cat(props)
    features = torch.cat(features)
    assert(props.shape[0] == features.shape[0])
    return props, features, lengths


def get_pem_dataset(proposal_csv_path: str, pgm_feature_path: str, json_path: str, video_info_new_csv_path: str,
                    subset: str):
    video_record_handler = PemDatasetHandler(proposal_csv_path, pgm_feature_path, subset)
    return ActivityNetDataset(json_path=json_path, video_info_new_csv_path=video_info_new_csv_path,
                              video_record_handler=video_record_handler, subset=subset)


class PemTrainer(Trainer):

    def __init__(self, cfg, trainer_cfg):
        self.cfg = cfg
        super(PemTrainer, self).__init__(trainer_cfg)
        self.model.cuda()
        self.loss.cuda()

    def _build_models(self) -> Iterable[Module]:
        self.model = Pem()
        self.loss = PemLoss()
        return [self.model, self.loss]

    def _get_optimizers(self) -> Iterable[Optimizer]:
        self.optimizer = Adam(self.model.parameters(), lr=self.cfg.pem.learning_rate,
                              weight_decay=self.cfg.pem.weight_decay)
        return [self.optimizer]

    def _add_user_config(self):
        pass

    def _get_dataloaders(self) -> Iterable[DataLoader]:
        cfg = self.cfg
        anet = cfg.anet
        pgm = cfg.pgm
        self.train_dataset = train_dataset = get_pem_dataset(pgm.proposal_csv_path, pgm.pgm_feature_path, anet.json_path,
                                        anet.video_info_new_csv_path, "training")

        self.val_dataset = val_dataset = get_pem_dataset(pgm.proposal_csv_path, pgm.pgm_feature_path, anet.json_path,
                                      anet.video_info_new_csv_path, "validation")

        self.train_loader = DataLoader(train_dataset, shuffle=True, num_workers=cfg.pem.num_workers,
                                       batch_size=cfg.pem.batch_size, collate_fn=pem_collate_cb)
        self.val_loader = DataLoader(val_dataset, shuffle=False, num_workers=cfg.pem.num_workers,
                                     batch_size=cfg.pem.batch_size, collate_fn=pem_collate_cb)
        return [self.train_loader, self.val_loader]

    def _train_one_epoch(self, epoch: int):
        stat = Statistic()
        self.model.train()

        if epoch == 10:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.cfg.pem.learning_rate / 10
            log.log_info("The learning rate is adjusted in the beginning of epoch 10 ")

        t = time.time()
        for idx, (props, features, _) in enumerate(self.train_loader):
            props = props.cuda()
            features = features.cuda()
            self.model.zero_grad()
            pred_scores = self.model(features)
            loss = self.loss(pred_scores, props[:, 5])

            # log.log_warn("Loss is None!")
            stat.update("train_loss", loss.item())
            loss.backward()
            self.optimizer.step(None)

        self.model.eval()
        for idx, (props, features, _) in enumerate(self.val_loader):
            props = props.cuda()
            features = features.cuda()
            pred_scores = self.model(features)
            loss = self.loss(pred_scores, props[:, 5])

            # log.log_warn("Loss is None!")
            stat.update("val_loss", loss.item())
        self.save_state(epoch, self.cfg.pem.save_root)
        log.log_info("[{:.2}s] Epoch {}: {}".format(time.time() - t, epoch, stat.format()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yml_cfg_file", default="./cfgs/bsn.yml")
    args = parser.parse_args()
    cfg = load_yml(args.yml_cfg_file)
    trainer = PemTrainer(cfg, cfg.pem)
    trainer.train()


if __name__ == '__main__':
    main()
