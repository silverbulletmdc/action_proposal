import os

import torch
from torch.utils.data import DataLoader
from action_proposals.utils import log, mkdir_p, load_yml
from pem_train import PemTrainer
import numpy as np
import pandas as pd
import argparse


def pem_collate_cb(data):
    props = [item[0] for item in data]
    features = [item[1] for item in data]
    video_records = [item[2] for item in data]
    lengths = [item.shape[0] for item in props]

    props = torch.cat(props)
    features = torch.cat(features)
    return props, features, lengths, video_records


class PemTester(PemTrainer):
    def __init__(self, cfg, trainer_cfg):
        super(PemTester, self).__init__(cfg, trainer_cfg)
        self.val_loader.collate_fn = pem_collate_cb

    def test(self):
        mkdir_p(self.cfg.pem.pem_csv_dir)
        self.load_state(self.cfg.pem.save_root)
        self.model.eval()

        for idx, (props, features, lengths, video_records) in enumerate(self.val_loader):
            features = features.cuda()
            pred_scores: torch.Tensor = self.model(features)
            pred_scores = pred_scores.detach().cpu().numpy()
            log.log_debug(lengths)
            for i, (length, video_record) in enumerate(zip(lengths, video_records)):
                latent_df = pd.DataFrame(columns=["xmin", "xmax", "xmin_score", "xmax_score", "iou_score"])
                latent_df["xmin"] = props[:length, 0]
                latent_df["xmax"] = props[:length, 1]
                latent_df["xmin_score"] = props[:length, 2]
                latent_df["xmax_score"] = props[:length, 3]
                props = props[length:, :]
                latent_df["iou_score"] = pred_scores[:length]
                pred_scores = pred_scores[length:]
                latent_df.to_csv(os.path.join(self.cfg.pem.pem_csv_dir, "{}.csv".format(video_record.video_name)), index=False)
            log.log_info("Handled {}/{} videos.".format(idx * self.cfg.pem.batch_size, len(self.val_dataset)))


if __name__ == '__main__':

    log.log_level = log.INFO

    parser = argparse.ArgumentParser()
    parser.add_argument("--yml_cfg_file", default="./cfgs/bsn.yml")
    args = parser.parse_args()
    cfg = load_yml(args.yml_cfg_file)

    pem_tester = PemTester(cfg, cfg.pem)
    pem_tester.test()
