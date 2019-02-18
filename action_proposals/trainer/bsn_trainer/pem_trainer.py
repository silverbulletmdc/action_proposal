import argparse
import os

import pandas as pd
import numpy as np

from action_proposals.trainer import Trainer
from action_proposals.utils import cover_args_by_yml
from action_proposals.dataset import ActivityNetDataset, VideoRecord, VideoRecordHandler


class PemDatasetHandler(VideoRecordHandler):
    def __init__(self, proposal_dir, feature_dir):
        self._proposal_dir = proposal_dir
        self._feature_dir = feature_dir
        self.feature_dict = {}

    def __call__(self, video_record: VideoRecord):
        if video_record.video_name in self.feature_dict:
            return self.feature_dict[video_record.video_name]

        else:
            proposals_path = os.path.join(self._proposal_dir, '{}.csv'.format(video_record.video_name))
            feature_path = os.path.join(self._feature_dir, '{}.npy'.format(video_record.video_name))
            prop_df = pd.read_csv(proposals_path)
            feature = np.load(feature_path)
            return prop_df, feature, video_record


def get_pem_dataset(proposal_csv_path: str, pgm_feature_path: str, json_path: str, video_info_new_csv_path: str,
                    subset: str):
    video_record_handler = PemDatasetHandler(proposal_csv_path, pgm_feature_path)
    return ActivityNetDataset(json_path=json_path, video_info_new_csv_path=video_info_new_csv_path,
                              video_record_handler=video_record_handler, subset=subset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yml_cfg_file", type=str, default="bsn_config.yml")
    cfg = parser.parse_args()
    cover_args_by_yml(cfg, cfg.yml_cfg_file)

    train_dataset = get_pem_dataset(cfg.proposal_csv_path, cfg.pgm_feature_path, cfg.json_path,
                                    cfg.video_info_new_csv_path, "training")

    val_dataset = get_pem_dataset(cfg.proposal_csv_path, cfg.pgm_feature_path, cfg.json_path,
                                    cfg.video_info_new_csv_path, "validation")
    prop_df: pd.DataFrame
    feature: np.ndarray
    video_record: VideoRecord
    for i, (prop_df, feature, video_record) in enumerate(train_dataset):
        print(i)
        print(video_record.video_name)
        print(prop_df.head(0))
        print(feature.shape)


if __name__ == '__main__':
    main()
