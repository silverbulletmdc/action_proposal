from action_proposals.models.bsn import Tem
from action_proposals.dataset.activitynet_dataset import ActivityNetDataset
from action_proposals.utils import cover_args_by_yml, mkdir_p
import argparse
import torch
from torch.utils.data import DataLoader
import os
import pickle
import numpy as np


def main():
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--yml_cfg_file", type=str)
    cfg = parser.parse_args()
    cover_args_by_yml(cfg, cfg.yml_cfg_file)

    # 加载模型
    tem_model = Tem(cfg.input_features)
    model_info_path = os.path.join(cfg.save_root, 'model_model_info.txt')
    with open(model_info_path) as f:
        model_path = f.readline().strip()
        epoch = int(f.readline().strip())
    state_dicts = torch.load(model_path)
    tem_model.load_state_dict(state_dicts['model_state'])

    # 构建数据集
    val_dataset = ActivityNetDataset.get_ltw_feature_dataset(cfg.csv_path, cfg.json_path, cfg.video_info_new_csv_path,
                                                             cfg.class_name_path, 'validation')
    train_dataset = ActivityNetDataset.get_ltw_feature_dataset(cfg.csv_path, cfg.json_path, cfg.video_info_new_csv_path,
                                                               cfg.class_name_path, 'training')
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=16)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=16)
    results = []
    tem_model.cuda()
    tem_model.eval()

    for i, (features, _) in enumerate(val_dataloader):
        features = features.cuda()
        result = tem_model(features)
        results.append(result.cpu().detach().numpy())
    val_results = np.concatenate(results)

    results = []
    for i, (features, _) in enumerate(train_dataloader):
        features = features.cuda()
        result = tem_model(features)
        results.append(result.cpu().detach().numpy())

    train_results = np.concatenate(results)

    mkdir_p(os.path.split(cfg.tem_results_file)[0])

    with open(cfg.tem_results_file, 'wb') as f:
        results = {
            "training": train_results,
            "validation": val_results
        }
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
