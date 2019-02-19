from action_proposals.models.bsn import Tem
from action_proposals.dataset.activitynet_dataset import ActivityNetDataset
from action_proposals.utils import load_yml, mkdir_p, log
import argparse
import torch
from torch.utils.data import DataLoader
import os
import pickle
import numpy as np

log.log_level = log.INFO


def main():
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--yml_cfg_file", type=str)
    args = parser.parse_args()
    cfg = load_yml(args.yml_cfg_file)

    # 加载模型
    tem_model = Tem(cfg.input_features)
    model_info_path = os.path.join(cfg.save_root, 'model_model_info.txt')
    with open(model_info_path) as f:
        model_path = f.readline().strip()
        epoch = int(f.readline().strip())
    state_dicts = torch.load(model_path)
    tem_model.load_state_dict(state_dicts['models'][0])

    # 构建数据集
    anet = cfg.anet_dataset
    val_dataset = ActivityNetDataset.get_ltw_feature_dataset(anet.csv_path, anet.json_path,
                                                             anet.video_info_new_csv_path,
                                                             anet.class_name_path, 'validation')
    train_dataset = ActivityNetDataset.get_ltw_feature_dataset(anet.csv_path, anet.json_path,
                                                               anet.video_info_new_csv_path,
                                                               anet.class_name_path, 'training')
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=16)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=16)
    tem_model.cuda()
    tem_model.eval()

    log.log(log.INFO, "Start processing train dataset.")
    results = []
    for i, (features, _) in enumerate(val_dataloader):
        features = features.cuda()
        result = tem_model(features)
        results.append(result.cpu().detach().numpy())
    val_results = np.concatenate(results)

    log.log(log.INFO, "Start processing validation dataset.")
    results = []
    for i, (features, _) in enumerate(train_dataloader):
        features = features.cuda()
        result = tem_model(features)
        results.append(result.cpu().detach().numpy())

    train_results = np.concatenate(results)

    mkdir_p(os.path.split(cfg.tem_results_file)[0])

    log.log(log.INFO, "Dumping to {}".format(cfg.tem_results_file))
    with open(cfg.tem_results_file, 'wb') as f:
        results = {
            "training": train_results,
            "validation": val_results
        }
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
