import argparse
import json
import os
import time
from typing import Tuple

import pandas as pd
import multiprocessing as mp

from easydict import EasyDict

from action_proposals.dataset import VideoRecordHandler, VideoRecord, ActivityNetDataset
from action_proposals.utils import soft_nms, load_yml, log, mkdir_p

import numpy as np

log.log_level = log.DEBUG


class PostProcessingHandler(VideoRecordHandler):

    def __init__(self, pem_csv_dir):
        super(PostProcessingHandler, self).__init__()
        self._pem_csv_dir = pem_csv_dir

    def __call__(self, video_record: VideoRecord) -> Tuple[np.ndarray, VideoRecord]:
        """

        :param video_record:
        :return: proposals: [N, 3] tstart, tend, score
                 video_record: video record.
        """
        csv_path = os.path.join(self._pem_csv_dir, "{}.csv".format(video_record.video_name))
        df = pd.read_csv(csv_path)
        props = df.values
        score: np.ndarray = props[:, 2] * props[:, 3] * props[:, 4]
        props = np.concatenate([props[:, :2], score.reshape([-1, 1])], axis=1)
        return props, video_record


def get_post_processing_dataset(pem_csv_dir: str, json_path: str, video_info_new_csv_path: str, subset: str):
    video_record_handler = PostProcessingHandler(pem_csv_dir)
    return ActivityNetDataset(json_path=json_path, video_info_new_csv_path=video_info_new_csv_path,
                              video_record_handler=video_record_handler, subset=subset)


def sub_proc(queue: mp.Queue, mp_dict, dataset: ActivityNetDataset, cfg: EasyDict):
    while True:
        try:
            idx = queue.get(block=False)
        except:
            break
        if idx % 100 == 0:
            log.log_info("Handled {}/{} videos.".format(idx, len(dataset)))
        props, video_record = dataset[idx]
        props = soft_nms(props)
        # new_props = new_props[np.argsort(new_props[:, 2])][::-1]
        # log.log_debug(len(new_props))

        props_list = []
        for i in range(min(100, props.shape[0])):
            props_list.append(
                {
                    "score": props[i, 2],
                    "segment": list(props[i, :2] * video_record.duration)
                }
            )

        mp_dict[video_record.video_name[2:]] = props_list
        # write_queue.put((video_record, new_props))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yml_cfg_file", default="./cfgs/bsn.yml")
    args = parser.parse_args()
    cfg = load_yml(args.yml_cfg_file)
    anet = cfg.anet
    val_dataset = get_post_processing_dataset(cfg.pem.pem_csv_dir, anet.json_path, anet.video_info_new_csv_path, 'validation')

    queue = mp.Queue()
    mp_dict = mp.Manager().dict()

    # for i in range(100):
    for i in range(len(val_dataset)):
        queue.put(i)

    procs = []
    for i in range(cfg.post_processing.pp_workers):
        proc = mp.Process(target=sub_proc, args=(queue, mp_dict, val_dataset, cfg))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    output_dict = {"version": "VERSION 1.3", "results": dict(mp_dict), "external_data": {}}
    log.log_info(len(mp_dict.keys()))
    mkdir_p(os.path.split(cfg.eval.results_json)[0])
    with open(cfg.eval.results_json, "w") as f:
        json.dump(output_dict, f)
    log.log_info("Dump results to {}.".format(cfg.eval.results_json))


if __name__ == '__main__':
    main()