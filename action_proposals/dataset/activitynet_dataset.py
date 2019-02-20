"""
Activitynet proposals.
"""
import os
from typing import List, Callable, Tuple
import json
import pandas as pd
import torch
import numpy as np
from action_proposals.dataset import TemporalActionProposalDataset, VideoRecord, AnnotationRecord, VideoRecordHandler
import math


class ActivityNetDataset(TemporalActionProposalDataset):

    def __init__(self, json_path: str, video_info_new_csv_path, subset: str, video_record_handler: VideoRecordHandler):
        """
        Load activitynet dataset.

        :param json_path: Activitynet annotation path in json format.
        :param video_info_new_csv_path: Activitynet annotation path in csv format. It filterd some empty video.
        :param
        """
        self._json_path = json_path
        self._video_info_new_csv_path = video_info_new_csv_path
        self._subset = subset
        super().__init__(video_record_handler=video_record_handler)

    def _load_video_records(self) -> List[VideoRecord]:
        """
        Overwrite this function to load different dataset.

        :return: The list of video record.
        """
        df = pd.read_csv(self._video_info_new_csv_path)
        with open(self._json_path) as f:
            video_json: dict = json.load(f)

        video_records = []
        idx = 0
        for i in range(len(df)):
            if df.subset.values[i] == self._subset:
                video_name = df.video.values[i]
                video_info = video_json[video_name]
                assert len(video_info['annotations']) > 0
                proposals = [
                    AnnotationRecord(*proposal['segment'], proposal['label']) for proposal in video_info['annotations']
                ]
                video_record = VideoRecord(video_name, '',
                                           video_info['duration_second'] * video_info['feature_frame'] / video_info[
                                               'duration_frame'],
                                           video_info['feature_frame'] / video_info['duration_second'],
                                           proposals,
                                           'https://www.youtube.com/watch?v=' + video_name[2:],
                                           video_info['duration_frame'],
                                           idx
                                           )
                idx += 1
                video_records.append(video_record)

        return video_records

    @classmethod
    def get_ltw_feature_dataset(cls, csv_path: str, json_path: str, video_info_new_csv_path: str, class_name_path: str,
                                subset: str):
        """Get Tianwei Lin's feature dataset.

        :param csv_path: csv feature root path
        :param json_path: annotation json file
        :param class_name_path: class name file
        :param subset: training, validation and testing
        :param return_video_info: whether or not return video info
        :return:
        """
        video_record_handler = BSNVideoRecordHandler(csv_path, class_name_path)
        return cls(json_path=json_path, video_info_new_csv_path=video_info_new_csv_path,
                   video_record_handler=video_record_handler, subset=subset)


class BSNVideoRecordHandler(VideoRecordHandler):
    """
    Tianwei Lin's feature loader.
    """

    def __init__(self, csv_path, class_name_path):
        self._csv_path = csv_path
        self._class_name_path = class_name_path
        with open(class_name_path) as f:
            self._cls_list = f.read().split('\n')[1:]
        self._cls_to_idx = {i: cls for i, cls in enumerate(self._cls_list)}
        self._sequence_length = 100
        self._feature_length = 400
        self._video_dict = {}

    def __call__(self, video_record: VideoRecord) -> Tuple[torch.Tensor, torch.Tensor, VideoRecord]:
        """
        Given a video_record, return features and proposals.

        :param video_record: Video record.
        :returns: features [100, 400], proposals [3, 100], start, actioness, end.
        """

        # 第一次进入会把特征存下来，之后的遍历直接取即可。既可以加速运行，又不妨碍调试。
        if video_record.video_name in self._video_dict.keys():
            return self._video_dict[video_record.video_name]

        else:
            csv_file = os.path.join(self._csv_path, video_record.video_name + '.csv')
            df = pd.read_csv(csv_file)
            feature: torch.Tensor = torch.tensor(df.values, dtype=torch.float).transpose(0, 1)

            # 0,1,2 represent for start, actioness, end.
            proposals: torch.Tensor = torch.zeros(3, self._sequence_length, dtype=torch.float)
            assert len(video_record.annotations) > 0
            for ann in video_record.annotations:
                # 1. get the feature idx of each proposal
                start_feature_idx = ann.start_time / video_record.duration * self._sequence_length
                end_feature_idx = ann.end_time / video_record.duration * self._sequence_length

                start_feature_idx = np.clip(start_feature_idx, 0, self._sequence_length - 1)
                end_feature_idx = np.clip(end_feature_idx, 0, self._sequence_length - 1)

                # handle proposal
                if end_feature_idx >= start_feature_idx:
                    # 1. handle actionness
                    proposals[1, int(start_feature_idx + 1):int(end_feature_idx)] = 1
                    proposals[1, int(start_feature_idx)] = max(proposals[1, int(start_feature_idx)],
                                                               int(start_feature_idx) + 1 - start_feature_idx)
                    proposals[1, int(end_feature_idx)] = max(proposals[1, int(end_feature_idx)],
                                                             end_feature_idx - int(end_feature_idx))

                    # 2. handle boundary
                    boundary_range = np.maximum((end_feature_idx - start_feature_idx) / 10, 1) / 2

                    start_left_boundary = np.clip(start_feature_idx - boundary_range, 0, self._sequence_length - 1)
                    start_right_boundary = np.clip(start_feature_idx + boundary_range, 0, self._sequence_length - 1)
                    proposals[0, int(start_left_boundary + 1):int(start_right_boundary)] = 1
                    proposals[0, int(start_left_boundary)] = max(proposals[0, int(start_left_boundary)],
                                                                 int(start_left_boundary) + 1 - start_left_boundary)
                    proposals[0, int(start_right_boundary)] = max(proposals[0, int(start_right_boundary)],
                                                                  start_right_boundary - int(start_right_boundary))
                    # minus small number to make sure the int can work as floor.
                    end_left_boundary = np.clip(end_feature_idx - boundary_range, 0, self._sequence_length - 1e-7)
                    end_right_boundary = np.clip(end_feature_idx + boundary_range, 0, self._sequence_length - 1e-7)
                    proposals[2, int(end_left_boundary + 1):int(end_right_boundary)] = 1
                    proposals[2, int(end_left_boundary)] = max(proposals[2, int(end_left_boundary)],
                                                               int(end_left_boundary) + 1 - end_left_boundary)
                    proposals[2, int(end_right_boundary)] = max(proposals[2, int(end_right_boundary)],
                                                                end_right_boundary - int(end_right_boundary))
            # assert np.all(((proposals >= 0.5).sum(1) != 0).numpy())
            self._video_dict[video_record.video_name] = (feature, proposals)

        return feature, proposals, video_record

