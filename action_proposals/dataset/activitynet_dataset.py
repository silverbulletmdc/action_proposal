"""
Activitynet proposals.
"""
import os
from typing import List, Callable
import json
import pandas as pd
import torch
import numpy as np
from action_proposals.dataset import TemporalActionProposalDataset, VideoRecord, AnnotationRecord, VideoRecordHandler


class ActivityNetDataset(TemporalActionProposalDataset):
    def __init__(self, json_path: str = '', video_record_handler: VideoRecordHandler = None):
        """
        Load activitynet dataset.

        :param json_path: Activitynet annotation path in json format.
        :param
        """
        self._json_path = json_path
        super().__init__(video_record_handler=video_record_handler)

    def _load_video_records(self) -> List[VideoRecord]:
        """
        Overwrite this function to load different dataset.

        :return: The list of video record.
        """
        with open(self._json_path) as f:
            video_json: dict = json.load(f)

        video_records = []
        for video_name, video_info in video_json.items():
            proposals = [
                AnnotationRecord(*proposal['segment'], proposal['label']) for proposal in video_info['annotations']
            ]
            video_record = VideoRecord(video_name, '',
                                       video_info['duration_second'] * video_info['feature_frame'] / video_info['duration_frame'],
                                       video_info['feature_frame'] / video_info['duration_second'],
                                       proposals,
                                       'https://www.youtube.com/watch?v=' + video_name[2:],
                                       video_info['duration_frame']
                                       )
            video_records.append(video_record)

        return video_records

    @classmethod
    def get_ltw_feature_dataset(cls, csv_path: str, json_path: str, class_name_path: str):
        """Get Tianwei Lin's feature dataset.

        :param csv_path:
        :param json_path:
        :return:
        """
        video_record_handler = BSNVideoRecordHandler(csv_path, class_name_path)
        return cls(json_path=json_path, video_record_handler=video_record_handler)


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

    def __call__(self, video_record: VideoRecord):
        csv_file = os.path.join(self._csv_path, video_record.video_name + '.csv')
        df = pd.read_csv(csv_file)
        feature: torch.Tensor = torch.tensor(df.values, dtype=torch.float)

        # 0,1,2 represent for start, actioness, end.
        proposals: torch.Tensor = torch.zeros(3, self._sequence_length, dtype=torch.float)
        for ann in video_record.annotations:
            # 1. get the feature idx of each proposal
            start_feature_idx = ann.start_time / video_record.duration * self._sequence_length
            end_feature_idx = ann.end_time / video_record.duration * self._sequence_length

            start_feature_idx = np.clip(start_feature_idx, 0, self._sequence_length-1)
            end_feature_idx = np.clip(end_feature_idx, 0, self._sequence_length-1)

            # handle proposal
            if end_feature_idx > start_feature_idx:
                # 1. handle actionness
                proposals[1, int(start_feature_idx+1):int(end_feature_idx)] = 1
                proposals[1, int(start_feature_idx)] = max(proposals[1, int(start_feature_idx)],
                                                           int(start_feature_idx) + 1 - start_feature_idx)
                proposals[1, int(end_feature_idx)] = max(proposals[1, int(end_feature_idx)],
                                                         end_feature_idx - int(end_feature_idx))

                # 2. handle boundary
                boundary_range = (end_feature_idx - start_feature_idx) / 20

                start_left_boundary = np.clip(start_feature_idx - boundary_range, 0, self._sequence_length-1)
                start_right_boundary = np.clip(start_feature_idx + boundary_range, 0, self._sequence_length-1)
                proposals[0, int(start_left_boundary+1):int(start_right_boundary)] = 1
                proposals[0, int(start_left_boundary)] = max(proposals[0, int(start_left_boundary)],
                                                               int(start_left_boundary) + 1 - start_left_boundary)
                proposals[0, int(start_right_boundary)] = max(proposals[0, int(start_right_boundary)],
                                                              start_right_boundary - int(start_right_boundary))

                end_left_boundary = np.clip(end_feature_idx - boundary_range, 0, self._sequence_length - 1)
                end_right_boundary = np.clip(end_feature_idx + boundary_range, 0, self._sequence_length - 1)
                proposals[2, int(end_left_boundary+1):int(end_right_boundary)] = 1
                proposals[2, int(end_left_boundary)] = max(proposals[2, int(end_left_boundary)],
                                                           int(end_left_boundary) + 1 - end_left_boundary)
                proposals[2, int(end_right_boundary)] = max(proposals[2, int(end_right_boundary)],
                                                            end_right_boundary - int(end_right_boundary))

        return feature, proposals
