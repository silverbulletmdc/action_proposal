"""
Activitynet proposals.
"""
import os
from typing import List, Callable
import json
import pandas as pd
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
                                       video_info['duration_second'],
                                       video_info['feature_frame'] / video_info['duration_second'],
                                       proposals
                                       )
            video_records.append(video_record)

        return video_records

    @classmethod
    def get_ltw_feature_dataset(cls, csv_path: str, json_path: str):
        """Get Tianwei Lin's feature dataset.

        :param csv_path:
        :param json_path:
        :return:
        """
        video_record_handler = LtwVideoRecordHandler(csv_path)
        return cls(json_path=json_path, video_record_handler=video_record_handler)


class LtwVideoRecordHandler(VideoRecordHandler):
    """
    Tianwei Lin's feature loader.
    """
    def __init__(self, csv_path):
        self._csv_path = csv_path

    def __call__(self, video_record: VideoRecord):
        csv_file = os.path.join(self._csv_path, video_record.video_name + '.csv')
        df = pd.read_csv(csv_file)
        proposals = []
        for ann in video_record.annotations:
            proposals.append([ann.start_time, ann.end_time])
        return df, proposals

