import os
from typing import List
import json

from action_proposals.dataset import TemporalActionProposalDataset, VideoRecord, AnnotationRecord


class ActivityNetDataset(TemporalActionProposalDataset):
    def __init__(self, json_path, frames_path, modality):
        self._json_path = json_path
        self._frames_path = frames_path
        self._modality = modality
        super().__init__()

    def __getitem__(self, idx):
        video_record = self.video_records[idx]

    def _load_video_records(self) -> List[VideoRecord]:
        with open(self._json_path) as f:
            video_json: dict = json.load(f)

        video_records = []
        for video_name, video_info in video_json.items():
            proposals = [
                AnnotationRecord(*proposal['segment'], proposal['label']) for proposal in video_info['annotations']
            ]
            video_record = VideoRecord(video_name, os.path.join(self._frames_path, video_name),
                                       video_info['duration_second'],
                                       video_info['feature_frame'] / video_info['duration_second'],
                                       proposals
                                       )
            video_records.append(video_record)

        return video_records
