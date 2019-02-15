"""
Base classes of all dataset of temporal action proposal.
"""

import os
from PIL import Image
import numpy as np
import torch, torchvision
from torch.utils.data import Dataset
from typing import List, Tuple, Iterable


class AnnotationRecord:
    """
    The data structure of an annotation. It contains the start time and end time of the proposal, and its' class label.
    """
    def __init__(self, start_time, end_time, class_):
        """

        :param start_time:
        :param end_time:
        :param class_:
        """
        self.start_time = start_time
        self.end_time = end_time
        self.class_ = class_


class VideoRecord:
    """
    The datastructure of a video.
    """

    def __init__(self, video_name: str = '', frame_path: str = '', duration: int = 0, fps: float = 0,
                 annotations: List[AnnotationRecord] = None, url: str = '', frames: int = 0, idx: int = 0):
        self.video_name = video_name
        self.frame_path = frame_path
        self.duration = duration
        self.frames = frames
        self.fps = fps
        self.annotations = annotations
        self.url = url
        self.idx = idx


class VideoRecordHandler:
    """
    Base class of the video record handler. You must implement the __call__ method, so that we can handle different
    type of features of a dataset.
    """

    def __call__(self, video_record: VideoRecord):
        raise NotImplementedError


class TemporalActionProposalDataset(Dataset):
    """
    Base class of a temporal action proposal dataset. You must implement
    """
    def __init__(self, video_record_handler: VideoRecordHandler):
        super(TemporalActionProposalDataset, self).__init__()
        self._video_record_handler = video_record_handler
        self._video_records = self._load_video_records()

    def __len__(self):
        return len(self._video_records)

    def __getitem__(self, idx):
        return self._video_record_handler(self._video_records[idx])

    def _load_video_records(self) -> List[VideoRecord]:
        """
        Overwrite this function to load different dataset.

        :return: The list of video record.
        """
        raise NotImplementedError

    def _load_frames(self, idx: int, ranges: Iterable[int], modality: str = "RGB") -> torch.Tensor:
        """A tool function to load some frames.

        :param idx: Video idx
        :param ranges: Which frames to be loaded.
        :param modality: "RGB" or "Flow" and so on.
        :return:
        """

        frame_path = self._video_records[idx].frame_path
        img_paths = [os.path.join(frame_path, "{}_{}".format(modality, x)) for x in ranges]
        images = []
        for img_path in img_paths:
            with open(img_path, 'rb') as f:
                image = Image.open(f)
                image = np.array(image)
                images.append(image)

        return torch.Tensor(images)

