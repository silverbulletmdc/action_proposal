import os
from PIL import Image
import numpy as np
import torch, torchvision
from typing import List, Tuple, Iterable


class AnnotationRecord:

    def __init__(self, start_time, end_time, class_):
        self.start_time = start_time
        self.end_time = end_time
        self.class_ = class_


class VideoRecord:

    def __init__(self, video_name: str = '', frame_path: str = '', duration: int = 0, fps: float = 0,
                 proposals: List[AnnotationRecord] = None):
        self.video_name = video_name
        self.frame_path = frame_path
        self.duration = duration
        self.fps = fps
        self.proposals = proposals


class TemporalActionProposalDataset(torch.utils.Dataset):

    def __init__(self):
        super(TemporalActionProposalDataset, self).__init__()
        self.video_records = self._load_video_records()

    def __len__(self):
        return len(self.video_records)

    def __getitem__(self, idx):
        pass

    def _load_video_records(self) -> List[VideoRecord]:
        """Load your video records in this method and return it.

        :return:
        """
        raise NotImplementedError

    def _load_frames(self, idx: int, ranges: Iterable[int], modality: str = "RGB") -> torch.Tensor:
        """A tool function to load some frames.

        :param idx: Video idx
        :param ranges: Which frames to be loaded.
        :param modality: "RGB" or "Flow" and so on.
        :return:
        """

        frame_path = self.video_records[idx].frame_path
        img_paths = [os.path.join(frame_path, "{}_{}".format(modality, x)) for x in ranges]
        images = []
        for img_path in img_paths:
            with open(img_path, 'rb') as f:
                image = Image.open(f)
                image = np.array(image)
                images.append(image)

        return torch.Tensor(images)

