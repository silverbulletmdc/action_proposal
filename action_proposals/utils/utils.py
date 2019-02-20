import errno
import os
import json
import urllib.request as urllib2


import numpy as np
import argparse
from easydict import EasyDict as edict
from action_proposals.utils.ap_yaml import my_safe_load
from typing import Union, List, Tuple
ArrayLike = Union[np.ndarray, List, Tuple]


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def load_yml(yml_file_path: str) -> edict:
    r"""Load a yml file and return an EasyDict Object.

    :param yml_file_path: The path of config yaml file
    :return: Easydict. But the state of args will be changed.
    """
    with open(yml_file_path) as f:
        file_cfgs = my_safe_load(f)

    return edict(file_cfgs)



def ioa_with_anchors(prop_xmin: float, prop_xmax: float, gt_xmins: ArrayLike, gt_xmaxs: ArrayLike):
    r"""Calc max IoA of a proposal with all gts.

    :param prop_xmin:
    :param prop_xmax:
    :param gt_xmins:
    :param gt_xmaxs:
    :return:
    """

    len_prop = prop_xmax - prop_xmin
    inter_xmin = np.maximum(prop_xmin, gt_xmins)
    inter_xmax = np.minimum(prop_xmax, gt_xmaxs)

    inter_len = np.maximum(inter_xmax - inter_xmin, 0)

    ioa = np.divide(inter_len, len_prop)

    return ioa


def iou_with_anchors(prop_xmin: float, prop_xmax: float, gt_xmins: ArrayLike, gt_xmaxs: ArrayLike):
    r"""
    Calc max IoU of a proposal with all gts.

    :param prop_xmin:
    :param prop_xmax:
    :param gt_xmins:
    :param gt_xmaxs:
    :return:
    """
    len_prop = prop_xmax - prop_xmin
    inter_xmin = np.maximum(prop_xmin, gt_xmins)
    inter_xmax = np.minimum(prop_xmax, gt_xmaxs)

    inter_len = np.maximum(inter_xmax - inter_xmin, 0)

    union_len = len_prop - inter_len + gt_xmaxs - gt_xmins
    iou = np.divide(inter_len, union_len)

    return iou


def IOU(s1: float, e1: float, s2: float, e2: float):
    """
    Calc IOU of two segments.

    :param s1:
    :param e1:
    :param s2:
    :param e2:
    :return:
    """
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor


def soft_nms(props: np.ndarray) -> np.ndarray:
    r"""Applying soft NMS to proposals.

    :param props: [N, 3], start, end, score
    :return: [N, 3], Recalc the scores.
    """
    # props = props[np.argsort(props[:, 2])]

    # 考虑到会有很多删除操作，用list比ndarray更高效
    tstart = list(props[:, 0])
    tend = list(props[:, 1])
    tscore = list(props[:, 2])

    rstart = []
    rend = []
    rscore = []

    while len(tscore) > 1 and len(rscore) < 101:
        max_index = tscore.index(max(tscore))
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = IOU(tstart[max_index], tend[max_index], tstart[idx], tend[idx])
                tmp_width = tend[max_index] - tstart[max_index]

                if tmp_iou > 0.65 + 0.25 * tmp_width:  # *1/(1+np.exp(-max_index)):
                    tscore[idx] = tscore[idx] * np.exp(-np.square(tmp_iou) / 0.75)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    return np.stack([rstart, rend, rscore]).T
