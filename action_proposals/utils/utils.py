import errno
import os
import json
import urllib.request as urllib2


import numpy as np
import argparse
from ruamel import yaml
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


def cover_args_by_yml(args: argparse.Namespace, yml_file_path: str):
    r"""Cover the args parsed by argparse using yaml file. It will ignore the argument passed by command line if the
    same arg is provided in yaml file.

    :param args: the return value of parser.parse_args()
    :param yml_file_path: The path of config yaml file
    :return: None. But the state of args will be changed.
    """
    with open(yml_file_path) as f:
        file_cfgs = yaml.safe_load(f)
    for arg_name in file_cfgs.keys():
        if arg_name in file_cfgs.keys():
            args.__setattr__(arg_name, file_cfgs[arg_name])


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

