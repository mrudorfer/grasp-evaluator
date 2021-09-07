import os
import argparse

from . import metrics
from . import io_utils


def check_all_the_shit_works(dataset_root, test_dir):
    shapes = io_utils.read_test_shapes(dataset_root)
    print('read shapes:', shapes)

    shape = shapes[0]
    gt_grasps = io_utils.load_gt_grasps(dataset_root, shape)
    print(f'loaded all grasps for shape {shape}, having shape: {gt_grasps.shape}')

    gt_grasps = io_utils.load_gt_grasps(dataset_root, shape, which='positives')
    print(f'loaded positive grasps for shape {shape}, having shape: {gt_grasps.shape}')

    epoch_list = io_utils.get_epochs_and_views(test_dir)
    print(f'parsed directory: {test_dir}\nlooked for epochs and views. found the following:\n{epoch_list}')
