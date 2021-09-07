# author: MR 2021
import csv
import os

import numpy as np
from tqdm import tqdm


def load_gt_grasps(dataset_root, shape, which='all'):
    """
    This will read the ground truth grasps for a given shape.

    :param dataset_root: base directory of the dataset
    :param shape: shape id
    :param which: declare which grasps you want: 'all', 'positives', 'negatives' (defaults to 'all')

    :return: ndarray with label[0], pos[1:4] and quat[4:8]
    """
    centers = np.load(os.path.join(dataset_root, 'annotations/candidate/', shape + '_c.npy'))
    quats = np.load(os.path.join(dataset_root, 'annotations/candidate/', shape + '_q.npy'))
    labels = np.load(os.path.join(dataset_root, 'annotations/simulateResult/', shape + '.npy'))

    if which != 'all':
        if which == 'positives':
            indices = np.nonzero(labels)
        elif which == 'negatives':
            indices = np.nonzero(1-labels)
        else:
            raise ValueError(f'parameter "which" must be one of "all", "positives", "negatives", but got "{which}"')
        centers = centers[indices]
        quats = quats[indices]
        labels = labels[indices]

    gt_grasps = np.concatenate([labels[:, np.newaxis], centers, quats], axis=1)
    return gt_grasps


def read_test_shapes(dataset_root):
    """
    gets the list of shapes in the test set
    """
    shapes = []

    filename = os.path.join(dataset_root, 'test_set.csv')
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            shapes.append(row[0])

    return shapes


def find_numbered_directories(base_dir, prefix):
    """
    finds directories within base_dir, which start with prefix followed by a number.
    returns a list of the numbers as integer.

    :param base_dir: the search directory
    :param prefix: string, the prefix (e.g. 'view' or 'epoch')

    :return: list of ints
    """
    num_list = []
    all_dirs = [x for x in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, x))]
    all_dirs = sorted(all_dirs)
    for candidate_dir in all_dirs:
        if not candidate_dir.startswith(prefix):
            # skip irrelevant folders
            continue

        number = candidate_dir[len(prefix):]
        if not number.isdigit():
            print(f'WARNING: skipping dodgy candidate folder {candidate_dir}, unable to extract number, found {number}')
            continue
        num_list.append(int(number))

    return num_list


def get_epochs_and_views(test_path):
    """
    parses the given directory and returns a list with epoch indices and view indices per epoch
    :return: nested list, first level contains (epoch, views), where views is a list of the view numbers
    """
    epoch_list = []

    epochs = find_numbered_directories(test_path, 'epoch')
    for epoch in epochs:
        epoch_dir = os.path.join(test_path, 'epoch' + str(epoch))
        views = find_numbered_directories(epoch_dir, 'view')
        epoch_list.append((epoch, views))

    return epoch_list


def read_sim_csv_file(filename, keep_num=None):
    """
    This reads the csv log file created during simulation.

    :return: returns a dict with shape id as keys and np array as value.
             the np array is of shape (n, 10): 0:3 pos, 3:7 quat, annotation id, sim result, sim success, empty
             keeps only keep_num entries (as of annotation idx order, which is ordered by descending score)
    """
    print(f'reading csv data from {filename}')
    sim_data = {}
    counters = {}
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in tqdm(reader):
            shape = row[0]
            if shape not in sim_data.keys():
                # we do not know the array length in advance, so start with 10k (will usually be less)
                data_array = np.zeros((10000, 11))
                sim_data[shape] = data_array
                counters[shape] = 0
            elif counters[shape] == len(sim_data[shape]):
                np.resize(sim_data[shape], (len(sim_data[shape]) + 10000, 11))

            sim_data[shape][counters[shape]] = [
                float(row[4]),  # pos: x, y, z
                float(row[5]),
                float(row[6]),
                float(row[10]),  # quat: w, x, y, z, converted from pybullet convention
                float(row[7]),
                float(row[8]),
                float(row[9]),
                int(row[1]),  # annotation id
                int(row[2]),  # simulation result
                int(row[2]) == 0,  # simulation success flag
                -1.   # left empty for rule-based success flag
            ]
            counters[shape] += 1

    # now reduce arrays to their actual content
    for key in sim_data.keys():
        sim_data[key] = np.resize(sim_data[key], (counters[key], 11))
        # also sort by annotation id
        order = np.argsort(sim_data[key][:, 7])
        sim_data[key] = sim_data[key][order]
        sim_data[key] = sim_data[key][:keep_num]

    return sim_data


def read_nms_poses_file(filename):
    """
    This reads the "nms_poses_view0.txt" file created after using NMS.
    It is equal to the "readShapePoses(fname)" function which can be found in a couple of files...

    Retuns a dict with shape names as key and list of grasps as values.
    Depending on which file type it is, it may have an additional score value.
    """
    shape_poses = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for l in lines:
            if not ',' in l:
                shape = l.strip()
                shape_poses[shape] = []
            else:
                grasp = np.array(l.strip().split(',')).astype(float).reshape(1, -1)
                shape_poses[shape].append(grasp)

    return shape_poses


def read_npz_files(npz_dir, limit=None):
    """
    Reads npz files from a view0 folder.
    Returns a dict with shape as key and a (n, 8) grasp array with 0:3 pos, 3:7 quat, 7 score
    You can limit the number of samples that will be loaded.
    """
    obj_dict = {}

    for file in tqdm(os.listdir(npz_dir)):
        fn = os.path.join(npz_dir, file)
        if os.path.isfile(fn) and file[-4:] == '.npz':
            shape = file[:-4]
            with np.load(fn) as data:
                # sort by score
                data_array = np.concatenate(
                    [data['centers'], data['quaternions'], data['scores'].reshape(-1, 1)], axis=1)
                order = np.argsort(-data['scores'])
                obj_dict[shape] = data_array[order][:limit]

    return obj_dict


def write_shape_poses_file(grasp_data, file_path, with_score=False):
    """
    Parameters
    ----------
    grasp_data: dict with shapes as keys and grasps as ndarrays
    file_path: full path to the file to write it in - will be overwritten
    with_score: bool, determines whether or not to add the scores to the file

    Returns
    -------
    creates or overwrites a file with shapes and poses (as used in simulation for example)
    """
    with open(file_path, 'w') as f:
        for shape, grasps in grasp_data.items():
            if len(grasps) > 0:
                f.write(shape + '\n')
                for g in grasps:
                    if with_score:
                        f.write('%f,%f,%f,%f,%f,%f,%f,%f\n' % (g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7]))
                    else:
                        f.write('%f,%f,%f,%f,%f,%f,%f\n' % (g[0], g[1], g[2], g[3], g[4], g[5], g[6]))
