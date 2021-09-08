import os
import argparse
import tempfile

import numpy as np

# try importing simulation
try:
    import gpnet_sim
except ImportError:
    gpnet_sim = None
    print('Warning: package gpnet_sim not found. Please install from https://github.com/mrudorfer/GPNet-simulator')


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


def evaluate(dataset_root, test_dir, nms, use_sim):
    # this creates a temporary file in the tmp dir of the operating system
    # we use it as interface to simulation
    sim_file_handle, sim_file = tempfile.mkstemp(suffix='.txt', text=True)
    sim_config = None
    if use_sim:
        if gpnet_sim is None:
            raise ImportError('cannot simulate grasp samples as package gpnet_sim is not loaded.')
        sim_config = gpnet_sim.default_conf()
        sim_config.z_move = True
        sim_config.testFile = sim_file

    shapes = io_utils.read_test_shapes(dataset_root)
    epoch_list = io_utils.get_epochs_and_views(test_dir)
    print(f'parsed directory: {test_dir}\nepochs and views:\n  {epoch_list}')

    all_results_list = []

    for epoch, views in epoch_list:
        for view in views:
            print(f'**\nepoch: {epoch}, view: {view}')
            # load predictions (assuming all of these are sorted with decreasing confidence)
            if nms:
                nms_fn = os.path.join(test_dir, f'epoch{epoch}', f'nms_poses_view{view}.txt')
                grasp_predictions = io_utils.read_nms_poses_file(nms_fn)
                # predictions do not have a score, i.e. length 7
            else:
                npz_dir = os.path.join(test_dir, f'epoch{epoch}', f'view{view}')
                grasp_predictions = io_utils.read_npz_files(npz_dir)
                # contains prediction score as last element! i.e. length 8

            n_grasps = 0
            for key, grasps in grasp_predictions.items():
                n_grasps += len(grasps)

            # produce simulation results (for all shapes combined should be much faster, because of parallelisation)
            if use_sim:
                print(f'simulating {n_grasps} grasp predictions')
                open(sim_file, 'w').close()  # clear file
                io_utils.write_shape_poses_file(grasp_predictions, sim_file, with_score=not nms)
                gpnet_sim.simulate(sim_config)
                sim_grasp_results = io_utils.read_sim_csv_file(sim_file[:-4] + '_log.csv')
                os.remove(sim_file[:-4] + '_log.csv')
                # dict with shape as key, and grasp array
                # (n, 10): 0:3 pos, 3:7 quat, annotation id, sim result, sim success, empty

            view_results = np.empty(n_grasps,
                                    dtype=([
                                        ('epoch', np.uint16),
                                        ('view', np.uint16),
                                        ('shape', 'S32'),
                                        ('prediction_confidence', np.float),
                                        ('simulation_result', np.uint8),
                                        ('rule_based_success', np.bool)
                                    ]))
            view_results['epoch'] = epoch
            view_results['view'] = view
            idx = 0

            for shape in shapes:
                preds = grasp_predictions[shape]

                view_results['shape'][idx:idx+len(preds)] = shape
                if nms:
                    # no confidence available
                    view_results['prediction_confidence'][idx:idx + len(preds)] = 0
                else:
                    view_results['prediction_confidence'][idx:idx + len(preds)] = preds[:, 7]

                if use_sim:
                    view_results['simulation_result'][idx:idx + len(preds)] = sim_grasp_results[shape][:, 8]
                else:
                    # this status flag is internal to gpnet_sim, should actually try to retrieve there
                    view_results['simulation_result'][idx:idx + len(preds)] = 7

                # compute rule based success
                gt_grasps = io_utils.load_gt_grasps(dataset_root, shape, which='positives')
                rb_success = metrics.rule_based_eval(gt_grasps, preds)
                view_results['rule_based_success'][idx:idx + len(preds)] = rb_success

                all_results_list.append(view_results)
                idx += len(preds)

    all_results = np.concatenate(all_results_list)
    save_file = os.path.join(test_dir, 'evaluation_results.npy')
    np.save(save_file, all_results)
    print(f'finished. stored all results in {save_file}')

    # clean up tmp file
    os.close(sim_file_handle)
    os.remove(sim_file)
