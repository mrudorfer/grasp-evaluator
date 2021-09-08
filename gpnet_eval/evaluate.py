import os
import argparse
import tempfile

import numpy as np
from tqdm import tqdm

# try importing simulation
try:
    import gpnet_sim
except ImportError:
    gpnet_sim = None
    print('Warning: package gpnet_sim not found. Please install from https://github.com/mrudorfer/GPNet-simulator')


from . import metrics
from . import io_utils
from . import tools


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
    print(f'test directory: {test_dir}')
    print('detected epochs and views:')
    for epoch, views in epoch_list:
        print(f'   epoch{epoch}: views: {views}')

    all_results_list = []

    for epoch, views in epoch_list:
        for view in views:
            print(f'***\nepoch: {epoch}, view: {view}')
            # load predictions (assuming all of these are sorted with decreasing confidence)
            if nms:
                nms_fn = os.path.join(test_dir, f'epoch{epoch}', f'nms_poses_view{view}.txt')
                # nms_fn = os.path.join(test_dir, f'epoch{epoch}', f'nms_poses_with_score.txt')
                grasp_predictions = io_utils.read_nms_poses_file(nms_fn)
                # predictions do not have a score, i.e. length 7
            else:
                npz_dir = os.path.join(test_dir, f'epoch{epoch}', f'view{view}')
                grasp_predictions = io_utils.read_npz_files(npz_dir)
                # contains prediction score as last element! i.e. length 8

            n_grasps = 0
            for key, grasps in grasp_predictions.items():
                n_grasps += len(grasps)
            print(f'got {n_grasps} grasp predictions')

            # produce simulation results (for all shapes combined should be much faster, because of parallelisation)
            if use_sim:
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

            print('computing rule-based success...')
            for shape in tqdm(shapes):
                if shape not in grasp_predictions.keys():
                    print(f'WARNING: no predictions for shape {shape}')
                    continue
                preds = grasp_predictions[shape]

                view_results['shape'][idx:idx+len(preds)] = shape
                if preds.shape[1] < 8:
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
    print(f'finished.\nstored all results in {save_file}')

    # clean up tmp file
    os.close(sim_file_handle)
    os.remove(sim_file)


def per_shape_stats(dataset_root, test_dir):
    """
    computes the success rates of the predictions with score above k%, aggregated across views, but separate for each
    epoch.
    requires the confidence values, which are usually not part of the nms_poses_view0.txt files!

    :param dataset_root: dataset base directory
    :param test_dir: directory containing the evaluation file (evaluation_results.npy)

    :return:
    """
    shapes = io_utils.read_test_shapes(dataset_root)

    log_fn = os.path.join(test_dir, 'per_shape_stats.txt')
    with open(log_fn, 'w') as log:
        log.write(test_dir)
        log.write('\n')

    data = np.load(os.path.join(test_dir, 'evaluation_results.npy'))
    epochs = np.unique(data['epoch'])
    for epoch in epochs:
        mask = data['epoch'] == epoch
        epoch_data = data[mask]

        all_precisions_sim = []
        all_precisions_rb = []
        all_k_values = []

        for shape in shapes:
            mask = np.array(epoch_data['shape'], dtype='<U32') == shape
            shape_data = epoch_data[mask]

            sim_success = shape_data['simulation_result'] == 0
            confidence = shape_data['prediction_confidence']  # may not be provided

            precision_at_k_sim, k_values = metrics.precision_at_k_score(sim_success, confidence)
            precision_at_k_rb, _ = metrics.precision_at_k_score(shape_data['rule_based_success'], confidence)

            with open(log_fn, 'a') as log:
                write_items = [[f'epoch{epoch}', shape], k_values, precision_at_k_sim, precision_at_k_rb]
                write_items = tools.flatten_nested_list(write_items)
                log.write(tools.log_line(write_items))

            all_k_values.append(np.array(k_values))
            all_precisions_sim.append(np.array(precision_at_k_sim))
            all_precisions_rb.append(np.array(precision_at_k_rb))

        all_precisions_rb = np.nanmean(np.array(all_precisions_rb), axis=0)
        all_precisions_sim = np.nanmean(np.array(all_precisions_sim), axis=0)
        all_k_values = np.mean(np.array(all_k_values), axis=0)
        with open(log_fn, 'a') as log:
            write_items = [[f'epoch{epoch}', 'avg'], all_k_values, all_precisions_sim, all_precisions_rb]
            write_items = tools.flatten_nested_list(write_items)
            log.write(tools.log_line(write_items))
            log.write('\n')

