import os
import shutil
import tempfile

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# try importing simulation
try:
    import gpnet_sim
except ImportError:
    gpnet_sim = None
    print('Warning: package gpnet_sim not found. Please install from https://github.com/mrudorfer/GPNet-simulator')

from . import metrics
from . import io_utils
from . import tools


def check_directory_parsing(dataset_root, test_dir):
    # use this to make sure the parsing of directories is correct and everything is found
    shapes = io_utils.read_test_shapes(dataset_root)
    print('read shapes:', shapes)

    shape = shapes[0]
    gt_grasps = io_utils.load_gt_grasps(dataset_root, shape)
    print(f'loaded all grasps for shape {shape}, having shape: {gt_grasps.shape}')

    gt_grasps = io_utils.load_gt_grasps(dataset_root, shape, which='positives')
    print(f'loaded positive grasps for shape {shape}, having shape: {gt_grasps.shape}')

    epoch_list = io_utils.get_epochs_and_views(test_dir)
    print(f'parsed directory: {test_dir}\nlooked for epochs and views. found the following:\n{epoch_list}')


def evaluate(dataset_root, test_dir, nms, use_sim, object_models_dir=None, coverage=False):
    # this creates a temporary file in the tmp dir of the operating system
    # we use it as interface to simulation
    sim_config = None
    if use_sim:
        if gpnet_sim is None:
            raise ImportError('cannot simulate grasp samples as package gpnet_sim is not loaded.')
        sim_config = gpnet_sim.default_conf()
        sim_config.z_move = True
        if object_models_dir is not None:
            sim_config.objMeshRoot = object_models_dir

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
                # predictions may not have a score, i.e. length 7 or 8
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
                sim_file_handle, sim_file = tempfile.mkstemp(suffix='.txt', text=True)
                sim_config.testFile = sim_file
                open(sim_file, 'w').close()  # clear file
                io_utils.write_shape_poses_file(grasp_predictions, sim_file, with_score=not nms)
                gpnet_sim.simulate(sim_config)
                sim_grasp_results = io_utils.read_sim_csv_file(sim_file[:-4] + '_log.csv')
                shutil.move(sim_file[:-4] + '_log.csv', os.path.join(test_dir, f'epoch{epoch}', f'nms_poses_view{view}_log.csv'))
                # os.remove(sim_file[:-4] + '_log.csv')
                # dict with shape as key, and grasp array
                # (n, 10): 0:3 pos, 3:7 quat, annotation id, sim result, sim success, empty

            result_type = ([
                                ('epoch', np.uint16),
                                ('view', np.uint16),
                                ('shape', 'S32'),
                                ('prediction_confidence', np.float),
                                ('simulation_result', np.uint8),
                                ('rule_based_success', np.bool)
                            ])
            if coverage:
                result_type.append(('rule_based_coverage', np.float))
            view_results = np.empty(n_grasps, dtype=result_type)
            view_results['epoch'] = epoch
            view_results['view'] = view
            idx = 0

            print('computing rule-based success...')
            for shape in tqdm(shapes):
                if shape not in grasp_predictions.keys() or len(grasp_predictions[shape]) == 0:
                    print(f'WARNING: no predictions for shape {shape}')
                    continue
                preds = grasp_predictions[shape]

                view_results['shape'][idx:idx + len(preds)] = shape
                if preds.shape[1] < 8:
                    # no confidence available
                    view_results['prediction_confidence'][idx:idx + len(preds)] = 0
                else:
                    view_results['prediction_confidence'][idx:idx + len(preds)] = preds[:, 7]

                if use_sim:
                    if shape not in sim_grasp_results.keys():
                        print(f'WARNING: no simulation results for shape {shape} -- this is unexpected.')
                        continue
                    view_results['simulation_result'][idx:idx + len(preds)] = sim_grasp_results[shape][:, 8]
                else:
                    # this status flag is internal to gpnet_sim, should actually try to retrieve there
                    view_results['simulation_result'][idx:idx + len(preds)] = 7

                # in order to support simulation-based results only as well, if no annotations available
                try:
                    gt_grasps = io_utils.load_gt_grasps(dataset_root, shape, which='positives')
                except FileNotFoundError:
                    print('WARNING: no ground truth annotation files found')
                    gt_grasps = np.empty(0)

                # compute rule based success
                if len(gt_grasps) == 0:  # there may be some weird shapes with no working GT grasps?
                    if coverage:
                        view_results['rule_based_coverage'][idx:idx + len(preds)] = np.nan
                    view_results['rule_based_success'][idx:idx + len(preds)] = np.nan
                else:  # this should be the default
                    if coverage:
                        rb_success, cumulated_coverage = metrics.rule_based_eval(gt_grasps, preds, coverage=True)
                        view_results['rule_based_coverage'][idx:idx + len(preds)] = cumulated_coverage
                    else:
                        rb_success = metrics.rule_based_eval(gt_grasps, preds, coverage=False)
                    view_results['rule_based_success'][idx:idx + len(preds)] = rb_success

                idx += len(preds)

            all_results_list.append(view_results)

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
        log.write('rule based results with predictions that have score above k%, aggregated over all views of a shape')
        log.write('\n')

    data = np.load(os.path.join(test_dir, 'evaluation_results.npy'))
    epochs = np.unique(data['epoch'])
    for epoch in epochs:
        mask = data['epoch'] == epoch
        epoch_data = data[mask]

        all_precisions_sim = []
        all_precisions_rb = []
        all_k_values = []

        k_steps = [0.95, 0.9, 0.75, 0.5]
        with open(log_fn, 'a') as log:
            headers = [['epoch'], ['shape'],
                       [f't{k}_k' for k in k_steps],
                       [f't{k}_sim' for k in k_steps],
                       [f't{k}_rb' for k in k_steps]]
            log.write(tools.log_line(tools.flatten_nested_list(headers)))

        for shape in shapes:
            mask = np.array(epoch_data['shape'], dtype='<U32') == shape
            shape_data = epoch_data[mask]

            sim_success = shape_data['simulation_result'] == 0
            confidence = shape_data['prediction_confidence']  # may not be provided

            precision_at_k_sim, k_values = metrics.precision_at_k_score(sim_success, confidence, k_score_list=k_steps)
            precision_at_k_rb, _ = metrics.precision_at_k_score(shape_data['rule_based_success'], confidence,
                                                                k_score_list=k_steps)

            with open(log_fn, 'a') as log:
                write_items = [[epoch, shape], k_values, precision_at_k_sim, precision_at_k_rb]
                write_items = tools.flatten_nested_list(write_items)
                log.write(tools.log_line(write_items))

            all_k_values.append(np.array(k_values))
            all_precisions_sim.append(np.array(precision_at_k_sim))
            all_precisions_rb.append(np.array(precision_at_k_rb))

        all_precisions_rb = np.nanmean(np.array(all_precisions_rb), axis=0)
        all_precisions_sim = np.nanmean(np.array(all_precisions_sim), axis=0)
        all_k_values = np.mean(np.array(all_k_values), axis=0)
        with open(log_fn, 'a') as log:
            write_items = [[epoch, 'avg'], all_k_values, all_precisions_sim, all_precisions_rb]
            write_items = tools.flatten_nested_list(write_items)
            log.write(tools.log_line(write_items))
            log.write('\n')

    print('stored per shape statistics in per_shape_stats.txt')


def standard_statistics(dataset_root, test_dir):
    """
    computes the regular topk% rule-based and simulation success rates for each epoch/view.

    :param dataset_root: dataset base directory
    :param test_dir: directory containing the evaluation file (evaluation_results.npy)

    :return:
    """
    shapes = io_utils.read_test_shapes(dataset_root)
    k_steps = [0, 0.1, 0.3, 0.5, 1.0]

    data = np.load(os.path.join(test_dir, 'evaluation_results.npy'))
    # check whether file has coverage data (has been added later, so we want to ensure backwards compatibility)
    use_coverage = 'rule_based_coverage' in data.dtype.names
    if not use_coverage:
        print('note: evaluation_results.npy does not contain coverage data.')

    log_fn = os.path.join(test_dir, 'standard_stats.txt')
    with open(log_fn, 'w') as log:
        log.write(test_dir)
        log.write('rule based results with topk% predictions, averaged over all shapes')
        log.write('\n')

        headers = [['epoch'], ['view'],
                   ['best_k'], [f't{k}_k' for k in k_steps[1:]],
                   ['best_sim'], [f't{k}_sim' for k in k_steps[1:]],
                   ['best_rb'], [f't{k}_rb' for k in k_steps[1:]]]
        if use_coverage:
            headers.append(['best_cov'])
            headers.append([f't{k}_cov' for k in k_steps[1:]])
        log.write(tools.log_line(tools.flatten_nested_list(headers)))

    epochs = np.unique(data['epoch'])
    for epoch in epochs:
        mask = data['epoch'] == epoch
        epoch_data = data[mask]
        views = np.unique(epoch_data['view'])

        all_precisions_sim = []
        all_precisions_rb = []
        all_k_values = []
        all_coverage = []

        for view in views:
            mask = epoch_data['view'] == view
            view_data = epoch_data[mask]

            view_precisions_sim = []
            view_precisions_rb = []
            view_k_values = []
            view_coverage = []

            for shape in shapes:
                mask = np.array(view_data['shape'], dtype='<U32') == shape
                shape_data = view_data[mask]

                sim_success = shape_data['simulation_result'] == 0
                precision_at_k_sim, k_values = metrics.precision_at_k_percent(sim_success, k_percent_list=k_steps)
                precision_at_k_rb, _ = metrics.precision_at_k_percent(shape_data['rule_based_success'],
                                                                      k_percent_list=k_steps)
                if use_coverage:
                    if k_values[0] == 0:  # all zeros, i.e. no predictions
                        coverage_at_k = [np.nan]*len(k_steps)
                    else:
                        k_indices = np.array(k_values) - 1
                        coverage_at_k = shape_data[k_indices]['rule_based_coverage']
                    view_coverage.append(coverage_at_k)

                view_k_values.append(np.array(k_values))
                view_precisions_rb.append(np.array(precision_at_k_rb))
                view_precisions_sim.append(np.array(precision_at_k_sim))

            view_precisions_rb = np.nanmean(np.array(view_precisions_rb), axis=0)
            view_precisions_sim = np.nanmean(np.array(view_precisions_sim), axis=0)
            view_k_values = np.mean(np.array(view_k_values), axis=0)
            if use_coverage:
                view_coverage = np.nanmean(np.array(view_coverage), axis=0)

            with open(log_fn, 'a') as log:
                write_items = [[epoch, view], view_k_values, view_precisions_sim, view_precisions_rb]
                if use_coverage:
                    write_items.append(view_coverage)
                write_items = tools.flatten_nested_list(write_items)
                log.write(tools.log_line(write_items))

            all_k_values.append(view_k_values)
            all_precisions_sim.append(view_precisions_sim)
            all_precisions_rb.append(view_precisions_rb)
            all_coverage.append(view_coverage)

        all_precisions_rb = np.nanmean(np.array(all_precisions_rb), axis=0)
        all_precisions_sim = np.nanmean(np.array(all_precisions_sim), axis=0)
        all_k_values = np.mean(np.array(all_k_values), axis=0)

        with open(log_fn, 'a') as log:
            write_items = [[epoch, 'avg'], all_k_values, all_precisions_sim, all_precisions_rb]
            if use_coverage:
                all_coverage = np.nanmean(np.array(all_coverage), axis=0)
                write_items.append(all_coverage)
            write_items = tools.flatten_nested_list(write_items)
            log.write(tools.log_line(write_items))
            log.write('\n')

    print('stored standard statistics in standard_stats.txt')


def precision_coverage_curve(dataset_root, test_dir, resolution=21):
    """
    creates precision coverage curves aggregated over all views of a specific epoch

    :param dataset_root: dataset base directory
    :param test_dir: directory containing the evaluation file (evaluation_results.npy)
    :param resolution: number of coverage steps from 0.0 to 1.0, e.g. 101 means compute precision at every 0.01 cov

    :return:
    """
    data = np.load(os.path.join(test_dir, 'evaluation_results.npy'))
    use_coverage = 'rule_based_coverage' in data.dtype.names
    if not use_coverage:
        print('cannot generate precision/coverage curve as no coverage data available.')
        print('recreate the evaluation_results.npy file')
        return
    if data['prediction_confidence'].mean() == 0:
        print('cannot generate precision/coverage curve as prediction confidence is not available.')
        print('redo the evaluation by providing nms/npz files that contain prediction confidence')
        return

    shapes = io_utils.read_test_shapes(dataset_root)
    score_steps = np.linspace(1.0, 0.5, num=resolution)

    plt.set_cmap('tab10')
    plots_by_epoch = {}

    epochs = np.unique(data['epoch'])
    for epoch in epochs:
        mask = data['epoch'] == epoch
        epoch_data = data[mask]

        coverages = []
        success_rates = []

        for score in score_steps:
            # for all shapes in all views, compute success_rate and coverage for predictions above score
            # average of those is our value pair for the plot
            score_data = epoch_data[epoch_data['prediction_confidence'] >= score]
            if len(score_data) == 0:
                continue
            score_coverages = []
            score_success_rates = []
            views = np.unique(score_data['view'])
            for view in views:
                view_data = score_data[score_data['view'] == view]
                for shape in shapes:
                    mask = np.array(view_data['shape'], dtype='<U32') == shape
                    shape_data = view_data[mask]
                    if len(shape_data) > 0:
                        score_coverages.append(np.max(shape_data['rule_based_coverage']))
                        success = shape_data['simulation_result'] == 0
                        score_success_rates.append(np.mean(success))
                    else:
                        # if no grasps at this confidence level available, then we do not contribute a success rate
                        # however, the coverage of this shape is zero
                        score_coverages.append(0)

            # only append if overall coverage is larger than before
            c = np.mean(score_coverages)
            s = np.mean(score_success_rates)
            if len(coverages) == 0 or c > coverages[-1]:
                coverages.append(c)
                success_rates.append(s)

        # now interpolate the plot as is usually done for precision-recall-plots
        # taken from here: https://stackoverflow.com/a/39862264/1264582
        # running maximum over the reversed vector of precision values, reverse the result back
        precision = np.array(success_rates)
        decreasing_max_precision = np.maximum.accumulate(precision[::-1])[::-1]

        fig, ax = plt.subplots(1, 1)
        ax.plot(coverages, decreasing_max_precision, label='max precision')
        plt.xlabel('coverage')
        plt.ylabel('success rate')
        # plt.axis([0, 1, 0, 1])  # turns out this was not useful, as the relevant part of the figure is quite small
        fn = f'ep{epoch}_precision_coverage.png'
        plt.savefig(os.path.join(test_dir, fn))
        print(f'saved {fn}')

        plots_by_epoch[epoch] = coverages, decreasing_max_precision

    # finally, make a plot with all epochs in it
    fig, ax = plt.subplots(1, 1)
    for key, item in plots_by_epoch.items():
        coverage, success_rate = item
        ax.plot(coverage, success_rate, label=str(key))

    plt.xlabel('coverage')
    plt.ylabel('success rate')
    plt.legend()
    fn = 'all_epochs_precision_coverage.png'
    plt.savefig(os.path.join(test_dir, fn))
    print(f'saved {fn}')
