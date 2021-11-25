# author: MR 2021

import numpy as np
from tqdm import tqdm


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def pairwise_euclidean_distances(positions_1, positions_2):
    """
    Computes the pairwise distances between the provided positions from set1 and set2.

    If the sets are too large, consider processing in chunks.

    :param positions_1: (N, 3) ndarray with x y z
    :param positions_2: (M, 3) ndarray with x y z

    :return: (N, M) matrix of distances
    """
    # make sure correct shape (can be different if only one element in set)
    positions_1 = positions_1.reshape(-1, 3)
    positions_2 = positions_2.reshape(-1, 3)

    # shape: (N, M, 3) = (N, 1, 3) - (1, M, 3)
    distances = np.linalg.norm(positions_1[:, 0:3][:, np.newaxis, :] - positions_2[:, 0:3][np.newaxis, :, :], axis=-1)
    return distances


def pairwise_angular_distances(quaternions_1, quaternions_2):
    """
    Computes the pairwise radian distances between the provided quaternions from set1 and set2.

    If the sets are too large, consider processing in chunks.

    :param quaternions_1: (N, 4) ndarray with w x y z
    :param quaternions_2: (M, 4) ndarray with w x y z

    :return: (N, M) matrix of radian distances
    """
    # make sure correct shape (can be different if only one element in set)
    quaternions_1 = quaternions_1.reshape(-1, 4)
    quaternions_2 = quaternions_2.reshape(-1, 4)

    angles = 2 * np.arccos(np.abs(quaternions_1.dot(np.transpose(quaternions_2, [1, 0]))))
    return angles


def pairwise_combined_distances(graspset1, graspset2):
    """
    Computes the pairwisse distances between the provided grasps from set1 and set2.

    This is a vectorized implementation, but should not be used if both sets are extremely large.

    :param graspset1: (N, 7) ndarray with 0:3 pos, 3:7 quaternion
    :param graspset2: (M, 7) ndarray with 0:3 pos, 3:7 quaternion

    :return: (N, M) matrix of distances (1 = 1mm or 1deg)
    """
    # also scale euclidean distance so that 1 = 1mm = 1 deg
    distances = pairwise_euclidean_distances(graspset1[:, 0:3], graspset2[:, 0:3]) * 1000
    distances += np.rad2deg(pairwise_angular_distances(graspset1[:, 3:7], graspset2[:, 3:7]))
    return distances


def rule_based_eval(gt_grasps, predictions, d_th=0.025, q_th=np.pi/6, coverage=False):
    """
    This is the rule-based evaluation as used in GPNet, i.e. only considering positive grasps.

    Parameters
    ----------
    gt_grasps: ndarray, 0: 0/1 label, 1:4 pos, 4:8 quat
    predictions: 0:3 pos, 3:7 quat
    d_th: distance threshold
    q_th: quaternion threshold
    coverage: bool, whether to produce coverage of gt data as well. If True, we will assume predictions are in
              descending order and for each prediction we will produce the cumulated coverage of GT grasps.
              Note that this is not vectorised and may take a lot of time if len(predictions) is large.

    Returns
    -------
    labels: bool array with True/False for each prediction; if coverage=True then also a float array with accumulated
            coverage for each prediction (accumulated by all previous predictions including the current one)
    """
    # make sure to only use positive gt_grasps
    positives = gt_grasps[:, 0] == 1.0
    gt_grasps = gt_grasps[positives]
    if len(gt_grasps) == 0:
        raise ValueError('No more values remaining after filtering gt_grasps (set first elem to label!).')

    # processing chunks of predictions against all gt_grasps
    chunk_size = 3000  # might need adaptive chunk size

    list_of_bool_arrays = []
    list_of_cov_arrays = []
    gt_covered = None
    if coverage:
        gt_covered = np.full(len(gt_grasps), fill_value=False, dtype=bool)

    for prediction in chunks(predictions, chunk_size):
        # separate thresholds for euclidean and angular distance
        cent_in_gt = pairwise_euclidean_distances(prediction[:, 0:3], gt_grasps[:, 1:4]) < d_th
        quat_in_gt = pairwise_angular_distances(prediction[:, 3:7], gt_grasps[:, 4:8]) < q_th
        matches = cent_in_gt * quat_in_gt
        pred_in_gt = np.any(matches, 1)
        list_of_bool_arrays.append(pred_in_gt)

        if coverage:
            # compute cumulated coverage, we track coverage status in gt_covered
            cumulated_cov = np.empty(len(prediction), dtype=float)
            for i in range(len(cumulated_cov)):
                gt_covered |= matches[i]
                cumulated_cov[i] = np.mean(gt_covered)
            list_of_cov_arrays.append(cumulated_cov)

    precision = np.concatenate(list_of_bool_arrays).flatten()
    if coverage:
        coverage = np.concatenate(list_of_cov_arrays).flatten()
        return precision, coverage
    return precision


def knn_based_eval(gt_grasps, predictions, n=5, th=0.015):
    """
    Classify predictions using knn and gt_grasps as reference set.

    Parameters
    ----------
    gt_grasps: ndarray, 0: 0/1 label, 1:4 pos, 4:8 quat, ...
    predictions: 0:3 pos, 3:7 quat, ...
    n: max number of neighbours to consider for majority voting
    th: if some of the n neighbours are not within this radius, they will not be considered

    Returns
    -------
    labels: bool array with True/false for each prediction
    """
    gt_grasps = gt_grasps[:, 0:8].reshape(-1, 8)
    predictions = predictions[:, 0:7].reshape(-1, 7)

    chunk_size = 4000
    list_of_bool_arrays = []
    for j, pred_chunk in enumerate(chunks(predictions, chunk_size)):
        print(f'processing predictions ({j*chunk_size+1}-{j*chunk_size + len(pred_chunk)})...')

        # for each gt chunk we gather the n shortest distances (and the corresponding indices)
        smallest_distances_list = []
        smallest_distances_idx_list = []
        for i, gt_chunk in tqdm(enumerate(chunks(gt_grasps, chunk_size))):

            distances = pairwise_combined_distances(pred_chunk, gt_chunk[:, 1:])

            # partition puts all values smaller than the nth to the left, all bigger to the right
            # argpartition gives the original indices of these elements (same shape as distances)
            # take_along_axis then allows to index the distances matrix with the index matrix
            smallest_distances_idx = np.argpartition(distances, n-1, axis=1)[:, :n]
            smallest_distances = np.take_along_axis(distances, smallest_distances_idx, axis=1)

            # need to add chunk offsets to the indices so that we reference gt_grasps instead of gt_chunk
            smallest_distances_idx += i * chunk_size

            smallest_distances_idx_list.append(smallest_distances_idx)
            smallest_distances_list.append(smallest_distances)

        # concatenate and evaluate the best hits
        smallest_distances = np.concatenate(smallest_distances_list, axis=1)
        smallest_distances_idx = np.concatenate(smallest_distances_idx_list, axis=1)
        mins_indices = np.argpartition(smallest_distances, n-1, axis=1)[:, :n]

        # gather the labels from the best hits
        gt_indices = np.take_along_axis(smallest_distances_idx, mins_indices, axis=1)
        scores = gt_grasps[gt_indices, 0]

        # check distance threshold
        if th is not None:
            mins_values = np.take_along_axis(smallest_distances, mins_indices, axis=1)
            scores[mins_values >= th * 1000] = 0.5  # they will have no effect on the decision

        means = np.mean(scores, axis=1)
        # print(means)
        list_of_bool_arrays.append(means > 0.5)

    return np.concatenate(list_of_bool_arrays)


def precision_at_k_percent(labels, k_percent_list=None):
    """
    For a given array with success values (labels), we will compute the precision at various steps.
    Counting from the front, we will use the first k percent of items in the list (at least 1).
    k_percent_list defaults to [0.1, 0.3, 0.5, 1.0]

    returns list of precisions and list of k_numbers
    """
    if k_percent_list is None:
        k_percent_list = [0.1, 0.3, 0.5, 1.0]

    if len(labels) == 0:
        # no predictions for this shape
        return [np.nan]*len(k_percent_list), [0]*len(k_percent_list)

    precisions = []
    k_numbers = []
    for k_percent in k_percent_list:
        k = int(k_percent * len(labels))
        if k == 0:
            k = 1
        precisions.append(np.mean(labels[:k]))
        k_numbers.append(k)

    return precisions, k_numbers


def precision_at_k_score(labels, scores, k_score_list=None):
    """
    For a given pair of labels and scores, we will compute the precision above a certain score level.
    The label is the success (0/1) and the scores the confidence values ([0, 1])
    k_score_list defaults to [0.95, 0.9, 0.75, 0.5]

    returns list of precisions and list of k_numbers
    """
    assert len(labels) == len(scores)

    if k_score_list is None:
        k_score_list = [0.95, 0.9, 0.75, 0.5]

    precisions = []
    k_numbers = []
    for k in k_score_list:
        contributing = scores > k
        k_num = np.count_nonzero(contributing)
        k_numbers.append(k_num)
        if k_num == 0:
            precisions.append(np.nan)
        else:
            precisions.append(np.mean(labels[contributing]))

    return precisions, k_numbers


def get_minimum_eppner_dist(gt_grasps, predictions, weight_factor=1000.0):
    """

    Parameters
    ----------
    gt_grasps: ndarray, 0: 0/1 label, 1:4 pos, 4:8 quat
    predictions: 0:3 pos, 3:7 quat
    weight_factor: for computation of combined euclidean and angular distance: w * d_euc[m] + d_ang[deg]

    Returns
    -------
    min combined distance (to next positive GT grasp)
    """
    import burg_toolkit as burg
    # todo: get rid of burg dependency - i think there is a method here that will compute combined distances
    chunk_size = 4000

    # make sure to only use positive gt_grasps
    positives = gt_grasps[:, 0] == 1.0
    gt_grasps = gt_grasps[positives]
    if len(gt_grasps) == 0:
        raise ValueError('No more values remaining after filtering gt_grasps (set first elem to label!).')

    min_distances = np.full(shape=(len(predictions)), fill_value=np.inf)
    for pred_i, preds in enumerate(chunks(predictions, chunk_size)):
        print(f'preds {pred_i * chunk_size}:{(pred_i+1)*chunk_size}')
        gs_preds = burg.grasp.GraspSet.from_translations_and_quaternions(preds[:, :3], preds[:, 3:7])
        # compute the min distance to the gt grasps
        for gt_i, gt in tqdm(enumerate(chunks(gt_grasps, chunk_size))):
            gs_gt = burg.grasp.GraspSet.from_translations_and_quaternions(gt[:, 1:4], gt[:, 4:8])
            dists = np.nanmin(burg.metrics.combined_distances(gs_preds, gs_gt, weight=weight_factor), axis=-1)
            min_distances[pred_i * chunk_size:(pred_i+1)*chunk_size] = \
                np.minimum(dists, min_distances[pred_i * chunk_size:(pred_i+1)*chunk_size])

    return min_distances
