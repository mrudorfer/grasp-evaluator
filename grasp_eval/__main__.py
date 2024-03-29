import argparse

from . import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-DR', '--dataset_root', type=str, help='root directory of the dataset')
    parser.add_argument('-TD', '--test_dir', type=str, help='test folder of the network (containing epoch dirs)')
    parser.add_argument('--nms', action='store_true',
                        help='use this option to use "nms_poses_view0.txt" file instead of all predictions (npz files)')
    parser.add_argument('--no_sim', action='store_false', dest='use_sim',
                        help='use this option to skip simulation results (only rule-based evaluation)')
    parser.add_argument('--no_cov', action='store_false', dest='use_coverage',
                        help='use this argument to skip computing coverage (shorten computation time). ' +
                        'However, if coverage is present in eval data it will be presented in stats in any case.')
    parser.add_argument('-OMD', '--object_models_dir', type=str,
                        help='please provide path to urdf files of objects (only required if using simulator)')
    parser.add_argument('--stats_only', action='store_true')
    parser.add_argument('--plot_res', type=int, default=21,
                        help='resolution (n data points) for success rate/coverage plots')
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()

    if not arguments.stats_only:
        evaluate.evaluate(
            arguments.dataset_root,
            arguments.test_dir,
            arguments.nms,
            arguments.use_sim,
            arguments.object_models_dir,
            arguments.use_coverage
        )

    evaluate.per_shape_stats(
        arguments.dataset_root,
        arguments.test_dir
    )

    evaluate.standard_statistics(
        arguments.dataset_root,
        arguments.test_dir
    )

    evaluate.precision_coverage_curve(
        arguments.dataset_root,
        arguments.test_dir,
        arguments.plot_res
    )
