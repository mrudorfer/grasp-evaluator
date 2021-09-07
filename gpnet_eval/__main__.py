import argparse

from . import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-DR', '--dataset_root', type=str, default='/home/rudorfem/datasets/GPNet_release_data_fixed/',
                        help='root directory of the GPNet dataset')
    parser.add_argument('-TD', '--test_dir', type=str, help='test folder of the network (containing epoch dirs)')
    parser.add_argument('--nms', action='store_true',
                        help='use this option to use "nms_poses_view0.txt" file instead of all predictions (npz files)')
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()

    evaluate.check_all_the_shit_works(
        arguments.dataset_root,
        '/home/rudorfem/dev/exp_GPNet_Deco/multi_view_epochs/gpnet'
    )
