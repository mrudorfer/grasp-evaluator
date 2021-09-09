# GPNet-evaluator

Project to evaluate predictions on GPNet dataset.
See the [GPNet repository](https://github.com/CZ-Wu/GPNet), 
which contains the Pytorch implementation of the GPNet paper:
[Grasp Proposal Networks: An End-to-End Solution for Visual Learning of Robotic Grasps](https://arxiv.org/abs/2009.12606).

This repo also makes use of the [GPNet-simulator](https://github.com/mrudorfer/GPNet-simulator).

## Installation

```
cd GPNet-evaluator
pip install .
```

This will install all dependencies including the GPNet-simulator.
However, it is recommended to install the simulator manually in editable mode.
This way the simulator can find the object models and gripper models contained in the package.
If you install without editable mode, you will need to provide the correct paths yourself.


## Usage

Can be used via CLI like this:

```
python -m gpnet_eval --dataset_root GPNet_release_data/ --test_dir network_predictions/
```

In this configuration, `gpnet_eval` will look for predictions of all epochs and all views based on the npz files.
It will compute both simulation and rule-based success values and store them in `evaluation_results.npy` in `test_dir`.
Based on this, it computes several statistics which will be saved in individual text files in the same folder.

Use `--help` for more info on usage:
```
usage: python -m gpnet_eval [-h] [-DR DATASET_ROOT] [-TD TEST_DIR] [--nms] [--no_sim] [--stats_only]

optional arguments:
  -h, --help            show this help message and exit
  -DR DATASET_ROOT, --dataset_root DATASET_ROOT
                        root directory of the GPNet dataset
  -TD TEST_DIR, --test_dir TEST_DIR
                        test folder of the network (containing epoch dirs)
  --nms                 use this option to use "nms_poses_view0.txt" file instead of all
                        predictions (npz files)
  --no_sim              use this option to skip simulation results (only rule-based evaluation)
  --stats_only
```