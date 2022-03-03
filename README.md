# Grasp Evaluator

## Context
During work on our [L2G: End-to-end Learning to Grasp from Object Point Clouds](https://github.com/antoalli/L2G), we used
this Python module to evaluate the predicted grasps and compute desired metrics.
It is based on the evaluation procedure in [GPNet](https://github.com/CZ-Wu/GPNet),
but extends its functionality by integrating further metrics such as per-shape-statistics, evaluation based on the 
predicted confidence score, and success-coverage plots.

Two different kinds of evaluation are supported:
- rule-based: compare predictions to grasp annotations from ground-truth dataset
- simulation-based: evaluate grasp success of predictions, using the [GPNet-simulator](https://github.com/mrudorfer/GPNet-simulator)

## Installation

To enable simulation-based evaluation, please first install [GPNet-simulator](https://github.com/mrudorfer/GPNet-simulator).
Then install this package with:
```
python setup.py install
```

## Usage

Can be used via CLI like this:

```
python -m grasp_eval --dataset_root ~/datasets/ShapeNetSem-8/ --test_dir network_predictions/
```

In this configuration, `grasp_eval` will look for predictions of all epochs and all views based on the npz files.
It will compute both simulation and rule-based success values and store them in `evaluation_results.npy` in `test_dir`.
Based on this, it computes several statistics which will be saved in individual text files in the same folder.

Use `--help` for more info on usage:
```
usage: python -m grasp_eval [-h] [-DR DATASET_ROOT] [-TD TEST_DIR] [--nms] [--no_sim] [--no_cov] [-OMD OBJECT_MODELS_DIR] [--stats_only] [--plot_res PLOT_RES]

optional arguments:
  -h, --help            show this help message and exit
  -DR DATASET_ROOT, --dataset_root DATASET_ROOT
                        root directory of the dataset
  -TD TEST_DIR, --test_dir TEST_DIR
                        test folder of the network (containing epoch dirs)
  --nms                 use this option to use "nms_poses_view0.txt" file instead of all predictions (npz files)
  --no_sim              use this option to skip simulation results (only rule-based evaluation)
  --no_cov              use this argument to skip computing coverage (shorten computation time). However, if coverage is present in eval data it will be presented in stats in any case.
  -OMD OBJECT_MODELS_DIR, --object_models_dir OBJECT_MODELS_DIR
                        please provide path to urdf files of objects (only required if using simulator)
  --stats_only
  --plot_res PLOT_RES   resolution (n data points) for success rate/coverage plots
```

## Conventions

The module assumes certain directory structures, as used in L2G and GPNet for datasets as well as predictions.
Directory structure for grasp predictions shall be:

```
- test_dir/                      # evaluation files will be put in this directory
    - epoch{e}/
        - nms_poses_view{v}.txt  # file with grasp predictions after NMS (GPNet)
        - view{v}/
            -  {s}.npz           # file with grasp predictions for shape {s}
                                 # contains only predictions with score > 0.5
                                 # data['centers'] - position of grasp centers
                                 # data['quaternions'] - orientation as quaternion (wxyz)
                                 # data['scores'] - confidence score in [0.5, 1]  
```

The module will automatically detect available epochs and views and will evaluate all of them.
The shapes will be drawn from `test_set.csv` from the dataset folder.
Using `--nms` option will use the `nms_poses_view{v}.txt`, otherwise the .npz files for each shape are used.
The NMS files contain predictions for all shapes; one line with the object name followed by the corresponding 
grasp proposals (in descending order by score), described by 8 values:
grasp center (x, y, z), grasp orientation as quaternion (w, x, y, z), and score.
Score is optional but should be provided in order to allow computation of all metrics.

Note that the code from [GPNet](https://github.com/CZ-Wu/GPNet) requires small adjustments to work with the evaluator:
- comment out zMove [here](https://github.com/CZ-Wu/GPNet/blob/master/test.py#L188)
- (optional) add score value [here](https://github.com/CZ-Wu/GPNet/blob/master/test.py#L207)

## Acknowledgments

This work was conducted within the BURG research project for Benchmarking and Understanding Robotic Grasping.
It is supported by CHIST-ERA and EPSRC grant no. EP/S032487/1.