# ReID Model Introduction

We follow the github repo:https://github.com/L1aoXingyu/reid_baseline, to get a strong ReID baseline.

We denote this folder as `$ReID_MODEL`

## Prequisite

 - pytorch 1.0
 - torchvision
 - ignite
 - yacs

## Training

1. Open `$ReID_MODEL/data/datasets/mtmc.py`, for line 9 variable `trn_dir`, set `trn_dir` to where training data located, this directory should contains many sub folders, each sub folder contains images extracted from a video.
2. Run `sh train_t2_softmax_triple.sh`, the training process will start. Trained model will be saved at `ReID_MODEL/export_dir/aic_track2/softmax_triplet`

## Test

1. Open `$ReID_MODEL/data/datasets/aic_track2.py`, for line 120 variable `test_dir`, set `test_dir` to where test images located, this directory should only contains JPG image.
2. For given trained model, run `sh inference.sh`, ReID feature will be generated in `feature.txt` in `$ReID_MODEL`, each line starts with the image name followed by feature of length 2048 seperated with blank characters.
