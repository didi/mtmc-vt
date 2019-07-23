# AI City Challenge 2019 Track1 MTMC Task
The code is for AI City Challenge 2019 Track1, MTMC Vehicle Tracking.

And we got the second place.

[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Li_Spatio-temporal_Consistency_and_Hierarchical_Matching_for_Multi-Target_Multi-Camera_Vehicle_Tracking_CVPRW_2019_paper.pdf)

## Maintainers

Peilun Li

Guozhen Li

Meiqi Lu

Zhangxi Yan

Youzeng Li

## Preparation

Dateset download: [Track1-download,Size: 16.2GB](https://www.aicitychallenge.org/2019-data-sets/)

For running code correctly, the data should be put as follows:

```
├─ aic19-track1-mtmc
│  ├─ train
│  │  ├─ S01
│  │  │  ├─ c001
│  │  │  │  ├─ det
│  │  │  │  ├─ gt
│  │  │  │  ├─ mtsc
│  │  │  │  ├─ segm
│  │  │  │  ├─ calibration.txt
│  │  │  │  ├─ roi.jpg
│  │  │  │  ├─ det_reid_features.txt
│  │  │  │  ├─ vdo.avi
│  │  │  ├─ c002
│  │  │  ├─ c003
│  │  │  ├─ c004
│  │  │  ├─ c005
│  │  ├─ S03
│  │  ├─ S04
│  ├─ test
│  │  ├─ S02
│  │  ├─ S05
│  └─ cam_timestamp
```

Note that the `det_reid_features.txt` is the middle result of `1b_merge_visual_feature_with_
other_feature.py`, and the other files are provided by organisers. 


## Step by Step for MTMC Vehicle Tracking


### running code orderly 


#### 1_crop_vehicle_img_from_vdo.py

For each bounding box, crop the vehicle image and calculate the gps, according to the results of detection.

input:
- input_dir:`./aic19-track1-mtmc/train`or`./aic19-track1-mtmc/test`

output:
- for each video, produce `det_gps_feature.txt` to save gps information
- for each video, save all cropped image.

#### 1a_extract_visual_feature_for_each_img.py

- extract reid feature for each corpped image, the train and inference pipeline follows reid-baseline


#### 1b_merge_visual_feature_with_other_feature.py

Merge reid feature and gps information into one file.

input:
- input_dir: `./aic19-track1-mtmc/train`or`./aic19-track1-mtmc/test`
- gps information file `/det_gps_feature.txt`, from `1`
- ReID feature file `/deep_features.txt`, from `1a`

output:
- for each video, produce `det_reid_features.txt` file


#### 2_tracking.py

multi targets tracking for each video.

input:
- input_dir: `./aic19-track1-mtmc/train`or`./aic19-track1-mtmc/test`
- ID file `already_used_number.txt`, avoid reusing number
- `/det_reid_features.txt` from `1b`

output:
- for each video, produce tracking result file `det_reid_track.txt`


#### 2a_post_process_for_tracking.py

Optimize tracking result to solve target lost.

input:
- input_dir: `./aic19-track1-mtmc/train`or`./aic19-track1-mtmc/test`
- fps file `train_fps.txt`
- for each video, need `det_reid_track.txt` from `2`

output:
- for each video, produce tracking result `optimized_track.txt`


#### 2b_remove_overlap_boxes.py

Remove overlapped bounding box.

input:
- input_dir: `./aic19-track1-mtmc/train`or`./aic19-track1-mtmc/test`
- for each video, need `optimized_track.txt` from `2a`

output:
- for each video, produce tracking result `optimized_track_no_overlapped.txt`


#### 3a_track_based_reid.py

Calculate reid similarity between tracks.

input:
- input_dir: `./aic19-track1-mtmc/train`or`./aic19-track1-mtmc/test`
- for each video, need `optimized_track_no_overlapped.txt` from `2b`

output:
- ReID similarity file `ranked`


#### 3b_trajectory_processing.py

Calculate the gps-trajectory cohesion between tracks, should run the code `trajectory_processing/main.py`

input:
- input_dir: `./aic19-track1-mtmc/train`or`./aic19-track1-mtmc/test`
- for each video, need `optimized_track_no_overlapped.txt` from `2b`

output:
- gps-trajectroy file `gps_and_time_new`

#### 4a_match_tracks_for_crossroad.py

MTMC tracking for crossroad scene

input:
- input_dir: `./aic19-track1-mtmc/train`or`./aic19-track1-mtmc/test`
- for each video, need `optimized_track_no_overlapped.txt` from `2b`
- `ranked` from `3a`
- `gps_and_time_new` from `3b`

output:
- match result `submission_crossroad_train`

#### 4b_match_tracks_for_arterialroad.py

MTMC tracking for arterial road scene

input:
- input_dir: `./aic19-track1-mtmc/train`or`./aic19-track1-mtmc/test`
- for each video, need `optimized_track_no_overlapped.txt` from `2b`
- `ranked` from `3a`
- `gps_and_time_new` from `3b`

output:
- match result `submission_normal_train`

#### 5a_merge_results.py

merge the results from different scenes

input:
- `submission_crossroad_train` from `4a`
- `submission_normal_train` from `4b`

output:
- merged result file `submission`

#### 5b_adapt_boxes.py

post process for each bounding box

input:
- input_dir: `./aic19-track1-mtmc/train`or`./aic19-track1-mtmc/test`
- `submission` from `5a`

output:
- result file `submission_adpt`

#### 5c_convert_to_submission.py

convert the result to submission format

input:
- `submission_adpt` from `5b`

output:
- submission file `track1.txt`


### Guide for use
Run the code from `1_\*.py` to `5c_\*.py` orderly.
The train and inference for ReID follows reid-baseline

We propose starting with `2_tracking.py`, if you are the first time to this project. And we provide the results of `1b`. 
You could download it, put them in the right place as metioned above and rename them as `det_reid_features.txt`.

## Extras
result of `1b`
