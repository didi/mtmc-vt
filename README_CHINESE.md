# AI City Challenge 2019 Track1 MTMC Task
AI City Challenge 2019 赛道1的代码, 用于做跨摄像头车辆跟踪。

我们最终取得了第二名的成绩。

[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/AI%20City/Li_Spatio-temporal_Consistency_and_Hierarchical_Matching_for_Multi-Target_Multi-Camera_Vehicle_Tracking_CVPRW_2019_paper.pdf)

## Maintainers

李佩伦,李国镇,卢美奇,严章熙,李友增
滴滴出行-车载事业部

## Preparation

下载数据的网站: [Track1-download,Size: 16.2GB](https://www.aicitychallenge.org/2019-data-sets/)

为了正确运行代码，数据存放的层级结构需与官方保持一致，如下：

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

注意，其中det_reid_features.txt并非官方提供，
是代码main_pipeline/1b_merge_visual_feature_with_
other_feature.py执行后的生成的。


## Step by Step for MTMC Vehicle Tracking


### 顺序运行代码


#### 1_crop_vehicle_img_from_vdo.py

根据每个视频的detection结果，裁剪车辆图像，并计算其gps坐标。

input:
- 存放数据的文件夹`./aic19-track1-mtmc/train`或`./aic19-track1-mtmc/test`

output:
- 对每个视频，都会生成一个`det_gps_feature.txt`文件，用于存放gps信息
- 同时所有crop得到的车辆图像，都存放在文件夹里`/aic19-track1-mtmc/adjust_c_cropped_imgs`


#### 1a_extract_visual_feature_for_each_img.py

用ReID模型，对文件夹`/aic19-track1-mtmc/adjust_c_cropped_imgs`内所有图像提取视觉特征。

训练和测试ReID模型的代码来自reid_baseline


#### 1b_merge_visual_feature_with_other_feature.py

合并gps特征和reid特征成一个文件

input:
- 存放数据的文件夹`./aic19-track1-mtmc/train`或`./aic19-track1-mtmc/test`
- gps信息文件`/det_gps_feature.txt`，来自1
- ReID特征文件`/deep_features.txt`， 来自1a

output:
- 对每个视频，生成一个`det_reid_features.txt`文件


#### 2_tracking.py

对每个视频做多目标跟踪

input:
- 存放数据的文件夹`./aic19-track1-mtmc/train`或`./aic19-track1-mtmc/test`
- 编号记录文件，保证编号不重复使用`already_used_number.txt`
- 对每个视频，需要`/det_reid_features.txt`文件，来自1b或下载

output:
- 对每个视频，生成跟踪结果文件`det_reid_track.txt`


#### 2a_post_process_for_tracking.py

优化跟踪结果，主要用于解决目标丢失的问题

input:
- 存放数据的文件夹`./aic19-track1-mtmc/train`或`./aic19-track1-mtmc/test`
- fps文件`train_fps.txt`
- 对每个视频，需要`det_reid_track.txt`文件，来自2

output:
- 对每个视频，生成跟踪结果文件`optimized_track.txt`


#### 2b_remove_overlap_boxes.py

对重叠度较高的两个框，只保留前面的一个，这是为了适配数据集的标注标准

input:
- 存放数据的文件夹`./aic19-track1-mtmc/train`或`./aic19-track1-mtmc/test`
- 对每个视频，需要`optimized_track.txt`文件，来自2a

output:
- 对每个视频，生成跟踪结果文件`optimized_track_no_overlapped.txt`


#### 3a_track_based_reid.py

所有视频的跟踪结果放在一起，计算track间的reid距离

input:
- 存放数据的文件夹`./aic19-track1-mtmc/train`或`./aic19-track1-mtmc/test`
- 对每个视频，需要`optimized_track_no_overlapped.txt`文件，来自2b

output:
- Reid距离文件`ranked`

#### 3b_trajectory_processing.py

所有视频的跟踪结果放在一起，提取track间的gps轨迹关系，该文件作为流程图中的占位符，实际执行应该在`trajectory_processing/main.py`

input:
- 存放数据的文件夹`./aic19-track1-mtmc/train`或`./aic19-track1-mtmc/test`
- 对每个视频，需要`optimized_track_no_overlapped.txt`文件，来自2b

output:
- 轨迹关系文件`gps_and_time_new`

#### 4a_match_tracks_for_crossroad.py

在十字路口场景下进行跨相机的track匹配

input:
- 存放数据的文件夹`./aic19-track1-mtmc/train`或`./aic19-track1-mtmc/test`
- 对每个视频，需要`optimized_track_no_overlapped.txt`文件，来自2b
- `ranked`文件，来自3a
- `gps_and_time_new`文件，来自3b

output:
- 匹配结果`submission_crossroad_train`

#### 4b_match_tracks_for_arterialroad.py

在主干道路场景下进行跨相机的track匹配

input:
- 存放数据的文件夹`./aic19-track1-mtmc/train`或`./aic19-track1-mtmc/test`
- 对每个视频，需要`optimized_track_no_overlapped.txt`文件，来自2b
- `ranked`文件，来自3a
- `gps_and_time_new`文件，来自3b

output:
- 轨迹关系文件`submission_normal_train`

#### 5a_merge_results.py

合并4a和4b的匹配结果成唯一的文件

input:
- `submission_crossroad_train`文件，来自4a
- `submission_normal_train`文件，来自4b

output:
- 合并文件`submission`

#### 5b_adapt_boxes.py

后处理每个bounding box（每个框都将扩展25个像素），使其适配标注标准

input:
- 存放数据的文件夹`./aic19-track1-mtmc/train`或`./aic19-track1-mtmc/test`
- `submission`文件，来自5a

output:
- 文件`submission_adpt`

#### 5c_convert_to_submission.py

调整文件格式，使它可以正确地被测试代码识别

input:
- `submission_adpt`文件，来自5b

output:
- 文件`track1.txt`


### 使用指南
需要配置pytorch的reid推理代码：训练和推理reid的代码来自reid_baseline
按顺序运行代码，从1_\*.py到5c_\*.py

但在你第一次使用这个项目的时候，我建议从2开始运行整个流程，下面我们提供了1b的运行结果，
你需要把每个视频的中间结果放在对应的地方（按照上面提到的数据存放的层级结构），并改名为`det_reid_features.txt`

## Extras
`1b`的中间结果

