# -*- coding: utf-8 -*-
#所有阈值和常量放在这里

TRACK_FILE_DIR= '../data/track_results/scene_2'#scene_3
GROUND_TRUTH_TRACK_DIR = '../data/GT/scene_2'

SMOOTH_DIS = 1e-6 # 每一个点与之前点进行比较，如果经纬度有一个坐标差值小于此阈值，抹掉这个点
OUTLIER_DIS = 0.007 # 每一个点与平均值点进行比较，如果经纬度有一个坐标差值大于此阈值，抹掉这个点0.007差不多777米

SLICE_THRESHOLD = 10 #对track进行切片的长度
DIRECTION_THRESHOLD = 10


COS_DIS_SAME_DIRECTION = 0.93
COS_DIS_OPP_DIRECTION = -0.5 #120度


MIN_DIS_THRESHOLD = 10 #

SPEED_THRESHOLD = 80 # 用于删除离群点