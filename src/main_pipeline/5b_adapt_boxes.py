# -*- coding: utf-8 -*-
# author: peilun
# 适当放大box，以适应标注box
# 14
"""
"""
import numpy as np
import os
import cv2
import glob
import shutil
import time

input_dir = "./aic19-track1-mtmc/train"
sub_path = "./aic19-track1-mtmc/submission"
res_path = "./aic19-track1-mtmc/submission_adpt"
th = 25


class Box(object):
    """
    match_state:一个box是否匹配到一个track中，若没有，应该生成新的track
    """

    def __init__(self, camera, id, frame_index, box):
        self.camera = camera
        self.frame_index = frame_index
        self.id = id
        self.box = box


def adaptation_box(box, resolution):
    h, w = resolution

    if box[0] + box[2]>w-1 or box[1] + box[3]>h-1 or box[0]<0 or box[1]<0:
        print('too big: ', resolution, box)

    p0x = int(max(0, box[0]-th))
    p0y = int(max(0, box[1]-th))
    p1x = int(min(box[0] + box[2] + th, w-1))
    p1y = int(min(box[1] + box[3] + th, h-1))
    return [p0x, p0y, p1x-p0x, p1y-p0y]


# 解析每个视频的大小
vdo_size_dict = {}
scene_dirs = []
scene_fds = os.listdir(input_dir)
for scene_fd in scene_fds:
    scene_dirs.append(os.path.join(input_dir, scene_fd))
for scene_dir in scene_dirs:
    camera_dirs = []
    fds = os.listdir(scene_dir)
    for fd in fds:
        if fd.startswith('c0'):
            camera_dirs.append(os.path.join(scene_dir, fd))
    for camera_dir in camera_dirs:
        vdo_path = os.path.join(camera_dir, 'vdo.avi')
        cap = cv2.VideoCapture(vdo_path)
        w = cap.get(3)
        h = cap.get(4)
        camera = camera_dir.split('/')[-1]
        vdo_size_dict[camera] = (h, w)
for k in vdo_size_dict:
    print(k, vdo_size_dict[k])

# 载入不同camera下的box
camera_dict = {}
lines = open(sub_path).readlines()
for line in lines:
    words = line.strip('\n').split(',')
    camera = words[0]
    id = words[1]
    frame_index = words[2]
    box_ori = [int(words[3]), int(words[4]), int(words[5]), int(words[6])]
    # print camera
    box = adaptation_box(box_ori, vdo_size_dict[camera])
    bx = Box(camera, id, frame_index, box)
    # print camera, id, frame_index, box
    if bx.camera not in camera_dict:
        camera_dict[bx.camera] = []
    camera_dict[bx.camera].append(bx)

f = open(res_path, 'w')
for cam_k in camera_dict:
    bx_list = camera_dict[cam_k]
    for bx in bx_list:
        bbox = bx.box
        ww = cam_k + ',' + bx.id + ',' + bx.frame_index + ',' + str(bbox[0]) + ',' + str(bbox[1])+ ',' + str(bbox[2])+ ',' + str(bbox[3]) + ',-1,-1\n'
        f.write(ww)
f.close()

