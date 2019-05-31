# -*- coding: utf-8 -*-
# author: peilun
# 在track匹配前，移除重叠的box，靠后的会被移除
# 14
"""
"""
import numpy as np
import os

# input = "./aic19-track1-mtmc/test"
input = "./aic19-track1-mtmc/train"

iou_th = 0.1


class Box(object):
    """
    match_state:一个box是否匹配到一个track中，若没有，应该生成新的track
    """
    def __init__(self, frame_index, id,  box, fts):
        self.frame_index = frame_index
        self.id = id
        self.box = box
        self.fts = fts
        self.area = box[2]*box[3]

    def equal(self, c_box):
        bx1 = self.box
        bx2 = c_box.box
        for i in range(4):
            if bx1[i] != bx2[i]:
                return False
        return True


def process_a_box_list(box_list):
    res_list = []
    for cur_bx in box_list:
        keep = True
        for cpr_bx in box_list:
            iou = compute_iou_for_Box(cur_bx, cpr_bx)

            cur_bottom = cur_bx.box[1] + cur_bx.box[3]
            cpr_bottom = cpr_bx.box[1] + cpr_bx.box[3]

            if iou > iou_th and cur_bottom < cpr_bottom:
                print('remove one ')
                keep = False
        if keep:
            res_list.append(cur_bx)
    return res_list


def compute_iou_for_Box(Box1, Box2):
    box1 = Box1.box
    box2 = Box2.box
    return compute_iou_shadow(box1, box2)


def compute_iou(box1, box2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    rec1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    rec2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return float(intersect) / (sum_area - intersect)


# def compute_iou_shadow(box1, box2):
#     """
#     computing IoU
#     :param rec1: (y0, x0, y1, x1), which reflects
#             (top, left, bottom, right)
#     :param rec2: (y0, x0, y1, x1)
#     :return: scala value of IoU
#     """
#     rec1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
#     rec2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
#
#     # computing area of each rectangles
#     S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
#     # S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
#
#     # computing the sum_area
#     # sum_area = S_rec1 + S_rec2
#
#     # find the each edge of intersect rectangle
#     left_line = max(rec1[1], rec2[1])
#     right_line = min(rec1[3], rec2[3])
#     top_line = max(rec1[0], rec2[0])
#     bottom_line = min(rec1[2], rec2[2])
#
#     # judge if there is an intersect
#     if left_line >= right_line or top_line >= bottom_line:
#         return 0
#     else:
#         intersect = (right_line - left_line) * (bottom_line - top_line)
#         return float(intersect) / S_rec1


def compute_iou_shadow(box1, box2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    rec1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    rec2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    # sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return max(float(intersect)/S_rec1, float(intersect)/S_rec2)


# 解析每个optimized_track.txt，按帧读入，并删除同帧中重叠框中较小的一个
scene_dirs = []
scene_fds = os.listdir(input)
for scene_fd in scene_fds:
    scene_dirs.append(os.path.join(input, scene_fd))
for scene_dir in scene_dirs:
    # if scene_dir != './aic19-track1-mtmc/test/S02':
    #     continue
    camera_dirs = []
    fds = os.listdir(scene_dir)
    for fd in fds:
        if fd.startswith('c0'):
            camera_dirs.append(os.path.join(scene_dir, fd))
    for camera_dir in camera_dirs:
        print(camera_dir)
        tk_path = os.path.join(camera_dir, 'optimized_track.txt')
        out_path = os.path.join(camera_dir, 'optimized_track_no_overlapped.txt')
        frame_dict = {}

        lines = open(tk_path).readlines()
        for line in lines:
            words = line.strip('\n').split(',')
            frame_index = words[0]
            id = words[1]
            box = [int(words[2]), int(words[3]), int(words[4]), int(words[5])]
            fts = []
            l = len(words)
            for i in range(6, l):
                fts.append(words[i])

            bx = Box(frame_index, id, box, fts)
            if bx.frame_index not in frame_dict:
                frame_dict[bx.frame_index] = []
            frame_dict[bx.frame_index].append(bx)

        for frame in frame_dict:
            box_list = frame_dict[frame]
            res_list = process_a_box_list(box_list)
            frame_dict[frame] = res_list

        f = open(out_path, 'w')

        for frame in frame_dict:
            bx_list = frame_dict[frame]
            for bx in bx_list:
                bbox = bx.box
                ww = bx.frame_index + ',' + bx.id + ',' + str(bbox[0]) + ',' + str(bbox[1]) + ',' + str(bbox[2]) + ',' + str(bbox[3])
                for w in bx.fts:
                    ww += ',' + w
                ww += '\n'
                f.write(ww)
        f.close()
