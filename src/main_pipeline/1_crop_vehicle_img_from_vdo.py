# -*- coding: utf-8 -*-
# author: peilun
# 提取图像，用于pytorch提取特征
# 车辆处于图像中间位置，detection框分数大于一定阈值，并且与其他框iou不大于0.7
import numpy as np
import os
import cv2

input_dir = "../aic19-track1-mtmc/train"
out_dir = "../aic19-track1-mtmc/adjust_c_cropped_imgs"

IMAGE_SIZE = 224
TH_SCORE = 0.5
PAD_SIZE = 10
IOU_TH = 0.7
W_PAD = 0
H_PAD = 0


def analysis_transfrom_mat(cali_path):
    first_line = open(cali_path).readlines()[0].strip('\r\n')
    cols = first_line.lstrip('Homography matrix: ').split(';')
    transfrom_mat = np.ones((3, 3))
    for i in range(3):
        values_string = cols[i].split()
        for j in range(3):
            value = float(values_string[j])
            transfrom_mat[i][j] = value
    inv_transfrom_mat = np.linalg.inv(transfrom_mat)
    return inv_transfrom_mat


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
        return 0.0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return float(intersect) / (sum_area - intersect)


def preprocess_roi_erode(roi):
    tc = cv2.copyMakeBorder(roi, PAD_SIZE, PAD_SIZE, PAD_SIZE, PAD_SIZE, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    kernel = np.ones((200, 300), np.uint8)
    tc = cv2.erode(tc, kernel, iterations=1)
    h, w, _ = tc.shape
    dst = tc[PAD_SIZE:h - PAD_SIZE, PAD_SIZE:w - PAD_SIZE]
    return dst


def preprocess_roi(roi):
    width_erode = 200
    height_erode = 200
    h, w, _ = roi.shape
    left = roi[:, 0:width_erode, :]
    right = roi[:, w-width_erode:w, :]
    top = roi[0:height_erode, :, :]
    bottom = roi[h-height_erode:h, :, :]

    left = left*0
    right = right*0
    top = top*0
    bottom = bottom*0

    return roi


class GtBox(object):
    def __init__(self, id, box, score):
        self.id = id
        self.box = box
        self.score = score
        self.center = (self.box[0] + self.box[2]/2, self.box[1] + self.box[3]/2)


# dict:key-帧序列数,value-GtBoxs
def analysis_to_frame_dict(file_path):
    frame_dict = {}
    lines = open(file_path, 'r').readlines()
    for line in lines:
        words = line.strip('\n').split(',')
        index = int(words[0])
        id = int(words[1])
        box = [int(float(words[2])), int(float(words[3])), int(float(words[4])), int(float(words[5]))]
        score = float(words[6])
        cur_gt_box = GtBox(id, box, score)
        if index not in frame_dict:
            frame_dict[index] = []
        frame_dict[index].append(cur_gt_box)
    return frame_dict


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


def get_num_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    return int(cap.get(7))


def _crop(img):
    height, width, _ = img.shape
    shorter = min(height, width)
    height_start = height - shorter
    width_start = width - shorter
    cropped_img = img[height_start:height, width_start:width]
    return cropped_img


def _center_crop(img):
    height, width, _ = img.shape
    shorter = min(height, width)
    height_start = int((height - shorter)/2)
    width_start = int((width - shorter)/2)
    cropped_img = img[height_start:height - height_start, width_start:width - width_start]
    return cropped_img


# 车辆处于图像中间位置，detection框分数大于一定阈值，并且与其他框iou不大于0.7
def preprocess_boxes(src_boxes, roi):
    boxes = []
    h, w, _ = roi.shape
    for src_b in src_boxes:
        x, y = src_b.center
        score = src_b.score
        if x > W_PAD and x < w-W_PAD and y > W_PAD and y < h-H_PAD and score > TH_SCORE:
            intersection = False
            for b in boxes:
                iou = compute_iou(src_b.box, b.box)
                if iou > IOU_TH:
                    intersection = True
            if not intersection:
                boxes.append(src_b)
    return boxes


def main():
    IMAGE_COUNT = 0

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
            print(camera_dir)

            video_path = camera_dir + '/vdo.avi'
            det_path = camera_dir + '/det/det_mask_rcnn.txt'
            out_path = camera_dir + '/det_gps_feature.txt'
            roi_path = camera_dir + '/roi.jpg'
            cali_path = camera_dir + '/calibration.txt'

            frame_dict = analysis_to_frame_dict(det_path)
            out_f = open(out_path, 'w')

            trans_mat = analysis_transfrom_mat(cali_path)
            roi_src = cv2.imread(roi_path)
            # roi = preprocess_roi(roi_src)
            cap = cv2.VideoCapture(video_path)
            all_frames = get_num_frames(video_path)

            for i in range(0, all_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break
                # print(i)

                tmp_i = i + 1  # gt 的标注与实际差了一帧
                if tmp_i in frame_dict:
                    src_boxes = frame_dict[tmp_i]  # 包含太多小目标和重复检测，需要去除
                    boxes = preprocess_boxes(src_boxes, roi_src)

                    for det_box in boxes:
                        if det_box.box[2] < 50 or det_box.box[3] < 50:
                            continue

                        box = det_box.box
                        score = det_box.score

                        # GET GPS coor
                        coor = det_box.center
                        image_coor = [coor[0], coor[1], 1]
                        GPS_coor = np.dot(trans_mat, image_coor)
                        GPS_coor = GPS_coor / GPS_coor[2]

                        cropped_img = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
                        img_name = str(IMAGE_COUNT).zfill(10) + '.jpg'
                        IMAGE_COUNT += 1
                        out_path = os.path.join(out_dir, img_name)
                        # cv2.imwrite(out_path, cropped_img)

                        ww = img_name + ',' + str(tmp_i) + ',-1,' + str(box[0]) + ',' + str(box[1]) + ',' + \
                             str(box[2]) + ',' + str(box[3]) + ',' + str(score) + ',-1,' + str(GPS_coor[0]) + ',' + str(
                            GPS_coor[1])
                        ww += '\n'
                        out_f.write(ww)

            out_f.close()


if __name__ == '__main__':
    main()
