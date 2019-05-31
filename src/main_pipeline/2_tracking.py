# -*- coding: utf-8 -*-
# 15
"""
"""
import numpy as np
import os
import cv2
from scipy.optimize import linear_sum_assignment

input_dir = "../aic19-track1-mtmc/train"
ard_uesd_num_path = "../already_used_number.txt"

IMAGE_SIZE = 224
TH_SCORE = 0.3
PAD_SIZE = 10
IOU_TH = 0.7
W_PAD = 0
H_PAD = 0

DISTENCE_TH = 100
WIGHTS = 0.05
MATCHED = True
NO_MATCHED = False


def creat_no_used_number():
    f = open(ard_uesd_num_path, 'a+')
    lines = f.readlines()
    max_num = int(lines[-1].strip('\n'))
    ww = str(max_num+1)+'\n'
    f.write(ww)
    f.close()
    return max_num+1


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


class Box(object):
    """
    match_state:一个box是否匹配到一个track中，若没有，应该生成新的track
    """
    def __init__(self, frame_index, id, box, score, gps_coor, feature):
        self.frame_index = frame_index
        self.id = id
        self.box = box
        self.score = score
        self.gps_coor = gps_coor
        self.feature = feature
        self.center = (self.box[0] + self.box[2] / 2, self.box[1] + self.box[3] / 2)
        self.match_state = NO_MATCHED


class Track(object):

    def __init__(self, id, sequence):
        self.id = id
        self.sequence = sequence
        self.match_state = MATCHED
        # self.leak_time = int(0)  # 镂空次数，如果连续两帧

    def append(self, box):
        self.sequence.append(box)

    def get_last(self):
        return self.sequence[-1]

    def get_last_feature(self):
        return self.sequence[-1].feature

    def get_last_gps(self):
        return self.sequence[-1].gps_coor

    def show(self):
        print("For track-" + str(self.id) + ' : ', "length-" + len(self.sequence), ", matchState-", self.match_state)


class Frame(object):
    def __init__(self, index, boxes):
        self.index = index
        self.boxes = boxes

    def append(self, box):
        self.boxes.append(box)

    def show(self):
        print("For frame index-" + str(self.index) + ' : ', "length-" + len(self.boxes))
        for i in range(len(self.boxes)):
            box = self.boxes[i]
            print('box', i, ': ', box)


def Hungary(task_matrix):
    b = task_matrix.copy()
    # 行和列减0
    for i in range(len(b)):
        row_min = np.min(b[i])
        for j in range(len(b[i])):
            b[i][j] -= row_min
    for i in range(len(b[0])):
        col_min = np.min(b[:, i])
        for j in range(len(b)):
            b[j][i] -= col_min
    line_count = 0
    # 线数目小于矩阵长度时，进行循环
    while (line_count < len(b)):
        line_count = 0
        row_zero_count = []
        col_zero_count = []
        for i in range(len(b)):
            row_zero_count.append(np.sum(b[i] == 0))
        for i in range(len(b[0])):
            col_zero_count.append((np.sum(b[:, i] == 0)))
        # 划线的顺序（分行或列）
        line_order = []
        row_or_col = []
        for i in range(len(b[0]), 0, -1):
            while (i in row_zero_count):
                line_order.append(row_zero_count.index(i))
                row_or_col.append(0)
                row_zero_count[row_zero_count.index(i)] = 0
            while (i in col_zero_count):
                line_order.append(col_zero_count.index(i))
                row_or_col.append(1)
                col_zero_count[col_zero_count.index(i)] = 0
        # 画线覆盖0，并得到行减最小值，列加最小值后的矩阵
        delete_count_of_row = []
        delete_count_of_rol = []
        row_and_col = [i for i in range(len(b))]
        for i in range(len(line_order)):
            if row_or_col[i] == 0:
                delete_count_of_row.append(line_order[i])
            else:
                delete_count_of_rol.append(line_order[i])
            c = np.delete(b, delete_count_of_row, axis=0)
            c = np.delete(c, delete_count_of_rol, axis=1)
            line_count = len(delete_count_of_row) + len(delete_count_of_rol)
            # 线数目等于矩阵长度时，跳出
            if line_count == len(b):
                break
            # 判断是否画线覆盖所有0，若覆盖，进行加减操作
            if 0 not in c:
                row_sub = list(set(row_and_col) - set(delete_count_of_row))
                min_value = np.min(c)
                for i in row_sub:
                    b[i] = b[i] - min_value
                for i in delete_count_of_rol:
                    b[:, i] = b[:, i] + min_value
                break
    row_ind, col_ind = linear_sum_assignment(b)
    min_cost = task_matrix[row_ind, col_ind].sum()
    best_solution = list(task_matrix[row_ind, col_ind])
    return best_solution, col_ind


# dict:key-帧序列数,value-Box_list
def analysis_to_frame_dict(file_path):
    frame_dict = {}
    lines = open(file_path, 'r').readlines()
    for line in lines:
        words = line.strip('\n').split(',')
        # print 'what: ', words[0], len(words)
        index = int(words[0])
        id = int(words[1])
        box = [int(float(words[2])), int(float(words[3])), int(float(words[4])), int(float(words[5]))]
        score = float(words[6])
        gps_x = float(words[8])
        gps_y = float(words[9])
        ft = np.zeros(len(words) - 10)
        for i in range(10, len(words)):
            ft[i - 10] = float(words[i])
        cur_box = Box(index, id, box, score, (gps_x, gps_y), ft)
        if index not in frame_dict:
            frame_dict[index] = Frame(index, [])
        frame_dict[index].append(cur_box)
    return frame_dict


def get_num_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    return int(cap.get(7))


def process_a_video(camera_dir):
    det_reid_ft_path = camera_dir + "/det_reid_features.txt"
    result_path = camera_dir + "/det_reid_track.txt"
    roi_path = camera_dir + '/roi.jpg'
    video_path = camera_dir + '/vdo.avi'
    all_frames = get_num_frames(video_path)
    roi_src = cv2.imread(roi_path)

    frame_dict = analysis_to_frame_dict(det_reid_ft_path)  # dict:key-帧序列数,value-Frame
    result_dict = {}  # 记录最后结果的字典：key-id,value-track
    flowing_track = []  # 记录当前正在跟踪的tracks
    print(all_frames)
    all_frames_num = len(frame_dict)
    count = 0
    for k in range(1, all_frames+1):
        cur_frame = frame_dict.get(k)
        if cur_frame == None:
            continue
        # print k, '**************************************************'
        # print len(cur_frame.boxes)
        processed_boxes = preprocess_boxes(cur_frame.boxes, roi_src)
        cur_frame.boxes = processed_boxes

        # print len(cur_frame.boxes)
        # 当前帧的所有box试图与现有的track匹配
        track_features = []
        # 如果一个track，它和当前帧所有的box都不相似，那么它应该被移除,不参与后面的匹配
        delete_tks = []
        for tk in flowing_track:
            tk_ft = tk.get_last_feature()
            tk_gps = tk.get_last_gps()
            no_matched_flag = True
            for box in cur_frame.boxes:
                # 计算特征差
                box_ft = box.feature
                feature_dis_vec = box_ft - tk_ft
                feature_dis = np.dot(feature_dis_vec.T, feature_dis_vec)

                # 计算gps差
                box_gps = box.gps_coor
                gps_dis_vec = ((tk_gps[0]-box_gps[0]), (tk_gps[1]-box_gps[1]))
                gps_dis = (gps_dis_vec[0]*100000)**2 + (gps_dis_vec[1]*100000)**2
                # print feature_dis, gps_dis_vec, gps_dis
                total_dis = gps_dis*WIGHTS + feature_dis
                # if feature_dis < 50:
                #     print 'near: ', gps_dis*WIGHTS, feature_dis, total_dis
                # if feature_dis > 150:
                #     print 'far: ', gps_dis*WIGHTS, feature_dis, total_dis

                if total_dis < DISTENCE_TH:
                    no_matched_flag = False
            if no_matched_flag:
                delete_tks.append(tk)

        # print 's:', len(flowing_track)
        for tk in delete_tks:
            result_dict[tk.id] = tk
            flowing_track.remove(tk)
        # print 'e:', len(flowing_track)

        # 匈牙利算法：每个tk都是一个待匹配的任务，找到最适合的解法
        # 为匈牙利算法生成cost矩阵
        tk_num = len(flowing_track)
        bx_num = len(cur_frame.boxes)
        mat_size = max(tk_num, bx_num)
        cost_matrix = np.zeros((mat_size, mat_size))
        for i in range(bx_num):
            box = cur_frame.boxes[i]
            box_ft = box.feature
            box_gps = box.gps_coor
            for j in range(tk_num):
                tk = flowing_track[j]
                tk_ft = tk.get_last_feature()
                tk_gps = tk.get_last_gps()

                # 计算特征差
                feature_dis_vec = box_ft - tk_ft
                feature_dis = np.dot(feature_dis_vec.T, feature_dis_vec)
                # 计算gps差
                gps_dis_vec = ((tk_gps[0] - box_gps[0]), (tk_gps[1] - box_gps[1]))
                gps_dis = (gps_dis_vec[0] * 100000) ** 2 + (gps_dis_vec[1] * 100000) ** 2
                total_dis = gps_dis * WIGHTS + feature_dis
                cost_matrix[i][j] = total_dis
        if bx_num == tk_num:
            # case1:boxes数量等于track数量（人数=任务数）
            costs, col_ind = Hungary(cost_matrix)
            # print costs
            for i in range(bx_num):
                if costs[i] < DISTENCE_TH:
                    tk_index = col_ind[i]
                    box = cur_frame.boxes[i]
                    cur_frame.boxes[i].match_state = MATCHED
                    flowing_track[tk_index].append(box)
                    flowing_track[tk_index].match_state = MATCHED
        elif bx_num > tk_num:
            # case2:boxes数量大于track数量（人数 > 任务数）
            costs, col_ind = Hungary(cost_matrix)
            # print costs
            for i in range(bx_num):
                if col_ind[i] < tk_num:
                    if costs[i] < DISTENCE_TH:
                        tk_index = col_ind[i]
                        box = cur_frame.boxes[i]
                        cur_frame.boxes[i].match_state = MATCHED
                        flowing_track[tk_index].append(box)
                        flowing_track[tk_index].match_state = MATCHED
        else:
            # case3:track数量大于boxes数量（任务数大于人数，有些任务会被删掉,bx_num < tk_num）
            costs, col_ind = Hungary(cost_matrix.T)
            # print costs
            for i in range(tk_num):
                if col_ind[i] < bx_num:
                    if costs[i] < DISTENCE_TH:
                        box_index = col_ind[i]
                        cur_frame.boxes[box_index].match_state = MATCHED
                        flowing_track[i].append(cur_frame.boxes[box_index])
                        flowing_track[i].match_state = MATCHED

        # 更新Track集，未匹配的track应该被移除
        delete_tks2 = []
        for tk in flowing_track:
            if tk.match_state == NO_MATCHED:
                delete_tks2.append(tk)
        if len(delete_tks2) > 0:
            for tk in delete_tks2:
                result_dict[tk.id] = tk
                flowing_track.remove(tk)

        # 未匹配的box创建新track
        for box in cur_frame.boxes:
            if box.match_state == NO_MATCHED:
                new_id = creat_no_used_number()
                new_track = Track(new_id, [])
                box.match_state = MATCHED
                new_track.append(box)
                flowing_track.append(new_track)
        # 把所有的track变为未匹配，方便下一轮判断
        for tk in flowing_track:
            tk.match_state = NO_MATCHED

    # 把最后的track加入result
    for tk in flowing_track:
        result_dict[tk.id] = tk

    # 用于后续
    f = open(result_path, 'w')
    for k in result_dict:
        tk = result_dict[k]
        for bx in tk.sequence:
            ww = str(bx.frame_index)+','+str(tk.id)+','+str(bx.box[0])+','+str(bx.box[1])+\
                 ','+str(bx.box[2])+','+str(bx.box[3])+','+str(bx.score) + ',' + \
                 str(bx.gps_coor[0]) + '-' + str(bx.gps_coor[1]) + ',-1,-1'
            for i in bx.feature:
                ww += ',' + str(i)
            ww += '\n'
            f.write(ww)
    f.close()


def main():
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
            process_a_video(camera_dir)


if __name__ == '__main__':
    main()