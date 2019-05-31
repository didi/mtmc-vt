# -*- coding: utf-8 -*-
# author: peilun
# 匹配跨视频的track
# 13,610
import numpy as np
import os
import math
import sys


MATCHED = True
NO_MATCHED = False
input_dir = "./aic19-track1-mtmc/test"
ard_uesd_num_path = "./already_used_number.txt"
out_path = "./aic19-track1-mtmc/submission_normal_train"

INNER_SIMILAR_TH = 10
TIME_TH = 4
# TIME_TH = 15
TKGP_ID_COUNT = 40000  # 用于分配不同的id号
ORI_TH = 200
feature_dict = {}
time_dict = {}
trail_dict = {}
near_th = 0
overlap_add = 0
near_time_reach_add = 0
hard_add = 0

# 载入gps，time匹配文件
# 载入深度特征距离文件


class DisItem(object):
    def __init__(self, id, dis, rank):
        self.id = id
        self.dis = dis
        self.rank = rank


class GPSTimeItem(object):
    def __init__(self, id0, id1, mincos, maxcos, mindis, reach):
        self.id0 = id0
        self.id1 = id1
        self.mincos = mincos
        self.maxcos = maxcos
        self.mindis = mindis
        self.reach = reach


class Box(object):
    """
    match_state:一个box是否匹配到一个track中，若没有，应该生成新的track
    """
    def __init__(self, camera, frame_index, id, box, score, gps_coor, orientation, time, feature):
        self.camera = camera
        self.frame_index = frame_index
        self.id = id
        self.box = box
        self.score = score
        self.gps_coor = gps_coor
        self.orientation = orientation
        self.time = time
        self.feature = feature
        self.center = (self.box[0] + self.box[2] / 2, self.box[1] + self.box[3] / 2)
        self.match_state = NO_MATCHED


# 只能用于单视频内的一条track
class Track(object):

    def __init__(self, id, sequence, cams):
        self.id = id
        self.sequence = sequence
        self.match_state = MATCHED
        self.cams = cams
        self.gps_move_vec = np.zeros(2)
        self.tkgp_id = -1

    def get_orientation(self):
        return self.sequence[0].orientation

    def append(self, box):
        self.sequence.append(box)

    def get_last(self):
        return self.sequence[-1]

    def get_first(self):
        return self.sequence[0]

    def get_length(self):
        return len(self.sequence)

    # 跟踪轨迹结束的位置，用于判定结束位置的合理性
    def get_last_gps(self):
        return self.sequence[-1].gps_coor

    def get_first_gps(self):
        return self.sequence[0].gps_coor

    # 移动距离，用于判断一个track是否根本没移动
    def get_moving_distance(self):
        start_p = self.sequence[0].center
        end_p = self.sequence[-1].center
        move_vec = ((end_p[0] - start_p[0]), (end_p[1] - start_p[1]))
        move_dis = move_vec[0]**2 + move_vec[1]**2
        # print gps_dis
        return move_dis

    # 整体移动向量，用于判断整体移动方向
    def get_moving_vector(self):
        start_p = self.sequence[0].gps_coor
        end_p = self.sequence[-1].gps_coor
        move_vec = np.zeros(2)
        move_vec[0] = end_p[0] - start_p[0]
        move_vec[1] = end_p[1] - start_p[1]
        # print gps_dis_vec
        return move_vec

    def get_moving_gps_distance(self):
        move_vec = self.get_moving_vector()*100000
        move_gps_dis = move_vec[0] ** 2 + move_vec[1] ** 2
        return math.sqrt(move_gps_dis)

    def get_moving_time(self):
        start_t = self.sequence[0].time
        end_t = self.sequence[-1].time
        return end_t - start_t

    def show(self):
        print("For track-" + str(self.id) + ' : ', "length-" + str(len(self.sequence)))
        print(self.get_moving_distance())

    def get_feature_list(self):
        ft_list = []
        for bx in self.sequence:
            cur_ft = bx.feature
            similar = False
            for ft in ft_list:
                dis = calu_feature_distance(ft, cur_ft)
                if dis < INNER_SIMILAR_TH:
                    similar = True
                    break
            if not similar:
                ft_list.append(cur_ft)
        return ft_list

    # 获取track的时间段
    def get_time_slot(self):
        return [self.sequence[0].time, self.sequence[-1].time]


class TrackGroups(object):

    def __init__(self, tkpg_id):
        self.tkpg_id = tkpg_id
        self.track_list = []
        self.cam_list = []
        self.feature_list = []
        self.time_list = []
        self.orientations = ''
        self.id_list = []

    def get_length(self):
        return len(self.track_list)

    def get_start_time(self):
        start_time = 999999
        for time_slot in self.time_list:
            start_time = min(start_time, time_slot[0])
        return start_time

    # 添加一个tk
    def append_tk(self, tk):
        self.track_list.append(tk)
        self.id_list.append(tk.id)
        self.cam_list.append(tk.cams)
        ft_list = tk.get_feature_list()
        self.feature_list = self.feature_list + ft_list
        time_slot = tk.get_time_slot()
        self.time_list.append(time_slot)
        # 添加方向
        cur_ori = tk.get_orientation()
        for i in cur_ori:
            if i not in self.orientations:
                self.orientations = self.orientations + i

    # 合并一个TrackGroup
    def merge(self, tk_gp):
        self.track_list = self.track_list + tk_gp.track_list
        self.id_list = self.id_list + tk_gp.id_list
        for i in tk_gp.cam_list:
            self.cam_list.append(i)
        self.feature_list = self.feature_list + tk_gp.feature_list
        self.time_list = self.time_list + tk_gp.time_list
        cur_ori = tk_gp.orientations
        for i in cur_ori:
            if i not in self.orientations:
                self.orientations = self.orientations + i


def calu_feature_distance(ft0, ft1):
    feature_dis_vec = ft1 - ft0
    feature_dis = np.dot(feature_dis_vec.T, feature_dis_vec)
    return feature_dis


def analysis_to_track_dict(file_path):
    # tag_id_count = 0  # 每条原始track加一个specificity_id，作为区分的标签，节省计算量
    camera = file_path.split('/')[-2]
    track_dict = {}
    lines = open(file_path, 'r').readlines()
    for line in lines:
        words = line.strip('\n').split(',')
        index = int(words[0])
        id = int(words[1])
        box = [int(float(words[2])), int(float(words[3])), int(float(words[4])), int(float(words[5]))]
        score = float(words[6])
        gps = words[7].split('-')
        # print gps
        gps_x = float(gps[0])
        gps_y = float(gps[2])
        orientation = words[8]
        time = float(words[9])
        ft = np.zeros(len(words) - 10)
        for i in range(10, len(words)):
            ft[i - 10] = float(words[i])
        if id not in track_dict:
            track_dict[id] = Track(id, [], camera)

        cur_box = Box(camera, index, id, box, score, (gps_x, gps_y), orientation, time, ft)
        track_dict[id].append(cur_box)
        track_dict[id].gps_move_vec = track_dict[id].get_moving_vector()  # 第一次初始化的是box加入的方式，需要手动设置移动方向，后面每次update的时候会更新移动方向到最新的
    return track_dict


#  这里读到字典key都是int，而不是string
def load_deep_feature_distences(reranked_file):
    global feature_dict
    lines = open(reranked_file).readlines()
    for line in lines:
        words = line.split()
        query = int(words[0])
        qry_dict = {}
        for i in range(1, len(words)):
            word = words[i]
            ws = word.split('_')
            id = int(ws[0])
            dis = int(ws[1])
            dis_item = DisItem(id, dis, i)
            qry_dict[id] = dis_item
        feature_dict[query] = qry_dict


def load_gps_and_time(gps_and_time):
    global time_dict
    global trail_dict

    lines = open(gps_and_time).readlines()
    for line in lines:
        words = line.split(',')
        A_id = int(words[0])
        A_st = float(words[1])
        A_et = float(words[2])
        B_id = int(words[3])
        mincos = float(words[4])
        maxcos = float(words[5])
        mindis = float(words[7])
        predict_time = float(words[8])
        # 把-99变成199，方便后面处理
        if predict_time < -50:
            predict_time = 199

        time_dict[A_id] = [A_st, A_et]
        if A_id not in trail_dict:
            trail_dict[A_id] = {}
        item = GPSTimeItem(A_id, B_id, mincos, maxcos, mindis, predict_time)
        trail_dict[A_id][B_id] = item


# 摄像头是否有重合
def cameras_diff(tk_gp0, tk_gp1):
    for i in tk_gp0.cam_list:
        for j in tk_gp1.cam_list:
            if i == j:
                return False
            else:
                continue
    return True


# todo 目前只用了最大相似夹角
def tkgp_same_direction(tk_gp0, tk_gp1):
    global trail_dict
    max_similar = -1
    tk_list0 = tk_gp0.track_list
    tk_list1 = tk_gp1.track_list
    for tk0 in tk_list0:
        for tk1 in tk_list1:
            A = tk0.id
            # print 'A type: ', type(A)
            B = tk1.id
            max_similar = max(trail_dict[A][B].maxcos, max_similar)
    # return max_similar
    if max_similar > 0.7:
        return True
    else:
        return False


def tkgp_same_direction_th(tk_gp0, tk_gp1):
    global trail_dict
    max_similar = -1
    min_similar = 99
    tk_list0 = tk_gp0.track_list
    tk_list1 = tk_gp1.track_list
    for tk0 in tk_list0:
        for tk1 in tk_list1:
            A = tk0.id
            # print 'A type: ', type(A)
            B = tk1.id
            max_similar = max(trail_dict[A][B].maxcos, max_similar)
            min_similar = min(trail_dict[A][B].mincos, min_similar)

    if min_similar > 0.99:
        return 100
    if max_similar > 0.5:
        return 0
    else:
        return -700


def tkgp_time_fit(tk_gp0, tk_gp1):
    global time_dict
    max_tiou = -1
    tk_list0 = tk_gp0.track_list
    tk_list1 = tk_gp1.track_list
    for tk0 in tk_list0:
        for tk1 in tk_list1:
            A = tk0.id
            B = tk1.id
            tk0_slot = time_dict[A]
            tk1_slot = time_dict[B]
            tk0_slot_wide = [tk0_slot[0] - TIME_TH, tk0_slot[1] + TIME_TH]
            tiou = time_iou(tk0_slot_wide, tk1_slot)
            max_tiou = max(tiou, max_tiou)
    if max_tiou > 0.00001:
        return True
    else:
        return False


def time_iou(A, B):
    if B[0] > A[1] or B[1] < A[0]:
        return 0.0
    else:
        M = (min(A[0], B[0]), max(A[1], B[1]))
        I = (max(A[0], B[0]), min(A[1], B[1]))
        I_len = I[1] - I[0]
        M_len = M[1] - M[0]
        return float(I_len)/M_len + 0.001


def tkgp_feature_distance(tk_gp0, tk_gp1):
    global feature_dict
    min_dis = 999999
    tk_list0 = tk_gp0.track_list
    tk_list1 = tk_gp1.track_list
    for tk0 in tk_list0:
        for tk1 in tk_list1:
            A = tk0.id
            B = tk1.id
            # print feature_dict[A][B].dis
            min_dis = min(min_dis, feature_dict[A][B].dis)
    return min_dis


# 计算两个track之间的特征距离，选最小的特征差距作为距离
def calu_track_feature_distance_new(tk_gp0, tk_gp1):
    global feature_dict
    global time_dict
    global trail_dict

    tag_list0 = tk_gp0.tag_id_list
    tag_list1 = tk_gp1.tag_id_list
    min_dis = 99999999
    for i in tag_list0:
        for j in tag_list1:
            key = str(i)+'-'+str(j)
            dis = feature_dict[key]
            if dis < min_dis:
                min_dis = dis
    return min_dis


def tkgp_reach(tk_gp0, tk_gp1):
    global trail_dict
    tag_list0 = tk_gp0.id_list
    tag_list1 = tk_gp1.id_list
    min_value = 199
    for i in tag_list0:
        for j in tag_list1:
            # value=0or99都是可信的，但若是-99，需要翻转判断一次
            value = min(trail_dict[i][j].reach, trail_dict[j][i].reach)
            # print value
            min_value = min(min_value, value)
    return min_value


def tkgp_gps_dis(tk_gp0, tk_gp1):
    global trail_dict
    tag_list0 = tk_gp0.id_list
    tag_list1 = tk_gp1.id_list
    min_value = 999999
    for i in tag_list0:
        for j in tag_list1:
            min_value = min(min_value, trail_dict[i][j].mindis)
    return min_value


def fuse_track_groups_pure_image(track_groups_list, near_th):
    global feature_dict
    global time_dict
    global trail_dict
    global TKGP_ID_COUNT
    result_list = []
    for cur_tk_gp in track_groups_list:
        # 与每条res库中的tk匹配
        creat_new_tk_gp = True
        for res_tk_gp in result_list:
            dis = tkgp_feature_distance(res_tk_gp, cur_tk_gp)
            if dis < near_th:
                res_tk_gp.merge(cur_tk_gp)
                creat_new_tk_gp = False
                break
        # 一条都未匹配上，形成新tk，加入res
        if creat_new_tk_gp:
            # print 'create new TrackGroup'
            TKGP_ID_COUNT += 1
            new_tk_gp = TrackGroups(TKGP_ID_COUNT)
            new_tk_gp.merge(cur_tk_gp)
            result_list.append(new_tk_gp)
    return result_list


# 根据朝向设置远近阈值
def fuse_track_groups_strict(track_groups_list, near_th):
    global feature_dict
    global time_dict
    global trail_dict
    global TKGP_ID_COUNT
    result_list = []
    for cur_tk_gp in track_groups_list:
        # 与每条res库中的tk匹配
        creat_new_tk_gp = True
        for res_tk_gp in result_list:

            reach = tkgp_reach(res_tk_gp, cur_tk_gp)
            similar_th = near_th

            if cameras_diff(res_tk_gp, cur_tk_gp) and reach < 150:
                dis = tkgp_feature_distance(res_tk_gp, cur_tk_gp)
                if dis < similar_th:
                    res_tk_gp.merge(cur_tk_gp)
                    creat_new_tk_gp = False
                    break
        # 一条都未匹配上，形成新tk，加入res
        if creat_new_tk_gp:
            # print 'create new TrackGroup'
            TKGP_ID_COUNT += 1
            new_tk_gp = TrackGroups(TKGP_ID_COUNT)
            new_tk_gp.merge(cur_tk_gp)
            result_list.append(new_tk_gp)
    return result_list


# 根据朝向设置远近阈值
def fuse_track_groups(track_groups_list, near_th):
    global overlap_add
    global near_time_reach_add
    global hard_add
    global feature_dict
    global time_dict
    global trail_dict
    global TKGP_ID_COUNT
    result_list = []
    for cur_tk_gp in track_groups_list:
        # 与每条res库中的tk匹配
        creat_new_tk_gp = True
        for res_tk_gp in result_list:

            ori_fit = face_orientation_fit(res_tk_gp, cur_tk_gp)
            time_ft = tkgp_time_fit(res_tk_gp, cur_tk_gp)
            reach = tkgp_reach(res_tk_gp, cur_tk_gp)
            min_dis = tkgp_gps_dis(res_tk_gp, cur_tk_gp)

            similar_th = near_th
            if time_ft and min_dis < 100:
                similar_th += near_time_reach_add
            if time_ft and min_dis < 10:
                similar_th += overlap_add

            if cameras_diff(res_tk_gp, cur_tk_gp) and reach < 150:
                dis = tkgp_feature_distance(res_tk_gp, cur_tk_gp)
                if dis < similar_th:
                    res_tk_gp.merge(cur_tk_gp)
                    creat_new_tk_gp = False
                    break
        # 一条都未匹配上，形成新tk，加入res
        if creat_new_tk_gp:
            # print 'create new TrackGroup'
            TKGP_ID_COUNT += 1
            new_tk_gp = TrackGroups(TKGP_ID_COUNT)
            new_tk_gp.merge(cur_tk_gp)
            result_list.append(new_tk_gp)
    return result_list


# 检漏最后还未匹配上的track
def fuse_track_groups_for_hard(track_groups_list, near_th):
    global overlap_add
    global near_time_reach_add
    global hard_add
    global feature_dict
    global time_dict
    global trail_dict
    global TKGP_ID_COUNT

    result_list = []
    for cur_tk_gp in track_groups_list:
        if cur_tk_gp.get_length() != 1:
            TKGP_ID_COUNT += 1
            new_tk_gp = TrackGroups(TKGP_ID_COUNT)
            new_tk_gp.merge(cur_tk_gp)
            result_list.append(new_tk_gp)

    for cur_tk_gp in track_groups_list:
        # 与每条res库中的tk匹配
        if cur_tk_gp.get_length() == 1:
            creat_new_tk_gp = True
            for res_tk_gp in result_list:

                ori_fit = face_orientation_fit(res_tk_gp, cur_tk_gp)
                time_ft = tkgp_time_fit(res_tk_gp, cur_tk_gp)
                reach = tkgp_reach(res_tk_gp, cur_tk_gp)
                min_dis = tkgp_gps_dis(res_tk_gp, cur_tk_gp)

                similar_th = near_th
                if time_ft and min_dis < 100:
                    similar_th += near_time_reach_add
                if time_ft and min_dis < 10:
                    similar_th += overlap_add

                # if cameras_diff(res_tk_gp, cur_tk_gp) and tkgp_same_direction(res_tk_gp, cur_tk_gp) and reach<150:
                if cameras_diff(res_tk_gp, cur_tk_gp) and reach < 150:
                    dis = tkgp_feature_distance(res_tk_gp, cur_tk_gp)
                    # if similar_th < 1100:
                    #     print dis, similar_th, res_tk_gp.orientations, cur_tk_gp.orientations
                    if dis < similar_th:
                        res_tk_gp.merge(cur_tk_gp)
                        creat_new_tk_gp = False
                        break
            # 一条都未匹配上，形成新tk，加入res
            if creat_new_tk_gp:
                # print 'create new TrackGroup'
                TKGP_ID_COUNT += 1
                new_tk_gp = TrackGroups(TKGP_ID_COUNT)
                new_tk_gp.merge(cur_tk_gp)
                result_list.append(new_tk_gp)
    return result_list


class MatchPair(object):
    global time_dict

    def __init__(self, tk0, tk1):
        self.cam0 = tk0.cams
        self.cam1 = tk1.cams
        self.id0 = tk0.id
        self.id1 = tk1.id
        self.t0 = time_dict[tk0.id]
        self.t1 = time_dict[tk1.id]

    def show(self):
        print(self.cam0 + '-' + self.cam1, self.t0, self.t1)

    def show_time_diff(self):
        print(self.t0[0], self.t1[0]-self.t0[0], self.cam0 + '-' + self.cam1, self.t0, self.t1)


def analysis_res_list(res_list):
    global feature_dict
    global time_dict
    global trail_dict

    cam_pair_dict = {}
    for tkgp in res_list:
        tk_list = tkgp.track_list
        for tk0 in tk_list:
            for tk1 in tk_list:
                id0 = tk0.id
                id1 = tk1.id
                if trail_dict[id0][id1].reach < 150:
                    cam0 = tk0.cams
                    cam1 = tk1.cams
                    key = cam0 + '-' + cam1
                    if key not in cam_pair_dict:
                        cam_pair_dict[key] = []
                    mp = MatchPair(tk0, tk1)
                    cam_pair_dict[key].append(mp)

    for k in cam_pair_dict:
        mp_list = cam_pair_dict[k]
        for mp in mp_list:
            mp.show_time_diff()


def face_orientation_fit(tk_gp0, tk_gp1):
    ori0 = tk_gp0.orientations
    ori1 = tk_gp1.orientations
    ## print ori0, ori1
    # 全等应该更相似
    # if ori0 == ori1:
    #     return 0
    # 部分相等有加分
    for i in ori0:
        for j in ori1:
            if i == j:
                # print ori0, ori1, 'True'
                return True
    # print ori0, ori1, 'False'
    return False


def main():
    global near_th
    global overlap_add
    global near_time_reach_add
    global hard_add
    global TKGP_ID_COUNT

    scene_dirs = []
    scene_fds = os.listdir(input_dir)
    f = open(out_path, 'w')

    for scene_fd in scene_fds:
        scene_dirs.append(os.path.join(input_dir, scene_fd))
    for scene_dir in scene_dirs:
        if scene_dir == './aic19-track1-mtmc/train/S01':
            continue
        # if scene_dir == './aic19-track1-mtmc/test/S02':
        #     continue

        track_groups_list = []
        camera_dirs = []
        fds = os.listdir(scene_dir)
        reranked_file = os.path.join(scene_dir, 'ranked')
        load_deep_feature_distences(reranked_file)
        gps_and_time = os.path.join(scene_dir, 'gps_and_time_new')
        load_gps_and_time(gps_and_time)

        for fd in fds:
            if fd.startswith('c0'):
                camera_dirs.append(os.path.join(scene_dir, fd))
        for camera_dir in camera_dirs:
            print(camera_dir)
            track_file_path = os.path.join(camera_dir, 'optimized_track_no_overlapped.txt')
            tk_dict = analysis_to_track_dict(track_file_path)
            for k in tk_dict:
                TKGP_ID_COUNT += 1
                cur_group = TrackGroups(TKGP_ID_COUNT)
                cur_group.append_tk(tk_dict[k])
                # cur_group.show()
                track_groups_list.append(cur_group)
        # print len(track_groups_list)

        # 按时间排序
        track_groups_list = sorted(track_groups_list, key=lambda tk_gp: tk_gp.get_start_time())

        # 合并track groups
        # ttthhh = 675 290 530
        print(len(track_groups_list))
        res_list0 = fuse_track_groups_strict(track_groups_list, near_th)
        print(len(res_list0))
        res_list1 = fuse_track_groups_strict(res_list0, near_th)
        print(len(res_list1))
        res_list2 = fuse_track_groups(res_list1, near_th)
        print(len(res_list2))
        res_list3 = fuse_track_groups(res_list2, near_th)
        print(len(res_list3))
        res_list = fuse_track_groups_for_hard(res_list3, near_th + hard_add)
        print(len(res_list))

        count = 0
        for tk_gp in res_list:
            l = tk_gp.get_length()
            if l == 1:
                count += 1
        print('1 zhanbi: ', count, '/', len(res_list))

        for tk_gp in res_list:
            for tk in tk_gp.track_list:
                for bx in tk.sequence:
                    ww = bx.camera + ',' + str(tk_gp.tkpg_id) + ',' + str(bx.frame_index) + ',' + str(bx.box[0]) + ',' + str(
                        bx.box[1]) + \
                         ',' + str(bx.box[2]) + ',' + str(bx.box[3]) + ',-1,-1\n'
                    f.write(ww)
    f.close()


if __name__ == '__main__':
    # near_th = int(sys.argv[1])
    # overlap_add = int(sys.argv[2])
    # near_time_reach_add = int(sys.argv[3])
    # hard_add = int(sys.argv[4])

    near_th = 300
    overlap_add = 0
    near_time_reach_add = 140
    hard_add = 80
    main()