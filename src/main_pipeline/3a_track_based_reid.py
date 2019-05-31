# -*- coding: utf-8 -*-
# author: peilun
# 匹配跨视频的track
# 19
import numpy as np
import os
import math
import operator
from matplotlib import pyplot as plt

MATCHED = True
NO_MATCHED = False
input_dir = "./aic19-track1-mtmc/train"

INNER_SIMILAR_TH = 10
MOVE_TH = 4  # todo 待修订
ALL_TIME = 210
TIME_TH = 4.5
SCALE = 10.0

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

    def get_area(self):
        return self.box[2]*self.box[3]


# 只能用于单视频内的一条track
class Track(object):

    def __init__(self, id, sequence, cams):
        self.id = id
        self.sequence = sequence
        self.match_state = MATCHED
        self.cams = cams
        self.gps_move_vec = np.zeros(2)

    def get_average_feature(self):
        dim = self.sequence[0].feature.size
        ft = np.zeros(dim)
        l = len(self.sequence)
        for bx in self.sequence:
            ft = ft + bx.feature
        ft = ft*(1.0/l)
        return ft

    def get_orientation(self):
        return self.sequence[0].orientation

    def get_camera(self):
        return self.cams

    def append(self, box):
        self.sequence.append(box)
        # if box.camera not in self.cams:
        #     self.cams.append(box.camera)

    # todo 待修改
    # def update(self, tk):
    #     for bx in tk.sequence:
    #         bx.id = self.id
    #         self.append(bx)
    #
    #     subscriber_mv = tk.gps_move_vec*10000
    #     master_mv = self.gps_move_vec*10000
    #     subscriber_dis = subscriber_mv[0] ** 2 + subscriber_mv[1] ** 2
    #     master_dis = master_mv[0] ** 2 + master_mv[1] ** 2
    #     if subscriber_dis > master_dis:
    #         print 'move vec changed: ', master_mv, ' to ', subscriber_mv
    #         self.gps_move_vec = tk.gps_move_vec

    def get_last(self):
        return self.sequence[-1]

    def get_first(self):
        return self.sequence[0]

    def get_last_feature(self):
        return self.sequence[-1].feature

    def get_first_feature(self):
        return self.sequence[0].feature

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
        # if self.sequence[-1].time < self.sequence[0].time:
        #     print [self.sequence[0].time, self.sequence[-1].time]

        return [self.sequence[0].time, self.sequence[-1].time]


def calu_feature_distance(ft0, ft1):
    feature_dis_vec = ft1 - ft0
    feature_dis = np.dot(feature_dis_vec.T, feature_dis_vec)
    return feature_dis


# 计算两个track之间的特征距离，选最小的特征差距作为距离
def calu_track_feature_distance(tk0, tk1):
    ft_list0 = tk0.feature_list
    ft_list1 = tk1.feature_list

    min_dis = 99999999
    mean_dis = 0.0
    for ft0 in ft_list0:
        for ft1 in ft_list1:
            dis = calu_feature_distance(ft0, ft1)
            mean_dis += dis
            if dis < min_dis:
                min_dis = dis
    # mean_dis = mean_dis / (len(ft_list0) * len(ft_list1))
    # avergae_ft_dis = calu_feature_distance(tk0.average_feature, tk1.average_feature) * SCALE
    return min_dis


# 计算两个track之间的特征距离，选最小的特征差距作为距离，加入图像大小
def calu_track_feature_distance_new(tk0, tk1):

    bx_list0 = tk0.sequence
    bx_list1 = tk1.sequence

    min_dis = 99999999

    for bx0 in bx_list0:
        for bx1 in bx_list1:
            ft0 = bx0.feature
            ft1 = bx1.feature
            dis = calu_feature_distance(ft0, ft1)

            scale = 1
            if bx0.get_area() < 10000:
                scale *= 1.5
            if bx1.get_area() < 10000:
                scale *= 1.5

            # print scale, dis
            dis = dis*scale

            if dis < min_dis:
                min_dis = dis

    return min_dis


# @pysnooper.snoop()
def analysis_to_track_dict(file_path):
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
        cur_box = Box(camera, index, id, box, score, (gps_x, gps_y), orientation, time, ft)
        if id not in track_dict:
            track_dict[id] = Track(id, [], camera)
        track_dict[id].append(cur_box)
        track_dict[id].gps_move_vec = track_dict[id].get_moving_vector()  # 第一次初始化的是box加入的方式，需要手动设置移动方向，后面每次update的时候会更新移动方向到最新的
        cmpfun = operator.attrgetter('time')  # 参数为排序依据的属性，可以有多个，这里优先id，使用时按需求改换参数即可
        track_dict[id].sequence.sort(key=cmpfun)
    return track_dict


# 参数依次为list,抬头,X轴标签,Y轴标签,XY轴的范围
def draw_hist(myList, Title, Xlabel, Ylabel, Xmin, Xmax, Ymin, Ymax):
    plt.hist(myList, 100)
    plt.xlabel(Xlabel)
    plt.xlim(Xmin, Xmax)
    plt.ylabel(Ylabel)
    plt.ylim(Ymin, Ymax)
    plt.title(Title)
    plt.show()


def main():

    scene_dirs = []
    scene_fds = os.listdir(input_dir)
    for scene_fd in scene_fds:
        scene_dirs.append(os.path.join(input_dir, scene_fd))
    for scene_dir in scene_dirs:
        # if scene_dir != './aic19-track1-mtmc/test/S02':
        #     continue
        # 每个场景应该保留一个整体的距离排名文件
        ranked_file = os.path.join(scene_dir, 'ranked')
        camera_dirs = []
        track_list = []
        fds = os.listdir(scene_dir)
        for fd in fds:
            if fd.startswith('c0'):
                camera_dirs.append(os.path.join(scene_dir, fd))
        for camera_dir in camera_dirs:
            print(camera_dir)
            track_file_path = os.path.join(camera_dir, 'optimized_track_no_overlapped.txt')
            tk_dict = analysis_to_track_dict(track_file_path)
            for k in tk_dict:
                track_list.append(tk_dict[k])

        # 构建id的顺序字典，并计算平均特征
        print("calu average feature.")
        id_order_dict = {}
        l = len(track_list)
        for i in range(l):
            tk = track_list[i]
            id_order_dict[tk.id] = i
            tk.average_feature = tk.get_average_feature()
            tk.feature_list = tk.get_feature_list()





        # 构建特征距离矩阵
        print("calu distence mat")
        distence_mat = np.zeros((l, l))
        for i in range(l):
            print(i)
            cur_tk = track_list[i]
            for j in range(l):
                cpr_tk = track_list[j]
                dis = calu_feature_distance(cur_tk.average_feature, cpr_tk.average_feature)
                distence_mat[i][j] = dis

        # 展示同track内的距离
        inner_total_dis = []
        for tk in track_list:
            boxes = tk.sequence
            for i in range(1, len(boxes)):
                inner_dis = calu_feature_distance(boxes[i].feature, boxes[i - 1].feature)
                # print "inner dis: ", inner_dis
                inner_total_dis.append(inner_dis)
        # draw_hist(inner_total_dis, 'inner dis', 'dis', 'number', 0.0, 400, 0.0, 60000)  # 直方图展示

        # 展示不同track间的距离
        outer_total_dis = []
        for i in range(len(track_list)):
            tk0 = track_list[i]
            for j in range(len(track_list)):
                if i == j:
                    continue
                tk1 = track_list[j]
                outer_dis = calu_feature_distance(tk0.sequence[0].feature, tk1.sequence[0].feature)
                # print "outer dis: ", outer_dis
                outer_total_dis.append(outer_dis)
        # draw_hist(outer_total_dis, 'outer dis', 'dis', 'number', 0.0, 400, 0.0, 400000)

        # if True:
        #     continue


        # 构建id的顺序字典，并计算平均特征
        print("calu average feature.")
        id_order_dict = {}
        l = len(track_list)
        for i in range(l):
            tk = track_list[i]
            id_order_dict[tk.id] = i
            tk.average_feature = tk.get_average_feature()
            tk.feature_list = tk.get_feature_list()

        # 保存ranking结果
        f = open(ranked_file, 'w')
        for cur_tk in track_list:
            print(track_list.index(cur_tk))
            dis_dict = {}
            for cpr_tk in track_list:
                if cur_tk.cams == cpr_tk.cams:
                    dis = 9999.0
                else:
                    dis = calu_track_feature_distance(cur_tk, cpr_tk) * SCALE
                    # dis = calu_feature_distance(cur_tk.average_feature, cpr_tk.average_feature)*SCALE
                dis_dict[cpr_tk.id] = dis

            dis_dict[cur_tk.id] = 0.0
            ranked_list = sorted(dis_dict.items(), key=lambda x: x[1])

            ww = str(cur_tk.id)
            for i in range(1, len(ranked_list)):
                item = ranked_list[i]
                ww += ' ' + str(item[0]) + '_' + str(int(item[1]))
            ww += '\n'
            f.write(ww)
        f.close()


if __name__ == '__main__':
    main()


