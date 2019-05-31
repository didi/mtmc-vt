# -*- coding: utf-8 -*-
import numpy as np
from configs import SMOOTH_DIS
class Point(object):
    '''
    point:
        x,
        y,
        t,
        id,# 同一个id的是一条轨迹
        video_id# 所在的video

    '''

    def __init__(self, x, y, t, id, camera_id):
        self.x = x
        self.y = y
        self.t = t
        self.id = id
        self.camera_id = camera_id

    def print_point(self):
        print('x:', self.x, 'y:', self.y, 't:', self.t, 'id:', self.id, 'camera_id:', self.camera_id)
        #print(int(self.x * 1000000 - 42490000),int(self.y* 1000000 - 90670000),self.camera_id,)

    def similar_point_in_pos(self, point_2):
        det_x = abs(point_2.x - self.x)
        det_y = abs(point_2.y - self.y)
        #print(det_x,det_y)
        if det_x < SMOOTH_DIS  or det_y < SMOOTH_DIS:
            return 1
        else:
            return 0

    def dis_between_point(self, point_2):
        det_x = (self.x - point_2.x)*1e5  # 纬度
        det_y = (self.y - point_2.y)*1e5 # 经度
        return np.sqrt(np.square(det_x) + np.square(det_y))


    def t_between_point(self, point_2):
        return abs(self.t - point_2.t)

    def speed_between_point(self,point_2):
        if self.t_between_point(point_2) == 0:
            return 0.
        return self.dis_between_point(point_2)/self.t_between_point(point_2)



