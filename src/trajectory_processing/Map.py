# -*- coding: utf-8 -*-
import numpy as np
# import matplotlib.pyplot as plt

from Track import Track
from Point import Point


# def check_distribution(data_list):
#     distribution_list = [0] * 30
#     for item in data_list:
#         #if speed
#         times = int(item / 10)
#         if times > 29:
#             continue
#
#         distribution_list[times] += 1
#
#     plt.bar(range(len(distribution_list)), distribution_list, color='rgb')
#
#     plt.show()
class Map(object):

    def __init__(self,tracks_dict):
        self.tracks_dict = tracks_dict
        #self.anchor = Point() #todo 每个场景的平均值点（用来消除离群点）

    def add_point(self, point):
        self.tracks_dict[point.id].add_point(point)

    def add_track(self, track):
        self.tracks_dict[track.id] = track

    def print_track(self):
        for key,value in self.tracks_dict.items():
            value.print_sequence()

    def get_length(self):
        return len(self.tracks_dict.keys())

    def statistics(self):
        total_track_number = len(self.tracks_dict.keys())
        total_point_number = 0
        track_length_list = []
        gps_x_list =[]
        gps_y_list =[]
        speed_list = []
        for id, track in self.tracks_dict.items():
            total_point_number += len(track.sequence)
            track_length_list.append(len(track.sequence))
            speed_list = speed_list + track.speed_list
            for point in track.sequence:
                gps_x_list.append(point.x)
                gps_y_list.append(point.y)

        print('total track number', total_track_number)
        print('total points number', total_point_number)
        print('max', max(track_length_list))
        print('min', min(track_length_list))
        print('mean', np.mean(track_length_list))
        print('median', np.median(track_length_list))
        counts = np.bincount(track_length_list)
        print('mode', np.argmax(counts))
        print('*'*30)
        print('max_x', max(gps_x_list))
        print('min_x', min(gps_x_list))
        print('mean_x', np.mean(gps_x_list))
        print('max_y', max(gps_y_list))
        print('min_y', min(gps_y_list))
        print('mean_y', np.mean(gps_y_list))
        return max(gps_x_list)
        '''
        print('*'*30)
        print('max_speed',max(speed_list))
        print('min_speed',min(speed_list))
        print('mean_speed',np.mean(speed_list))
        print('median',np.median(speed_list))
        counts = np.bincount(speed_list)
        print('mode', np.argmax(counts))
        #check_distribution(speed_list)'''






    def visualize_track_by_id(self, id):
        track = self.tracks_dict[id]
        #track.print_sequence()
        track.visualization()

    def visualization(self):
        pass#todo

    def sort_every_track_by_time(self):
        pass

    def sort_every_track_by_length(self):
        pass

    def tracks_matching_direction(self):
        #speed_match_dict = {} #id:{}-> id_x:{id_y: yes or no}
        #time_match_dict = {}
        cos_match_dict={}
        id_list = list(self.tracks_dict.keys())
        for i in range(0,len(id_list) - 1):
            id_1 = id_list[i]

            track_1 = self.tracks_dict[id_1]
            for j in range(i+1, len(id_list)):
                id_2 = id_list[j]
                if id_1 == id_2:
                    continue
                track_2 = self.tracks_dict[id_2]

                if id_1 not in cos_match_dict:
                    cos_match_dict[id_1] = [{id_2:track_1.angle_with_another_track(track_2)}]#angle_with_another_track_coarse
                else:
                    cos_match_dict[id_1].append({id_2:track_1.angle_with_another_track(track_2)})
        return cos_match_dict

    def tracks_matching_direction_distance(self):
        match_dict = {}
        id_list = list(self.tracks_dict.keys())
        for i in range(0, len(id_list) - 1):
            id_1 = id_list[i]

            track_1 = self.tracks_dict[id_1]
            for j in range(i + 1, len(id_list)):
                id_2 = id_list[j]
                if id_1 == id_2:
                    continue
                track_2 = self.tracks_dict[id_2]

                if id_1 not in match_dict:
                    match_dict[id_1] = [{id_2: track_1.angle_and_mindis_with_another_track(track_2)}]  # angle_with_another_track_coarse
                else:
                    match_dict[id_1].append({id_2: track_1.angle_and_mindis_with_another_track(track_2)})
        return match_dict

    def tracks_matching_direction_distance_time(self):
        match_dict = {}
        id_list = list(self.tracks_dict.keys())
        for i in range(0, len(id_list) - 1):
            id_1 = id_list[i]

            track_1 = self.tracks_dict[id_1]
            for j in range(i + 1, len(id_list)):
                id_2 = id_list[j]
                if id_1 == id_2:
                    continue
                track_2 = self.tracks_dict[id_2]

                if id_1 not in match_dict:
                    match_dict[id_1] = [{id_2: track_1.angle_and_mindis_with_another_track(track_2)
                                         +track_1.time_to_another_track(track_2)}]  # angle_with_another_track_coarse
                else:
                    match_dict[id_1].append({id_2: track_1.angle_and_mindis_with_another_track(track_2)
                                             +track_1.time_to_another_track(track_2)})
        return match_dict


    def tracks_matching_prediction(self):
        time_prediction_dict = {}# id_x:{id_y: t} x轨迹到达y的时间（>=0)
        id_list = list(self.tracks_dict.keys())
        for i in range(0, len(id_list)):
            id_1 = id_list[i]

            track_1 = self.tracks_dict[id_1]
            for j in range(0, len(id_list)):
                id_2 = id_list[j]
                if id_1 == id_2:
                    continue
                track_2 = self.tracks_dict[id_2]

                if id_1 not in time_prediction_dict:
                    time_prediction_dict[id_1] = [{id_2:track_1.time_to_another_track(track_2)}]
                else:
                    time_prediction_dict[id_1].append({id_2:track_1.time_to_another_track(track_2)})
        return time_prediction_dict


    def tracks_matching_final(self):

        match_dict = {}
        id_list = list(self.tracks_dict.keys())
        for i in range(0, len(id_list)):
            id_1 = id_list[i]

            track_1 = self.tracks_dict[id_1]
            for j in range(0, len(id_list)):
                id_2 = id_list[j]

                track_2 = self.tracks_dict[id_2]
                # if len(track_2.sequence) == 0:
                #     continue

                combine_id = id_1 + '_' + id_2
                reverse_combine_id = id_2 + '_' + id_1
                if combine_id not in match_dict and reverse_combine_id not in match_dict:
                    match_dict[combine_id] = track_1.angle_and_mindis_with_another_track(track_2) + track_1.time_to_another_track(track_2)
                elif combine_id not in match_dict and reverse_combine_id in match_dict:
                    match_dict[combine_id] = match_dict[reverse_combine_id][:4] + track_1.time_to_another_track(track_2)

        return match_dict