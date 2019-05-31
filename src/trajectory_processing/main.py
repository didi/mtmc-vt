# -*- coding: utf-8 -*-
import os
import numpy as np
import sys
from configs import TRACK_FILE_DIR
from Point import Point
from Track import Track
from Map import Map


input_dir = "../aic19-track1-mtmc/train"


def analysis_to_track_dict(file_path,track_map):
    file_name = file_path.split('/')[-1]
    camera_id = file_name.replace('_train.txt', '')
    camera_id = camera_id.replace('c', '')

    lines = open(file_path, 'r').readlines()
    for line in lines:
        words = line.strip('\n').split(',')
        index = words[0]
        #id = camera_id + '_' +words[1]
        id = words[1]
        gps = words[7].split('-')
        gps_x = float(gps[0])
        gps_y = float(gps[2])
        time = float(words[9])
        point = Point(gps_x, gps_y,time, id, camera_id)

        if id not in track_map.tracks_dict:
            track = Track(id,[])
            track.add_point(point)
            track_map.add_track(track)
        else:
            track_map.add_point(point)
    return len(lines)


def track_matching(file_path_list, output_file):
    track_map = Map({})
    for file_path in file_path_list:
         analysis_to_track_dict(file_path,track_map)
    print(len(track_map.tracks_dict.keys()))
    for id,track in track_map.tracks_dict.items():
        track.sort_by_time_and_smooth()

    print(len(track_map.tracks_dict.keys()))
    matching_dict = track_map.tracks_matching_final()
    print(len(track_map.tracks_dict.keys()))
    print(len(matching_dict.keys()))

    with open(output_file, 'w') as f:
        for combine_id, min_max_mean_mindis_time in matching_dict.items():
            ids = combine_id.split('_')
            id_1 = ids[0]
            id_2 = ids[1]
            if len(min_max_mean_mindis_time) != 5:
                continue
            if len(track_map.tracks_dict[id_1].sequence) != 0:
                id_1_start = track_map.tracks_dict[id_1].sequence[0].t
                id_1_end = track_map.tracks_dict[id_1].sequence[-1].t
            else:
                id_1_start = 0.
                id_1_end = 0.
            mincos = min_max_mean_mindis_time[0]
            maxcos = min_max_mean_mindis_time[1]
            meancos = min_max_mean_mindis_time[2]
            mindis = min_max_mean_mindis_time[3]
            time = min_max_mean_mindis_time[4]
            f.write(id_1+','+str(id_1_start)+','+str(id_1_end) + ',' + id_2 + ',' + str(mincos)+','+str(maxcos)+','+str(meancos)+','+str(mindis)+','+str(time)+'\n')
    f.close()


def main():
    scene_dirs = []
    scene_fds = os.listdir(input_dir)
    # global TKGP_ID_COUNT
    # f = open(out_path, 'w')

    for scene_fd in scene_fds:
        scene_dirs.append(os.path.join(input_dir, scene_fd))
    for scene_dir in scene_dirs:
        tracked_path_list = []
        camera_dirs = []
        fds = os.listdir(scene_dir)
        out_path = os.path.join(scene_dir, 'gps_and_time_new')
        for fd in fds:
            if fd.startswith('c0'):
                camera_dirs.append(os.path.join(scene_dir, fd))
        for camera_dir in camera_dirs:
            print(camera_dir)
            tracked_path = os.path.join(camera_dir, 'optimized_track_no_overlapped.txt')
            tracked_path_list.append(tracked_path)

        track_matching(tracked_path_list, out_path)


if __name__ == '__main__':
    main()
    # file_list = os.listdir('../data/track_results_old/scene_2')
    # file_path_list = []
    # for file in file_list:
    #     file_path_list.append(os.path.join('../data/track_results_old/scene_2', file))
    # track_matching(file_path_list, 'track_similarity.txt')
