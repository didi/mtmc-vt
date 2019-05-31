# -*- coding: utf-8 -*-
# author: peilun
# 特征融合
# 15
import numpy as np
import os

input_dir = "../aic19-track1-mtmc/train"


def load_ft_file(feature_file):
    # load ft file
    img2deepft_dict = {}
    file = open(feature_file, 'r')
    count = 0
    while True:
        line = file.readline()
        count += 1
        if line:
            words = line.split()
            key = words[0]
            l = len(words)-1
            ft = np.zeros(l)
            img2deepft_dict[key] = ft
            for i in range(1, len(words)):
                img2deepft_dict[key][i-1] = float(words[i])
        else:
            break
    return img2deepft_dict


def load_gps_ft_file(gps_ft_file):
    img2gpsft_dict = {}
    lines = open(gps_ft_file).readlines()
    for line in lines:
        words = line.strip('\n').split(',')
        key = words[0]
        ww = ''
        for w in words[1:]:
            ww += w + ','
        img2gpsft_dict[key] = ww
    return img2gpsft_dict


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
            other_ft_file = camera_dir + '/det_gps_feature.txt'
            deep_ft_file = camera_dir + '/deep_features.txt'
            out_path = camera_dir + '/det_reid_features.txt'
            img2gpsft_dict = load_gps_ft_file(other_ft_file)

            print('loading deep feature file...')
            img2deepft_dict = load_ft_file(deep_ft_file)
            print('load done.')

            f = open(out_path, 'w')
            for key in img2gpsft_dict:
                ww = img2gpsft_dict[key]
                fts = img2deepft_dict[key]

                ww += str(fts[0])
                for i in range(1, fts.size):
                    ft = fts[i]
                    ww += ',' + str(ft)
                ww += '\n'
                f.write(ww)
            f.close()


if __name__ == '__main__':
    main()