# -*- coding: utf-8 -*-
# author: peilun
# 转为提交格式
"""
"""

input_path = "./aic19-track1-mtmc/submission_adpt"
out_path = "./aic19-track1-mtmc/track1.txt"

f = open(out_path, 'w')
lines = open(input_path).readlines()
for line in lines:
    words = line.strip('\n').split(',')
    ww = str(int(words[0][2:]))
    for i in words[1:]:
        ww += ' ' + i
    ww += '\n'
    f.write(ww)

f.close()