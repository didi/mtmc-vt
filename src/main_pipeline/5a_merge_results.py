# -*- coding: utf-8 -*-
import numpy as np
from numpy import random
import os
import glob
import shutil
import time


input1 = "./aic19-track1-mtmc/submission_crossroad_train"
input2 = "./aic19-track1-mtmc/submission_normal_train"
out = "./aic19-track1-mtmc/submission"


f = open(out, 'w')

lines = open(input1).readlines()
for line in lines:
    f.write(line)

lines = open(input2).readlines()
for line in lines:
    f.write(line)

f.close()