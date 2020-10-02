#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Sep 30 15:11:29 2020
Example of how to use DCBC function to evaluation expected cortical parcellations

Author: Da Zhi
'''

import scipy.io as spio
from eval_DCBC import eval_DCBC
import numpy as np

returnSubjs = [2,3,4,6,8,9,10,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,30,31]
subj_name = ['s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11',
             's12','s13','s14','s15','s16','s17','s18','s19','s20','s21','s22',
             's23','s24','s25','s26','s27','s28','s29','s30','s31']

parcels = spio.loadmat("DCBC/nullModel_rotated_162_L.mat")['rotation']
parcels = parcels[:,0]

T = eval_DCBC(sn=returnSubjs, subj_name=subj_name, hems='L', parcels=parcels)
spio.savemat("eval.mat", T)
