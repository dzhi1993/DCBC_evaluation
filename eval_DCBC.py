#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Aug 17 11:31:32 2020

Distance-Controlled Boundaries Coefficient (DCBC) evaluation
for a functional parcellation of brain cortex

INPUTS:
sn:                   The return subject number
hems:                 Hemisphere to test. 'L' - left hemisphere; 'R' - right hemisphere; 'all' - both hemispheres
binWidth:             The spatial binning width in mm, default 1 mm
maxDist:              The maximum distance for vertices pairs
parcels:              The cortical parcellation labels (integer value) to be evaluated, shape is (N,)
                      N is the number of vertices, 0 - medial wall
condType:             The condition type for evaluating
                      'unique' - evaluation will be done by using unique task conditions of the task set
                      'all' - evaluation will be done by all task conditions of the task set
taskSet:              The task set of MDTB to use for evaluating. 1 - taskset A; 2 - taskset B; [1,2] - both
resolution:           The resolution of surface space, either 32k or 164k, 32k as default
distType:             The distance metric of vertices pairs, for example Dijkstra's distance, GOD distance
                      Euclidean distance. Dijkstra's distance as default
icoRes:               Icosahedron resolution, 42, 162, 362, 642, 1002, ... default to use 2562
mwallFile:            The medial wall to be excluded from the evaluation


OUTPUT:
M:                    Gifti object- can be saved as a *.func.gii or *.label.gii file

Author: Da Zhi
'''

import os
import numpy as np
import pandas as pd
import scipy.io as spio
from scipy.sparse import find
import nibabel as nb


def eval_DCBC(sn=[2],subj_name=['s02'], hems='L', maxDist=35, binWidth=1, parcels='',
              condType='unique', taskSet=[1],resolution='32k', distType='Dijkstra',
              icoRes=162, mWallFile='icos_162'):
    taskConds = pd.read_table('DCBC/sc1_sc2_taskConds.txt', delim_whitespace=True)
    numBins = int(np.floor(maxDist / binWidth))

    if distType is 'Dijkstra':
        dist = spio.loadmat("DCBC/distAvrg_sp.mat")['avrgDs']
    elif distType is 'Sphere':
        dist = spio.loadmat("DCBC/distSphere_sp.mat")['avrgDs']
    else:
        raise TypeError("Distance type cannot be recognized!")

    # Determine which hemisphere shall be evaluated
    if hems is 'all':
        hems = ['L', 'R']
    elif hems is 'L' or 'R':
        hems = [hems]
    else:
        raise TypeError("Hemisphere type cannot be recognized!")

    # Initialization of the result buffers
    studyNum, SN, hem = [], [], []
    N, bwParcel, distmin, distmax, meanCorr, weightedCorr = [], [], [], [], [], []
    for h in hems:
        mWall = np.where(parcels == 0)[0]
        parcels = np.delete(parcels, mWall) # remove medial wall
        parcels = np.abs(parcels - parcels[:, np.newaxis])

        dist=dist.todense()
        dist = np.delete(dist, mWall, 0)
        dist = np.delete(dist, mWall, 1)
        row, col, dist = find(dist)
        sameRegion = np.zeros((dist.shape[0],), dtype=int)

        for i in range(len(row)):
            if parcels[row[i]][col[i]] == 0:
                sameRegion[i] = 1 # within-parcel
            else:
                sameRegion[i] = 2 # between-parcel

        del parcels

        for ts in taskSet:
            taskConds = taskConds[taskConds['StudyNum'] == ts]
            if condType is 'unique':  # unique conditions in taskset ts
                condIdx = taskConds['condNum'][taskConds['overlap']==0]
            elif condType is 'all':  # all conditions in taskset ts
                condIdx = taskConds['condNum']
            else:
                raise TypeError("Invalid condition type input!")

            for s in sn:
                this_wcon = nb.load("DCBC/%s/%s.%s.sc%s.con.%s.func.gii" %
                                    (subj_name[s-1],subj_name[s-1], h, ts, resolution))
                this_wcon = [x.data for x in this_wcon.darrays]
                this_wcon = np.reshape(this_wcon, (len(this_wcon), len(this_wcon[0]))).transpose()
                res = np.sqrt(this_wcon[:,-1])
                this_wcon = np.delete(this_wcon, [0, this_wcon.shape[1] - 1], axis=1) # remove instruction
                this_wcon = np.concatenate((this_wcon, np.zeros((this_wcon.shape[0], 1))), axis=1) # add rest

                for i in range(this_wcon.shape[0]): # noise normalize
                    this_wcon[i, :] = this_wcon[i, :] / res[i]

                this_wcon = np.delete(this_wcon, mWall, axis=0)
                this_wcon = this_wcon[:,condIdx-1] # take the right subset
                mean_wcon = this_wcon.mean(1)

                for i in range(this_wcon.shape[0]):
                    this_wcon[i, :] = this_wcon[i, :] - mean_wcon[i]

                this_wcon = this_wcon.astype('float32').transpose()
                K=this_wcon.shape[0]
                del res, mean_wcon

                SD = np.sqrt(np.sum(np.square(this_wcon), axis=0)/K) # standard deviation
                SD = np.reshape(SD, (SD.shape[0], 1))
                VAR = np.matmul(SD, SD.transpose())
                COV = np.matmul(this_wcon.transpose(), this_wcon) / K
                VAR = VAR[row,col]
                COV = COV[row,col]
                del SD, this_wcon
                print("\n")

                for bw in range(1,3):
                    for i in range(numBins):
                        print(".")
                        inBin = np.zeros((dist.shape[0],), dtype=int)

                        for j in range(len(inBin)):
                            if (dist[j] > i*binWidth) & (dist[j] <= (i+1)*binWidth) & (sameRegion[j] == bw):
                                inBin[j] = 1

                        # inBin = np.where(dist>i*binWidth) & (dist<=(i+1)*binWidth) & (sameRegion==bw)
                        # inBin = np.reshape(inBin, (inBin.shape[1],))
                        N = np.append(N, np.count_nonzero(inBin == 1))
                        studyNum = np.append(studyNum, ts)
                        SN = np.append(SN, s)
                        hem = np.append(hem, h)
                        bwParcel = np.append(bwParcel, bw - 1)
                        distmin = np.append(distmin, i * binWidth)
                        distmax = np.append(distmax, (i + 1) * binWidth)
                        meanCorr = np.append(meanCorr, np.nanmean(COV[inBin == 1]) / np.nanmean(VAR[inBin == 1]))
                        del inBin

                del VAR, COV
                num_w = N[bwParcel == 0]
                num_b = N[bwParcel == 1]
                weight = 1/(1/num_w + 1/num_b)
                weight = weight / np.sum(weight)
                weightedCorr = np.append(meanCorr * weight)
                print("\n")

    struct = {
        "SN": SN,
        "hem": hem,
        "studyNum": studyNum,
        "N": N,
        "bwParcel": bwParcel,
        "distmin": distmin,
        "distmax":distmax,
        "meanCorr": meanCorr,
        "weightedCorr": weightedCorr
    }
    return struct

