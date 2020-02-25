# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:01:10 2020

@author: bhull
"""

import uproot
import numpy as np
import Kinematic as kin
import DataManager as dm
import matplotlib.pyplot as plt
import Plotter as pt
import os


def GetParameter(parameter):
    if(parameter == 1):
        return m12
    if(parameter == 2):
        return m34
    if(parameter == 3):
        return hel_1
    if(parameter == 4):
        return hel_3  
    if(parameter == 5):
        return phi

def Cut(lst, cuts, i, j):
    if j != None:
        if lst[j] >= cuts[i] and lst[j] <= cuts[i+1]:
            return j


def CutData(values, parameter, cut):
    out = []
    for i in range(2):
        ind = []
        for val in range(len(values)):
            val = Cut(parameter, cut, i, val)
            if val != None:
                ind.append(val)
        out.append(ind)
    return out


def IndexToData(indices, values):
    data = []
    for i in range(len(indices)):
        d = []
        for j in range(len(indices[i])):
            d.append(values[indices[i][j]])
        data.append(d)

    return data


def Bin_algorithm(parameter, values, incriments):
    cut_lims = np.linspace(min(parameter), max(parameter), incriments)
    cuts = []
    for m in range(len(cut_lims)):
        for n in range(len(cut_lims)):
            cut_min = cut_lims[m]
            cut_max = cut_lims[n]
            if cut_min != cut_max and cut_max > cut_min:
                #print('trying: ' + str(cut_min) + ', ' + str(cut_max))
                cut = np.linspace(cut_min, cut_max, 3)
                out = CutData(values, parameter, cut)
                cuts.append(out)
    keep = [[0], [0]]
    for i in range(len(cuts)):
        if(i == 0):
            keep = cuts[i]
        else:
            lens = [len(x) for x in cuts[i]]
            if(lens[0] + lens[1] > 0.995 * len(values)):
                if lens[0] > len(keep[0]) and lens[1] > len(keep[1]):
                    keep = cuts[i]
    return keep


p = dm.AmpGendf('B_10K_A1.root', False)

m12 = kin.Mag_4(p['p_1'] + p['p_2'])
m34 = kin.Mag_4(p['p_3'] + p['p_4'])
hel_1 = kin.HelicityAngle(p['p_1'] + p['p_2'], p['p_0'], p['p_1'])
hel_3 = kin.HelicityAngle(p['p_3'] + p['p_4'], p['p_0'], p['p_3'])
phi = kin.Decay_Plane_Angle(p['p_0'], p['p_1'], p['p_2'], p['p_3'], p['p_4'])

order = [3, 5, 4, 1, 2] #most asymmetric bins in ascening order
#order = [2, 1, 4, 5, 3]
iterations = [1, 10, 50, 100, 1000]

C_T = kin.Scalar_TP(kin.Vector_3(p['p_3']), kin.Vector_3(p['p_4']), kin.Vector_3(p['p_1']))

print('values needed are calculated.')


binsToSlice = C_T

for i in range(5):
    param = GetParameter(order[i])
    if(i == 0):
        cut = np.linspace(min(param), max(param), 3)
        bin_index = CutData(binsToSlice, param, cut)
        binsToSlice = IndexToData(bin_index, binsToSlice)
        print(i, len(binsToSlice[0]), len(binsToSlice[1]))
    else:
        bin_temp = []
        for dat in binsToSlice:
            out = Bin_algorithm(param, dat, iterations[i])
            bin_temp.append(IndexToData(out, dat))
        binsToSlice = []
        for k in range(len(bin_temp)):
            for j in range(len(bin_temp[k])):
                binsToSlice.append(bin_temp[k][j])
        print(i, len(bin_temp[k][0]), len(bin_temp[k][1]))

print(np.sum([len(x) for x in binsToSlice]))










