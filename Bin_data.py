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


"""Maps CM variables to the parameter number for easy use"""
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


"""Returns the Loop number if data for that loop number is in range of the cut"""
def Cut(lst, cuts, i, j):
    if j != None:
        if lst[j] >= cuts[i] and lst[j] <= cuts[i+1]:
            return j


"""Slices data into 2 bins which contain the index number only.
   Does this for 2 bins only (could enlargen)"""
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


"""Index matches the bins to the data in question"""
def IndexToData(indices, values):
    data = []
    for i in range(len(indices)):
        d = []
        for j in range(len(indices[i])):
            d.append(values[indices[i][j]])
        data.append(d)

    return data


"""Algorithm which splits data such that the number of items in a bin is evenly split.
   Hence will adjust the bin limits to satisfy the conditions."""
def Bin_algorithm(parameter, values, incriments):
    cut_lims = np.linspace(min(parameter), max(parameter), incriments) # creates a list of bin limits to try
    cuts = []
    for m in range(len(cut_lims)):
        for n in range(len(cut_lims)):
            cut_min = cut_lims[m]
            cut_max = cut_lims[n]
            """if statement avoids double counting and zero width bins"""
            if cut_min != cut_max and cut_max > cut_min:
                #print('trying: ' + str(cut_min) + ', ' + str(cut_max))
                cut = np.linspace(cut_min, cut_max, 3)  # cuts the data at the midpoint
                out = CutData(values, parameter, cut)  # cuts the data and returns the indicies of values
                cuts.append(out)  # stores the cut for comparison check
    keep = [[0], [0]]
    for i in range(len(cuts)):
        if(i == 0):
            keep = cuts[i]  # first cut tried added
        else:
            lens = [len(x) for x in cuts[i]]  # gets the number of items in the bin
            """firsts check if data lost is minimal (roughly 2sigma tolerance)"""
            if(lens[0] + lens[1] > 0.995 * len(values)):
                """Then checks if the item numbers are larger than the previous bins which were accepted"""
                if lens[0] > len(keep[0]) and lens[1] > len(keep[1]):
                    keep = cuts[i]
    return keep


"""Required if the COM has an extremely narrow resonance so binning is quick"""
def ResonanceCut(parameter, divisionNumber=10):
    bins, edges = np.histogram(m34, bins=divisionNumber)  # creates a histogram of the bin
    cut = edges[np.argmax(bins)]  # gets the largest frequency COM
    return [min(parameter), cut, max(parameter)]  # returns the cut


"""Will bin data into 32 indivdual bins formed from 5 parameter spaces. Bins are adjusted to fit the larget amount possible (for the given conditions)"""
def BinData(order, iterations, CutResonance, initialData):
    binsToSlice = initialData  # data which needs to be divided
    for i in range(5):
        param = GetParameter(order[i])  # gets parameter from a unique numbering system. See GetParameter for definitions
        """This is the first cut, so we don't maximise the density, otherwise we do."""
        if i == 0:
            """Only use for paramter space which have sharp resonaces i.e. one value swamps the distribution"""
            if CutResonance is True:
                cut = ResonanceCut(param, 1000)
            else:
                cut = np.linspace(min(param), max(param), 3)  # cut data at the midpoint
            bin_index = CutData(binsToSlice, param, cut)  # returns indices for data that needs to be in the respective bin.
            binsToSlice = IndexToData(bin_index, binsToSlice)  # assigns value of data by index macthing
            print(i, len(binsToSlice[0]), len(binsToSlice[1]))  # prints an example of the number of elements in each bin
        else:
            bin_temp = []
            """Loop through every bin in binsToSlice"""
            for dat in binsToSlice:
                out = Bin_algorithm(param, dat, iterations[i])  # applys the binning algorithm to match the bin densitys
                bin_temp.append(IndexToData(out, dat))  # index matches bin indicies to binsToSlice and stores the bin
            binsToSlice = []
            """Use this to un-nest the bins in the list"""
            for k in range(len(bin_temp)):
                for j in range(len(bin_temp[k])):
                    binsToSlice.append(bin_temp[k][j])
            print(i, len(bin_temp[k][0]), len(bin_temp[k][1]))  # prints an example of the number of elements in each bin

    print('amount of data lost: ' + str(np.sum([len(x) for x in binsToSlice])))  # informs user of the amount of events lost.
    return binsToSlice


"""Generates 5 of the 34 possible parameters to calculate the LIPS. splits data into parity even/odd bins"""
def DalitzParameters(particles):
    #C_Tl, C_Tu = kin.Segment_TP(particles)  # splits data into parity odd-even

    # Helicity angles
    cos_p_1 = kin.HelicityAngle(particles['p_1'] + particles['p_2'], particles['p_0'], particles['p_1'])

    cos_p_3 = kin.HelicityAngle(particles['p_3'] + particles['p_4'], particles['p_0'], particles['p_3'])

    # Invariant masses
    m_12 = kin.Mag_4(particles['p_1'] + particles['p_2'])

    m_34 = kin.Mag_4(particles['p_3'] + particles['p_4'])

    # Decay plane angle
    phi = kin.Decay_Plane_Angle(particles['p_0'], particles['p_1'], particles['p_2'], particles['p_3'], particles['p_4'])

    return [cos_p_1, cos_p_3, m_12, m_34, phi]


"""Used to generate DalitzParameters for datasets too large to be computed at once."""
def MultiSampleDalitzParameters(particles):
    data = dm.SplitEvents(particles, 10)
    parameters = []
    progress = 0
    for d in data:
        progress += 1
        print(progress/len(data) * 100)
        params = DalitzParameters(d)
        parameters.append(params)

    new_list = []
    for i in range(5):
        subset = []
        for j in range(len(parameters)):
            subset.append(parameters[j][i])
        new_list.append(subset)
    return new_list


"""Main Body"""
datas = dm.GenerateDataFrames('\Phase-0', False)
datas_CP = dm.GenerateDataFrames('\Phase-0_CP', True)

p = datas[9*10:(9+1)*10]
p = dm.MergeData(p)
pbar = datas_CP[9*10:(9+1)*10]
pbar = dm.MergeData(pbar)

particles = {'p_0': p[0], 'p_1': p[1], 'p_2': p[2], 'p_3': p[3], 'p_4': p[4]}

#m12 = kin.Mag_4(p[1] + p[2])
#m34 = kin.Mag_4(p[3] + p[4])
#hel_1 = kin.HelicityAngle(p[1] + p[2], p[0], p[1])
#hel_3 = kin.HelicityAngle(p[3] + p[4], p[0], p[3])
#phi = kin.Decay_Plane_Angle(*p)

hel_1, hel_3, m12, m34, phi = MultiSampleDalitzParameters(particles)



order = [2, 3, 5, 1, 4]
iterations = [1, 10, 2, 10, 30]


C_T = kin.Scalar_TP(kin.Vector_3(p[3]), kin.Vector_3(p[4]), kin.Vector_3(p[1]))
C_Tbar = -kin.Scalar_TP(kin.Vector_3(pbar[3]), kin.Vector_3(pbar[4]), kin.Vector_3(pbar[1]))

print('values needed are calculated.')

bins = BinData(order, iterations, True, C_T)
bins_CP = BinData(order, iterations, True, C_Tbar)

A_Ts = []
for i in range(len(bins)):
    A_T = kin.TP_Amplitude(bins[i])
    A_Ts.append(A_T)


A_Tbars = []
for i in range(len(bins)):
    A_Tbar = kin.TP_Amplitude(bins[i])
    A_Tbars.append(A_Tbar)





