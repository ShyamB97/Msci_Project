# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:01:10 2020

@author: bhull
"""

import uproot
from math import log10, floor
from scipy.special import factorial
import numpy as np
import Kinematic as kin
import DataManager as dm
import matplotlib.pyplot as plt
import Plotter as pt
import os


def round_to(x, y):
    return round(x, -int(floor(log10(abs(y)))))


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
            val = Cut(parameter, cut, i, val)  # check if data is in range of bin widths
            if val != None:
                ind.append(val)
        out.append(ind)
    return out


"""Index matches the bins to the data in question"""
def IndexToData(indices, values):
    data = []
    """Cycles through each index bin and gets the value of the
    data we wanted to bin in the first place"""
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
    ranges = []

    for m in range(len(cut_lims)):
        for n in range(len(cut_lims)):
            cut_min = cut_lims[m]
            cut_max = cut_lims[n]

            """if statement avoids double counting and zero width bins"""
            if cut_min != cut_max and cut_max > cut_min:
                #print('trying: ' + str(cut_min) + ', ' + str(cut_max))
                cut = np.linspace(cut_min, cut_max, 3)  # cuts the data at the midpoint
                ranges.append(cut) # store the bin range
                out = CutData(values, parameter, cut)  # cuts the data and returns the indicies of values
                cuts.append(out)  # stores the cut for comparison check

    keep = [[0], [0]]
    for i in range(len(cuts)):
        if(i == 0):
            keep = cuts[i]  # first cut tried added
            rangesToKeep = ranges[i]  # add bin ranges to keep
        else:
            lens = [len(x) for x in cuts[i]]  # gets the number of items in the bin

            """firsts check if data lost is minimal (roughly 2sigma tolerance)"""
            if(lens[0] + lens[1] > 0.995 * len(values)):
                """Then checks if the item numbers are larger than the previous bins which were accepted"""
                if lens[0] > len(keep[0]) and lens[1] > len(keep[1]):
                    keep = cuts[i]
                    rangesToKeep = ranges[i]

    return keep, rangesToKeep


"""Required if the COM has an extremely narrow resonance so binning is quick"""
def ResonanceCut(parameter, divisionNumber=10):
    bins, edges = np.histogram(parameter, bins=divisionNumber)  # creates a histogram of the bin
    cut = edges[np.argmax(bins)]  # gets the largest frequency COM
    return [min(parameter), cut, max(parameter)]  # returns the cut


"""Will bin data into 32 indivdual bins formed from 5 parameter spaces. Bins are adjusted to fit the larget amount possible (for the given conditions)"""
def BinData(order, iterations, initialData, ResCutiter=1000, cutResonance=[False, False, False, False, False]):
    binsToSlice = initialData  # data which needs to be divided
    binLimits = []

    for i in range(5):
        param = GetParameter(order[i])  # gets parameter from a unique numbering system. See GetParameter for definitions
        """This is the first cut, so we don't maximise the density, otherwise we do."""
        if i == 0:
            """Only use for paramter space which have sharp resonaces i.e. one value swamps the distribution"""
            if cutResonance[i] is True:
                cut = ResonanceCut(param, ResCutiter)
            else:
                cut = np.linspace(min(param), max(param), 3)  # cut data at the midpoint

            bin_index = CutData(binsToSlice, param, cut)  # returns indices for data that needs to be in the respective bin.
            binsToSlice = IndexToData(bin_index, binsToSlice)  # assigns value of data by index macthing

            print(i, len(binsToSlice[0]), len(binsToSlice[1]))  # prints an example of the number of elements in each bin

            binLimits.append((cut[0], cut[1]))
            binLimits.append((cut[1], cut[2]))  # keeps the bin size for reference
        else:
            bin_temp = []
            """Loop through every bin in binsToSlice"""
            for dat in binsToSlice:
                if cutResonance[i] is False:
                    out, cut = Bin_algorithm(param, dat, iterations[i])  # applys the binning algorithm to match the bin densitys
                    bin_temp.append(IndexToData(out, dat))  # index matches bin indicies to binsToSlice and stores the bin
                    binLimits.append((cut[0], cut[1]))
                    binLimits.append((cut[1], cut[2]))  # keeps the bin size for reference
                else:
                    cut = ResonanceCut(param, ResCutiter)
                    bin_index = CutData(dat, param, cut)
                    bin_temp.append(IndexToData(bin_index, dat))
                    binLimits.append((cut[0], cut[1]))
                    binLimits.append((cut[1], cut[2]))  # keeps the bin size for reference

            binsToSlice = []
            """Use this to un-nest the bins in the list"""
            for k in range(len(bin_temp)):
                for j in range(len(bin_temp[k])):
                    binsToSlice.append(bin_temp[k][j])

            print(i, len(bin_temp[k][0]), len(bin_temp[k][1]))  # prints an example of the number of elements in each bin

    print('amount of data lost: ' + str(len(initialData) - np.sum([len(x) for x in binsToSlice])))  # informs user of the amount of events lost.
    edges = GetBinEdges(binLimits)
    return binsToSlice, edges


"""This function will create the 5D bin widths from the optimal cuts created by the bin algorithm"""
def GetBinEdges(cuts):
    # cuts contains the bin width of every iteration of the algorithm
    # so we need to group the cuts relative to the scheme iteration
    groupedCuts = []
    nest = [2, 4, 8, 16, 32]  # the number of unique bins per nested binning scheme
    end = 0
    for i in range(5):
        start = end  # previous place we ended is the start
        end = start + nest[i]  # end point is the start point plus the number of unique bins
        groupedCuts.append(cuts[start:end])

    edges = []
    """Creates the unique bin widths for limits of the 5 parameters"""
    for j in range(2):
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    for n in range(2):
                        edges.append((groupedCuts[0][j], groupedCuts[1][k], groupedCuts[2][l], groupedCuts[3][m], groupedCuts[4][n]))
    return edges


"""Generates 5 of the 34 possible parameters to calculate the LIPS. splits data into parity even/odd bins"""
def DalitzParameters(particles, CP=False):
    if CP is True:
        f = -1
    else:
        f = 1
    #C_Tl, C_Tu = kin.Segment_TP(particles)  # splits data into parity odd-even

    # Helicity angles
    cos_p_1 = kin.HelicityAngle(particles['p_1'] + particles['p_2'], particles['p_0'], particles['p_1'])

    cos_p_3 = kin.HelicityAngle(particles['p_3'] + particles['p_4'], particles['p_0'], particles['p_3'])

    # Invariant masses
    m_12 = kin.Mag_4(particles['p_1'] + particles['p_2'])

    m_34 = kin.Mag_4(particles['p_3'] + particles['p_4'])

    # Decay plane angle
    phi = kin.Decay_Plane_Angle(particles['p_0'], particles['p_1'], particles['p_2'], particles['p_3'], particles['p_4'])

    C_T = f * kin.Scalar_TP(kin.Vector_3(particles['p_3']), kin.Vector_3(particles['p_4']), kin.Vector_3(particles['p_1']))

    return [cos_p_1, cos_p_3, m_12, m_34, phi, C_T]


"""Used to generate DalitzParameters for datasets too large to be computed at once."""
def MultiSampleDalitzParameters(particles, CP=False, splitNum=100):
    data = dm.SplitEvents(particles, splitNum)  # splits events into smaller sets
    parameters = []
    progress = 0

    """Calcualte CM variables and C_T"""
    for d in data:
        progress += 1
        print(progress/len(data) * 100)
        params = DalitzParameters(d, CP)  # calulate statistics for the data set
        parameters.append(params)

    new_list = []
    """Merges each CM variable and C_T calulated for each data set"""
    for i in range(6):
        subset = []
        for j in range(len(parameters)):
            subset.append(parameters[j][i])
        new_list.append(subset)

    final_data = []
    """Puts the calculated values in a single list"""
    for i in range(6):
        final_data.append(np.concatenate(new_list[i]))

    return final_data


def SortAsymmetries():
    sortby1 = edges[0][4]
    sortby2 = edges[16][4]
    edge1 = []
    edge2 = []

    A_T_1 = []
    A_T_2 = []

    A_Tbar_1 = []
    A_Tbar_2 = []

    A_CP_1 = []
    A_CP_2 = []

    for i in range(len(edges)):
        if edges[i][4] == sortby1:
            A_T_1.append(A_Ts[i])
            A_Tbar_1.append(A_Tbars[i])
            A_CP_1.append(A_CPs[i])
            edge1.append(edges[i])
        if edges[i][4] == sortby2:
            A_T_2.append(A_Ts[i])
            A_Tbar_2.append(A_Tbars[i])
            A_CP_2.append(A_CPs[i])
            edge2.append(edges[i])


    A_T_new = A_T_1 + A_T_2
    A_Tbar_new = A_Tbar_1 + A_Tbar_2
    A_CP_new = A_CP_1 + A_CP_2


"""Main Body"""

print('loading data')
datas = dm.GenerateDataFrames('\P45_A0.75_1M', False)
datas_CP = dm.GenerateDataFrames('\P45_A0.75_1M_CP', True)

print('selecting sample')
p = datas[9*10:(9+1)*10]
p = dm.MergeData(datas)
pbar = datas_CP[9*10:(9+1)*10]
pbar = dm.MergeData(datas_CP)

p = {'p_0': p[0], 'p_1': p[1], 'p_2': p[2], 'p_3': p[3], 'p_4': p[4]}
pbar = {'p_0': pbar[0], 'p_1': pbar[1], 'p_2': pbar[2], 'p_3': pbar[3], 'p_4': pbar[4]}

#m12 = kin.Mag_4(p[1] + p[2])
#m34 = kin.Mag_4(p[3] + p[4])
#hel_1 = kin.HelicityAngle(p[1] + p[2], p[0], p[1])
#hel_3 = kin.HelicityAngle(p[3] + p[4], p[0], p[3])
#phi = kin.Decay_Plane_Angle(*p)

print("calculating CM variables and C_T/C_Tbar")
hel_1, hel_3, m12, m34, phi, C_T = MultiSampleDalitzParameters(p, False, 1000)
_, _, _, _, _, C_Tbar = MultiSampleDalitzParameters(pbar, True, 1000)


#order = [2, 3, 5, 1, 4]
#iterations = [1, 2, 2, 10, 30]
#r = [True, True, False, False, False]

order = [3, 4, 2, 1, 5]
iterations = [2, 2, 2, 8, 2]
r = [False, False, True, False, False]

print("binning data")
bins_CP, edges_CP = BinData(order, iterations, C_Tbar, 100000, r)
bins, edges = BinData(order, iterations, C_T, 100000, r)

print("calculating asymmetries")
A_Ts = []
A_Tbars = []
A_CPs = []
for i in range(len(bins)):
    A_T = kin.TP_Amplitude(bins[i])
    A_Tbar = kin.TP_Amplitude(bins_CP[i])
    A_CP = kin.A_CP(A_T, A_Tbar)
    A_Ts.append(A_T)
    A_Tbars.append(A_Tbar)
    A_CPs.append(A_CP)


bin_regions = np.linspace(1, 32, 32)


ax = plt.subplot(131)
Asym_mean = [A_Ts[i][0] for i in range(len(A_Ts))]
Asym_error = [A_Ts[i][1] for i in range(len(A_Ts))]
pt.ErrorPlot([bin_regions, Asym_mean], axis=True, x_axis='Bin Region', y_axis='Asymmetry value', y_error=Asym_error, legend=True, label='$A_{T}$')

ax = plt.subplot(132)
Asym_mean = [A_Tbars[i][0] for i in range(len(A_Tbars))]
Asym_error = [A_Tbars[i][1] for i in range(len(A_Tbars))]
pt.ErrorPlot([bin_regions, Asym_mean], axis=True, x_axis='Bin Region', y_axis='Asymmetry value', y_error=Asym_error, legend=True, label='$\\bar{A}_{T}$')

ax = plt.subplot(133)
Asym_mean = [A_CPs[i][0] for i in range(len(A_CPs))]
Asym_error = [A_CPs[i][1] for i in range(len(A_CPs))]
pt.ErrorPlot([bin_regions, Asym_mean], axis=True, x_axis='Bin Region', y_axis='Asymmetry value', y_error=Asym_error, legend=True, label='$\mathcal{A}_{CP}$')










