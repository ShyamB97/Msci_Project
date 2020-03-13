# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:01:10 2020

@author: Shyam Bhuller

@Descripton: Bins data in the phase space and calculates the asymmetries in each region.
Cuts for the same axis is cosntant per region and only 1 cut per axis is done (this makes 32 regions so is
more than enough to study). Could support more cuts (Unlikely).
"""
import Kinematic as kin  # vecotrised 4-vector kineamtics
import Plotter as pt  # generic plotter with consistent formatting and curve fitting
import DataManager as dm  # handles data opened from data files
from math import log10, floor
import numpy as np
import matplotlib.pyplot as plt

"""rounds x to the 1st significant figure of y"""
def round_to(x, y):
    return round(x, -int(floor(log10(abs(y)))))


"""Maps CM variables to the parameter number for easy use"""
def GetParameter(parameter):
    global m12, m34, hel_1, hel_3, phi
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
   Does this for 2 bins only."""
def CutData(values, parameter, cut):
    out = []
    for i in range(len(cut)-1):
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
        f = -1  # CP needs to flip sign of C_Tbar
    else:
        f = 1

    """Helicity angles"""
    cos_p_1 = kin.HelicityAngle(particles['p_1'] + particles['p_2'], particles['p_0'], particles['p_1'])  # helicity angle of p1

    cos_p_3 = kin.HelicityAngle(particles['p_3'] + particles['p_4'], particles['p_0'], particles['p_3'])  # helicity angle of p3

    """Invariant masses"""
    m_12 = kin.Mag_4(particles['p_1'] + particles['p_2'])  # invariant mass of p1p2

    m_34 = kin.Mag_4(particles['p_3'] + particles['p_4']) # .. p3p4

    """Decay plane angle"""
    phi = kin.Decay_Plane_Angle(particles['p_0'], particles['p_1'], particles['p_2'], particles['p_3'], particles['p_4'])

    """scalar triple product momentum"""
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
        print("\r"+str(round(progress/len(data) * 100, 2)), end="")
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


"""Will load data with 1M regular and 1M conjugate decays and will bin and compute the asymmetries per bin and save them into a file"""
def CreateData():
    order = [3, 4, 2, 1, 5]  # order of the CM variable in which to bin
    iterations = [2, 2, 2, 8, 2]  # how many different bin regions should we permute through
    r = [False, False, True, False, False]  # which CM variable should we bin near the resonance

    print('loading data')
    datas = dm.GenerateDataFrames('\P45_A0.75_1M', False)  # load regular particle dictionaries, each one has a unique random seed
    datas_CP = dm.GenerateDataFrames('\P45_A0.75_1M_CP', True)  # load conjugate particle dictionary

    p = dm.MergeData(datas)  # merge the samples into one
    pbar = dm.MergeData(datas_CP)

    p = {'p_0': p[0], 'p_1': p[1], 'p_2': p[2], 'p_3': p[3], 'p_4': p[4]}  # recreate the particle dictionary
    pbar = {'p_0': pbar[0], 'p_1': pbar[1], 'p_2': pbar[2], 'p_3': pbar[3], 'p_4': pbar[4]}


    print("calculating CM variables and C_T")
    hel_1, hel_3, m12, m34, phi, C_T = MultiSampleDalitzParameters(p, False, 1000)  # get CM variables and C_T of regular decays
    print("\nA_T\n: " + str(kin.TP_Amplitude(C_T)))  # print asymmetry over all phasespace
    print("\nbinning data")
    bins, edges = BinData(order, iterations, C_T, 100000, r)  # bin C_T and return the bin regions

    print("calculating CM variables and C_Tbar")
    hel_1, hel_3, m12, m34, phi, C_Tbar = MultiSampleDalitzParameters(pbar, True, 1000)  # get CM variables and C_T of regular decays
    print("\nA_Tbar\n: " + str(kin.TP_Amplitude(C_Tbar)))
    print("\nbinning data")

    """We want to bin C_T wrt to bin regions optimised for A_T. this keeps the regions consistent when
    calculating A_CP."""
    bins_CP = []  # bin list for the conugate decays
    parameters = [GetParameter(order[0]), GetParameter(order[1]), GetParameter(order[2]), GetParameter(order[3]), GetParameter(order[4])]  # get parameters in order of previous binning scheme
    """finds the values of C_T which satisfies all the bin regions"""
    for i in range(32):
        tmp = []
        for j in range(len(C_Tbar)):
            if(parameters[0][j] >= edges[i][0][0] and parameters[0][j] <= edges[i][0][1]):
                if(parameters[1][j] >= edges[i][1][0] and parameters[1][j] <= edges[i][1][1]):
                    if(parameters[2][j] >= edges[i][2][0] and parameters[2][j] <= edges[i][2][1]):
                        if(parameters[3][j] >= edges[i][3][0] and parameters[3][j] <= edges[i][3][1]):
                            if(parameters[4][j] >= edges[i][4][0] and parameters[4][j] <= edges[i][4][1]):
                                tmp.append(C_T[j])
        bins_CP.append(tmp)


    print("calculating asymmetries")
    A_Ts = []
    A_Tbars = []
    A_CPs = []
    """For each bin, calculate the asymmetries"""
    for i in range(len(bins)):
        A_T = kin.TP_Amplitude(bins[i])  # P asymmetry (includes error)
        A_Tbar = kin.TP_Amplitude(bins_CP[i])  # P asymmetry of conjugate decay
        A_CP = kin.A_CP(A_T, A_Tbar)  # CP asymmetry
        A_Ts.append(A_T)
        A_Tbars.append(A_Tbar)
        A_CPs.append(A_CP)


    zipped = list(zip(edges, A_CPs, A_Ts, A_Tbars))  # keep these parameters in a zipped object
    # randomly shuffle the bin regions.
    #This randomly assigns a bin region to a number (its index in the list)
    #required to remove a bias in the numbering due to the binning scheme.
    np.random.shuffle(zipped)

    edges, A_CPs, A_Ts, A_Tbars = zip(*zipped)  # unzip data

    np.save("bin_edges.npy", edges)  # save data into a numpy array, dont want to spend forever recalculating these if possible
    np.save("bin_A_T.npy", A_Ts)
    np.save("bin_A_Tbar.npy", A_Tbars)
    np.save("bin_A_CP.npy", A_CPs)


"""Will print a table of all the bin regions, formatted for use in excel/latex"""
def PrintBins(bin_regions, edges):
    print("cos(theta_D)  | cos(theta_pi) |    m_Kpi     |     m_DDbar       |    phi")  # labels
    """Constructs the row for a single region and prints it"""
    for i in range(len(bin_regions)):
        regions = edges[i]
        strings = ["(" + str(round(regions[x][0], 2)) + ", " + str(round(regions[x][1], 2)) + ")" for x in range(5)]  # creates the bin region per CM variable in a string, up to 2 significant figures
        string = " |& ".join(strings)  # join the strings by the defined one, helpful for Latexformatting
        print(string, r"\\")  # pad with \\ for Latex formatting


"""Plots the asymmetries for each bin as well as the mean value of the asymmetries with confidence intervals"""
def PlotData(A_Ts, A_Tbars, A_CPs):
    Asyms = [A_Ts, A_Tbars, A_CPs]  # stores values in a list for easy access
    labels = ['$A_{T}$', '$\\bar{A}_{T}$', '$\mathcal{A}_{CP}$']  # labels for each plot

    plot = 131  # 1 row, 3 columns, start at 1st postition
    """Loops through each figure and plots the asymmetry, global asymmetry and 1 and 5 sigma error bars for the global asymmetry"""
    for i in range(len(Asyms)):
        ax = plt.subplot(plot+i)  # create the subplot
        Asym_mean = [Asyms[i][j][0] for j in range(len(Asyms[i]))]  # get mean values
        avg = np.mean(Asym_mean)  # get global asymmetry
        Asym_error = [Asyms[i][j][1] for j in range(len(Asyms[i]))]  # get uncertainties
        avg_error = np.mean(Asym_error)  # get global uncertainty
        pt.ErrorPlot([bin_regions, Asym_mean], axis=True, x_axis='Bin Region', y_axis='Asymmetry value', y_error=Asym_error, legend=True, label=labels[i])  # plot bin region values
        plt.hlines(avg, -5, 35, linestyle='--')  # plot line indicating the global asymmetry
        plt.fill_between(np.linspace(-5, 35, 40), avg + avg_error, avg - avg_error, color="black", alpha=0.1)  # 1 sigma global uncertainty region
        plt.fill_between(np.linspace(-5, 35, 40), avg + 5*avg_error, avg - 5*avg_error, color="black", alpha=0.1)  # 5 sigma global uncertainty region
        ax.set_ylim((-0.1, 0.1))  # det plot limits equal to each other for easy interpretation
        ax.set_xlim((-5, 35))


"""Main Body Call the functions above here or in the spyder terminal."""
# if data hasn't been made, call that function first
bin_regions = np.linspace(1, 32, 32)  # define bin regions
edges = np.load("bin_edges.npy")  # load bin regions
A_Ts = np.load("bin_A_T.npy")  # load P asymmetries
A_Tbars = np.load("bin_A_Tbar.npy")  # load P conjugate asymmetries
A_CPs = np.load("bin_A_CP.npy")  # load CP asymmetries

PrintBins(bin_regions, edges)  # shows the bins in a table
PlotData(A_Ts, A_Tbars, A_CPs)  # plot the data
