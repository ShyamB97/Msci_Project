# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:26:14 2020

@author: Shyam Bhuller

@Description: Code used to analyse how SP wave interference casues a varying asymmetry. calculates
ans plots the asymmetries for varying amounts of interference.
"""
import Kinematic as kin  # vecotrised 4-vector kineamtics
import Plotter as pt  # generic plotter with consistent formatting and curve fitting
import DataManager as dm  # handles data opened from data files
import time
import matplotlib.pyplot as plt
import numpy as np

"""Analyses and plots parity asymmetry varying the relative p-wave amplitude"""
def AnalysePwave(directory, factors, label, CP=False, plot=True, color=None, marker=None):
    data = dm.GenerateDataFrames(directory, CP) # gets particle data from the event files in directory

    """"Adjusts value of C_T if we use CP data or not"""
    if CP is False:
        f = 1
    else:
        f = -1

    value = []  # mean asymmetry value
    error = []
    for i in range(len(factors)):
        data_amp = data[i*10:(i+1)*10]  # gets every 10 data points
        data_amp = dm.MergeData(data_amp)  # combines the different seeded data into one set

        C_T = f * kin.Scalar_TP(kin.Vector_3(data_amp[3]), kin.Vector_3(data_amp[4]), kin.Vector_3(data_amp[1]))  # calculate scalar triple product
        A_T = kin.TP_Amplitude(C_T)  # calculates parity asymmetry

        value.append(A_T[0]*100)
        error.append(A_T[1]*100)

    # will return the plot so figures can be built
    if plot is True:
        return pt.ErrorPlot([factors, value], label=label, legend=True, axis=True, y_error=error, x_axis="relative P-wave amplitudes", y_axis="", capsize=5, markersize=5, linestyle="-", marker=marker,color=color)
    # in case we want to do something else
    if plot is False:
        return value, error


"""Analyses and plots A_CP varying the relative p-wave amplitude"""
def AnalyseCPwave(direct, direct_CP, factors, label, plot=True):
    datas = dm.GenerateDataFrames(direct, False)  # gets particle data from the event files in directory
    datas_CP = dm.GenerateDataFrames(direct_CP, True)  # gets particle CP data from the event files in directory

    value = []  # mean asymmetry value
    error = []
    for i in range(len(factors)):
        data_amp = datas[i*10:(i+1)*10]  # gets every 10 data points
        data_amp = dm.MergeData(data_amp)  # combines the different seeded data into one set

        data_amp_CP = datas_CP[i*10:(i+1)*10]
        data_amp_CP = dm.MergeData(data_amp_CP)

        C_T = kin.Scalar_TP(kin.Vector_3(data_amp[3]), kin.Vector_3(data_amp[4]), kin.Vector_3(data_amp[1]))  # calculate scalar triple product
        C_Tbar = -kin.Scalar_TP(kin.Vector_3(data_amp_CP[3]), kin.Vector_3(data_amp_CP[4]), kin.Vector_3(data_amp_CP[1]))

        A_T = kin.TP_Amplitude(C_T)  # calculates parity asymmetry
        A_Tbar = kin.TP_Amplitude(C_Tbar)
        A_CP = kin.A_CP(A_T, A_Tbar)  # calculates CP asymmetry

        value.append(A_CP[0])
        error.append(A_CP[1])

    # will return the plot so figures can be built
    if plot is True:
        return pt.ErrorPlot([factors, value], label=label, legend=False, axis=True, y_error=error, x_axis="relative P-wave amplitudes", y_axis="$\mathcal{A}_{CP}$")
    # in case we want to do something else
    if plot is False:
        return value, error


"""Function used to plot A_T or A_CP for varying SP interferences, plots different phases in different figures"""
def MultiFigure(phases, factors, labels):
    plotpos = 241  # create a 2 row, 4 column figure
    
    """Analyse and plot the asymmetries"""
    for i in range(len(phases)):
        s = time.time()
        folder = "\Phase-" + str(phases[i])  # directory to open
        #folder_CP = folder + "_CP"  # directory of CP files
    
        ax = plt.subplot(plotpos+i)  # define plot location
        ax.set_xscale('log')  # change x axis to a log scale
        ax.set_ylim([-0.06, 0.06])  # bound the yaxis for every plot
        ax.set_title(labels[i])  # set the title as the phase

        AnalysePwave(folder, factors, "$A_{T}$", False)  # plot A_T
        #AnalysePwave(folder_CP, factors, "$\\bar{A}_{T}$", True)  # plot A_Tbar
        #AnalyseCPwave(folder, folder_CP, factors, "")  # plot A_CP

        """define the legend only once"""
        if(i == 0):
            h, l = ax.get_legend_handles_labels()  # get labels from first plot
            plt.figlegend(h, l)  # make that label the label for the entire figure
        e = time.time()
        print('time:' + str(e-s))

    plt.tight_layout()


"""Function used to plot A_T or A_CP for varying SP interferences, plots all phases in different figures"""
def SingleFigure(phases, factors, labels):
    color=['red', 'magenta', 'purple', 'blue', 'cyan', 'lime', 'yellow', 'orange']  # define the color for each phase
    for i in range(len(phases)):
        s = time.time()
        folder = "\Phase-" + str(phases[i])  # directory to open
        AnalysePwave(folder, factors, labels[i], False, color[i])  # plot A_T
        e = time.time()
        print('time:' + str(e-s))

    plt.xscale('log')  # set x axis to a log scale
    plt.ylabel("$A_{T}$%")
    plt.tight_layout()


"""Main Body call a function here or in the spyder console"""
"""Define phases and labels for each plot"""
factors = [1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 1E1, 1E2, 1E3, 1E4, 1E5]  # relative p-wave amplitudes
phases = [0, 45, 90, 135, 180, 225, 270, 315]
labels = ['$0$', '$\\frac{\pi}{4}$', '$\\frac{\pi}{2}$', '$\\frac{3\pi}{4}$',
          '$\pi$', '$\\frac{5\pi}{4}$', '$\\frac{3\pi}{2}$', '$\\frac{7\pi}{4}$']

#SingleFigure(phases, factors, labels)

for i in range(len(phases)):
    s = time.time()
    folder = "\Phase-" + str(phases[i])  # directory to open
    value, error = AnalysePwave(folder, factors, labels[i], plot=False)
    np.save("interference"+folder+"_value"+".npy", value)
    np.save("interference"+folder+"_error"+".npy", error)
    e = time.time()
    print(e-s)

np.save("interference\\factors.npy", factors)
np.save("interference\\phases.npy", phases)