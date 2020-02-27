# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:26:14 2020

@author: bhull
"""
import time
import numpy as np
import Kinematic as kin
import DataManager as dm
import Plotter as pt
from natsort import natsorted, ns
import matplotlib.pyplot as plt


def AnalysePwave(directory, label, CP=False):
    data = dm.GenerateDataFrames(directory, CP)

    if CP is False:
        f = 1
    else:
        f = -1

    value = []
    error = []
    factors = [1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 1E1, 1E2, 1E3, 1E4, 1E5]
    for i in range(len(factors)):
        data_amp = data[i*10:(i+1)*10]
        data_amp = dm.MergeData(data_amp)
        C_T = f * kin.Scalar_TP(kin.Vector_3(data_amp[3]), kin.Vector_3(data_amp[4]), kin.Vector_3(data_amp[1]))
        A_T = kin.TP_Amplitude(C_T)
        value.append(A_T[0])
        error.append(A_T[1])


    return pt.ErrorPlot([factors, value], label=label, legend=False, axis=True, y_error=error, x_axis="relative P-wave amplitudes", y_axis="")


def AnalyseCPwave(direct, direct_CP, label):
    datas = dm.GenerateDataFrames(direct, False)
    datas_CP = dm.GenerateDataFrames(direct_CP, True)

    value = []
    error = []
    factors = [1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 1E1, 1E2, 1E3, 1E4, 1E5]
    for i in range(len(factors)):
        data_amp = datas[i*10:(i+1)*10]
        data_amp = dm.MergeData(data_amp)
        data_amp_CP = datas_CP[i*10:(i+1)*10]
        data_amp_CP = dm.MergeData(data_amp_CP)
        C_T = kin.Scalar_TP(kin.Vector_3(data_amp[3]), kin.Vector_3(data_amp[4]), kin.Vector_3(data_amp[1]))
        C_Tbar = -kin.Scalar_TP(kin.Vector_3(data_amp_CP[3]), kin.Vector_3(data_amp_CP[4]), kin.Vector_3(data_amp_CP[1]))
        A_T = kin.TP_Amplitude(C_T)
        A_Tbar = kin.TP_Amplitude(C_Tbar)
        A_CP = kin.A_CP(A_T, A_Tbar)
        value.append(A_CP[0])
        error.append(A_CP[1])

    return pt.ErrorPlot([factors, value], label=label, legend=False, axis=True, y_error=error, x_axis="relative P-wave amplitudes", y_axis="$\mathcal{A}_{CP}$")


phases = [0, 45, 90, 135, 180, 225, 270, 315]
labels = ['$0$', '$\\frac{\pi}{4}$', '$\\frac{\pi}{2}$', '$\\frac{3\pi}{4}$',
          '$\pi$', '$\\frac{5\pi}{4}$', '$\\frac{3\pi}{2}$', '$\\frac{7\pi}{4}$']

plotpos = 241
for i in range(len(phases)):
    s = time.time()
    folder = "\Phase-" + str(phases[i])
    folder_CP = folder + "_CP"
    ax = plt.subplot(plotpos+i)
    ax.set_xscale('log')
    ax.set_ylim([-0.06, 0.06])
    ax.set_title(labels[i])
#    plot = AnalyseCPwave(folder, folder_CP, "")
    plot = AnalysePwave(folder, "$A_{T}$", False)
    plot = AnalysePwave(folder_CP, "$\\bar{A}_{T}$", True)
    if(i == 0):
        h, l = ax.get_legend_handles_labels()
        plt.figlegend(h, l)
    e = time.time()
    print('time:' + str(e-s))

plt.tight_layout()
#plt.savefig('Pwave_Phase', dpi=500)
