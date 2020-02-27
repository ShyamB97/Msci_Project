# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 09:49:41 2020

@author: bhull
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from scipy.optimize import curve_fit
from math import log10, floor

plt.style.use('default')
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)


def round_to(x, y):
    return round(x, -int(floor(log10(abs(y)))))


"""Generate sample data to plot with."""
def SampleData(size=100):
    size = int(size)
    return np.random.randint(0, 1000, size=size)


"""Generic code to hande labels/axis etc."""
def AxisControl(legend, axis, label, x_axis, y_axis):
    if axis is True:
        plt.xlabel(x_axis, fontsize=14)
        plt.ylabel(y_axis, fontsize=14)
    #plt.title('title')
    if legend is True:
        plt.legend(fontsize=14)
    plt.tight_layout()


"""Plots a line graph with errors"""
def ErrorPlot(data, legend=False, axis=False, label='', x_axis='', y_axis='', x_error=None, y_error=None, capsize=5, linestyle='', marker='o', markersize=5):
    plot = plt.errorbar(data[0], data[1], y_error, x_error, capsize=capsize, linestyle=linestyle, marker=marker, markersize=markersize, label=label)
    AxisControl(legend, axis, label, x_axis, y_axis)
    return plot


"""Plots a Histogram (What else?)"""
def Histogram(data, legend=False, axis=False, label='', x_axis='', y_axis='', bins=50, density=False, histtype='bar', edgecolor='black', linewidth=1.2, alpha=0.25, weights=None):
    plt.hist(data, label=label, bins=bins, density=density, histtype=histtype, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, weights=weights)
    AxisControl(legend, axis, label, x_axis, y_axis)


"""Plot multiple histograms"""
def Histogram_Multi(data, legend=False, axis=False, labels=None, x_axis='', y_axis='', bins=50, density=False, histtype='bar', edgecolor='black', linewidth=1.2, alpha=0.25):
    for i in range(len(data)):
        if labels == None:
            args = [i, x_axis, y_axis, bins, density, histtype, edgecolor, linewidth, alpha]
        else:
            args = [labels[i], x_axis, y_axis, bins, density, histtype, edgecolor, linewidth, alpha]
        Histogram(data[i], False, False, *args)

    AxisControl(legend, axis, labels, x_axis, y_axis)


def BW(x, A, M, T):
    gamma = np.sqrt(M**2 * (M**2 + T**2))
    k = A * ((2 * 2**0.5)/np.pi) * (M * T * gamma)/((M**2 + gamma)**0.5)
    return k/((x**2 - M**2)**2 + (M*T)**2)


def BWMulti(x, A1, M1, T1, A2, M2, T2, A3, M3, T3):#, M4, T4, M5, T5):
    return BW(x, A1, M1, T1) + BW(x, A2, M2, T2) + BW(x, A3, M3, T3)# + BW(x, M4, T4) + BW(x, M5, T5)


"""Fits Breit Wigner curve to invariant mass plots and returns resonance mass and lifetime in MeV"""
def BWCurveFit(data, legend=False, axis=False, label='', x_axis='', y_axis='', lines=False, single=True, fit_parameters=[1, 1, 1], weights=None, binWidth=0.01, binNum=50):
    """E must be in units of GeV"""
    data = data/1000
    hist, bins = np.histogram(data, bins=binNum, density=1, weights=weights)
    x = (bins[:-1] + bins[1:])/2  # center of bins

    if single is True:
        popt, cov = curve_fit(BW, x, hist, p0=fit_parameters)
    if single is False:
        popt, cov = curve_fit(BWMulti, x, hist, p0=fit_parameters)

    x_inter = np.linspace(x[0], x[-1], 500)

    if single is True:
        y = BW(x_inter, *popt)
        _, p = stats.chisquare(hist, BW(x, *popt))
        Mass = popt[1]
        Decay = popt[2]
        Mass_Error = cov[1, 1]
        Decay_Error = cov[2, 2]
        label += " ($M_{R}$=" + str(round_to(Mass, Mass_Error)) + "$\pm$" + str(round_to(Mass_Error, Mass_Error)) + "$GeV$)"

    if single is False:
        y = BWMulti(x_inter, *popt)
        _, p = stats.chisquare(hist, BWMulti(x, *popt))

    p = stats.norm.ppf(p)

    plt.bar(x, hist, binWidth, alpha=0.5, label=label)
    plt.plot(x_inter, y)
    if lines is True:
        if single is True:
            plt.vlines(popt[1], min(y), max(y), linewidth=2, linestyle="--")
            #plt.hlines(max(y)/2, -popt[2]/2 + popt[1], popt[2]/2 + popt[1], linewidth=2, linestyle="--")
        if single is False:
            plt.vlines(popt[1], min(y), max(y), linewidth=2, linestyle="--")
            plt.vlines(popt[4], min(y), max(y), linewidth=2, linestyle="--")
            plt.vlines(popt[7], min(y), max(y), linewidth=2, linestyle="--")

    AxisControl(legend, axis, label, x_axis, y_axis)

    popt *= 1000
    cov *= 1000
    return [popt[1], cov[1, 1]], [popt[2], cov[2, 2]]


"""Will plot multiple datasets in the form provieded by this module"""
def BWMultiFit(data, legend=False, axis=False, labels=None, x_axis='', y_axis='', lines=False, single=True, fit_parameters=[1, 1 ,1]):
    for i in range(len(data)):
        if labels == None:
            args = [i, x_axis, y_axis, lines, single, fit_parameters]
        else:
            args = [labels[i], x_axis, y_axis, lines, single, fit_parameters]
        BWCurveFit(data[i], False, False, *args)
    AxisControl(legend, axis, labels, x_axis, y_axis)


#data = [SampleData(), SampleData()]
#Multi_Plotter(data, Histogram, legend=False, labels=['sample 1', 'sample 2'])















