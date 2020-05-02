# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 09:49:41 2020

@author: Shyam Bhuller

@Description: Contians Generic plotting functions I use all the time. It is beneficial as set formating is done,
and keeps the plots I generate consistent with respects to the style. Also contains some code which is used for
Fitting data.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from scipy.optimize import curve_fit
from math import log10, floor

plt.style.use('default')  # set the overall plot style here
font = {'size'   : 16}  # set font here

matplotlib.rc('font', **font)  # assign font


"""Round x to the 1st significant figure of y."""
def round_to(x, y):
    return round(x, -int(floor(log10(abs(y)))))


"""Generate sample data to plot with."""
def SampleData(size=100):
    return np.random.randint(0, 1000, size=int(size))


"""Generic code to hande labels/axis etc."""
def AxisControl(legend, axis, label, x_axis, y_axis):
    """If we need to plot the axis labels"""
    if axis is True:
        plt.xlabel(x_axis, fontsize=16)
        plt.ylabel(y_axis, fontsize=16)

    """If we need to plot labels for different data"""
    if legend is True:
        plt.legend(fontsize=16)

    plt.tight_layout()  # optimises size of the plot to fit the default window size


"""Plots a line graph with errors"""
def ErrorPlot(data, legend=False, axis=False, label='', x_axis='', y_axis='', x_error=None, y_error=None, capsize=5, linestyle='', marker='o', markersize=5, color=None, alpha=1):
    plot = plt.errorbar(data[0], data[1], y_error, x_error, capsize=capsize, linestyle=linestyle, marker=marker, markersize=markersize, label=label, color=color, alpha=alpha)  # use Matplotlib errorbars
    AxisControl(legend, axis, label, x_axis, y_axis)  # handle the labels, axis etc.
    return plot


"""Plots a histogram (What else?)"""
def Histogram(data, legend=False, axis=False, label='', x_axis='', y_axis='', bins=50, density=False, histtype='bar', edgecolor='black', linewidth=1.2, alpha=0.25, weights=None):
    plt.hist(data, label=label, bins=bins, density=density, histtype=histtype, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, weights=weights)
    AxisControl(legend, axis, label, x_axis, y_axis)


"""Plot multiple histograms into one figure"""
def Histogram_Multi(data, legend=False, axis=False, labels=None, x_axis='', y_axis='', bins=50, density=False, histtype='bar', edgecolor='black', linewidth=1.2, alpha=0.25):
    """Cycles through each dataset and plots them according to if we want to label them or not"""
    for i in range(len(data)):
        """We need to define a label, if they were not, then make it the list index"""
        if labels == None:
            args = [i, x_axis, y_axis, bins, density, histtype, edgecolor, linewidth, alpha]
        else:
            args = [labels[i], x_axis, y_axis, bins, density, histtype, edgecolor, linewidth, alpha]
        Histogram(data[i], False, False, *args)  # plot the histogram for the ith sample. we dont generate a legend or axis as we hangle this below.

    AxisControl(legend, axis, labels, x_axis, y_axis)


"""Breit Wigner distribution with and ampltitude"""
def BW(x, A, M, T):
    # see https://en.wikipedia.org/wiki/Relativistic_Breit%E2%80%93Wigner_distribution for its definition
    gamma = np.sqrt(M**2 * (M**2 + T**2))  # formula is complex, so split it into multiple terms
    k = A * ((2 * 2**0.5)/np.pi) * (M * T * gamma)/((M**2 + gamma)**0.5)
    return k/((x**2 - M**2)**2 + (M*T)**2)


"""A linear sum of 3 Breit Wigner distributions, used to fit data with multiple resonances."""
def BWMulti(x, A1, M1, T1, A2, M2, T2, A3, M3, T3):
    return BW(x, A1, M1, T1) + BW(x, A2, M2, T2) + BW(x, A3, M3, T3)


"""Fits Breit Wigner curve to invariant mass plots and returns resonance mass and lifetime in MeV"""
def BWCurveFit(data, legend=False, axis=False, label='', x_axis='', y_axis='', lines=False, single=True, fit_parameters=[1, 1, 1], weights=None, binWidth=0.01, binNum=50):
    """E must be in units of GeV"""
    data = data/1000  # GeV conversion helps fitting
    hist, bins = np.histogram(data, bins=binNum, density=1, weights=weights)  # generate histogram values
    x = (bins[:-1] + bins[1:])/2  # center of bins

    """Should we plot multiple BW curves or not"""
    if single is True:
        popt, cov = curve_fit(BW, x, hist, p0=fit_parameters)  # use scipiy.optimise.curve_fit to do a least sum of squares fit. returns covariance matrix and optimal fit parameters
    if single is False:
        popt, cov = curve_fit(BWMulti, x, hist, p0=fit_parameters)  # ...

    cov = np.sqrt(cov)  # get standard deviations
    x_inter = np.linspace(x[0], x[-1], 500)  # create x values to draw the best fit curve

    """Single BW curve"""
    if single is True:
        y = BW(x_inter, *popt)  # create y values of interpoplated x values, using optimal fit parameters
        _, p = stats.chisquare(hist, BW(x, *popt))  # compute a Chi square test
        Mass = popt[1] 
        Decay = popt[2]  # decay with, not using this right now
        Mass_Error = cov[1, 1]
        Decay_Error = cov[2, 2]  # decay with error
        label += " ($M_{R}$=" + str(round_to(Mass, Mass_Error)) + "$\pm$" + str(round_to(Mass_Error, Mass_Error)) + "$GeV$)"  # creates a label containing the resonance mass value predicted
        #label += "\n ($\Gamma_{R}$=" + str(round_to(Decay, Decay_Error)) + "$\pm$" + str(round_to(Decay_Error, Decay_Error)) + "$GeV$)"  
        
    """Multiple BW curves"""
    if single is False:
        y = BWMulti(x_inter, *popt)
        _, p = stats.chisquare(hist, BWMulti(x, *popt))

    p = stats.norm.ppf(p)  # confidence of fit, in sigma, not using this right now

    plt.bar(x, hist, binWidth, alpha=0.5, label=label)  # draw histogram using a bar plot
    plt.plot(x_inter, y)  # plot interpolated data
    """If we want, we can draw lines illustating the decay width and the resonance mass"""
    if lines is True:
        if single is True:
            plt.vlines(popt[1], min(y), max(y), linewidth=2, linestyle="--")  # line at the mass
            #plt.hlines(max(y)/2, -popt[2]/2 + popt[1], popt[2]/2 + popt[1], linewidth=2, linestyle="--")  # lines indicating the decay width
        if single is False:
            plt.vlines(popt[1], min(y), max(y), linewidth=2, linestyle="--")  # line at resonant mass 1
            plt.vlines(popt[4], min(y), max(y), linewidth=2, linestyle="--")  # .. 2
            plt.vlines(popt[7], min(y), max(y), linewidth=2, linestyle="--")  # .. 3

    AxisControl(legend, axis, label, x_axis, y_axis)  # sort out the axis and labels

    popt *= 1000  # convert mass and decay width into MeV
    cov *= 1000  # convert covariance matrix into MeV
    if single is True:
        print([popt[1], cov[1, 1]], [popt[2], cov[2, 2]])  # print resonant mass and decay width with errors
    if single is False:
        print([popt[1], cov[1, 1]], [popt[2], cov[2, 2]])
        print([popt[4], cov[4, 4]], [popt[5], cov[5, 5]])
        print([popt[7], cov[4, 4]], [popt[8], cov[8, 8]])
    return [popt[1], cov[1, 1]], [popt[2], cov[2, 2]]  # return resonant mass and decay width with errors


"""Will plot BW curves for multiple datasets on the same figure, with similar style"""
def BWMultiFit(data, legend=False, axis=False, labels=None, x_axis='', y_axis='', lines=False, single=True, fit_parameters=[1, 1 ,1]):
    """Cycles through each dataset and plots them according to if we want to label them or not"""
    for i in range(len(data)):
        """We need to define a label, if they were not, then make it the list index"""
        if labels == None:
            args = [i, x_axis, y_axis, lines, single, fit_parameters]
        else:
            args = [labels[i], x_axis, y_axis, lines, single, fit_parameters]
        results = BWCurveFit(data[i], False, False, *args)  # Fit the curves but not the labels or axis, it is handled below
    
    AxisControl(legend, axis, labels, x_axis, y_axis)
    return results

