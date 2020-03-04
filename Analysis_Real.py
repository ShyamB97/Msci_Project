# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:35:59 2019

@author: bhull
"""
import uproot
import numpy as np
import Kinematic as kin
import DataManager as dm
import matplotlib.pyplot as plt
import Plotter as pt
import os
import pandas as pd
from scipy.optimize import curve_fit
from scipy import integrate
from math import log10, floor
from scipy.special import factorial

def round_to(x, y):
    return round(x, -int(floor(log10(abs(y)))))

def Gaussian(x, A, B, C):
    return A*np.exp(-0.5 * np.square((x-B)/C))


def BoostIntoRest(p):
    n = kin.Direction(p['p_0'])
    A = kin.Beta(p['p_0'])
    G = kin.Gamma(A)

    Mats = []
    for i in range(len(A)):
        mat = kin.Boost(G[i], A[i], n[i])
        Mats.append(mat)
    Mats = np.array(Mats)


    B0 = kin.VectorMatMul(Mats, p['p_0'])
    D = kin.VectorMatMul(Mats, p['p_1'])
    Dbar = kin.VectorMatMul(Mats, p['p_2'])
    K = kin.VectorMatMul(Mats, p['p_3'])
    pi = kin.VectorMatMul(Mats, p['p_4'])
    return [B0, D, Dbar, K, pi]


def PrepareData():
    p, pbar, weights, weightsbar = dm.ReadRealData()
    p = BoostIntoRest(p)
    pbar = BoostIntoRest(pbar)
    return p, pbar, weights, weightsbar


def CompWieght(data, x_axis, x_axisW):
    ax = plt.subplot(121)
    ax.set_title("(a)")
    pt.Histogram(data, axis=True, x_axis=x_axis, y_axis='Number of Events')
    ax = plt.subplot(122)
    ax.set_title("(b)")
    pt.Histogram(data, weights=weights, axis=True, x_axis=x_axisW, y_axis='Weighted Number of Events')


"""Plots the unweighted and weighted invariant mass plots and fits the weighted data"""
def InvariantMassComp(particles, weights):
    inv_mass = kin.Mag_4(np.sum(particles[1:], axis=0))
    ax = plt.subplot(121)
    pt.Histogram(inv_mass, axis=True, x_axis='$m_{K^{+}\pi^{-}}(MeV)$', y_axis='number of events')
    ax = plt.subplot(122)
    #pt.Histogram(inv_mass, axis=True, x_axis='$m_{K^{+}\pi^{-}}(MeV)$', y_axis='weighted number of events', weights=weights)
    mass, width = pt.BWCurveFit(inv_mass, weights=weights, fit_parameters=[0, 5, 1], legend=True, binWidth=0.01, binNum=50, x_axis='$S(GeV)$', y_axis='weighted number of events', axis=True)


def CalculateAT(particles, weights):
    C_T = kin.Scalar_TP(kin.Vector_3(particles[3]), kin.Vector_3(particles[4]), kin.Vector_3(particles[1]))

    bins, edges = np.histogram(C_T/1E8, bins=15, weights=weights)
    print(np.sum(bins))
    x = (edges[:-1] + edges[1:])/2

    popt, cov = curve_fit(Gaussian, x, bins, p0=[max(bins), 0, 2])
    popterr = [cov[0, 0], cov[1, 1], cov[2, 2]]

    popt_lower = [popt[0] - popterr[0], popt[1] - popterr[1], popt[2] - popterr[2]]
    popt_upper = [popt[0] + popterr[0], popt[1] + popterr[1], popt[2] + popterr[2]]

    expected = Gaussian(x, *popt)
    chisqr = np.sum((bins - expected)**2 / expected)
    dof = len(bins) - 1

    chisqrRed = chisqr/dof

    x_inter = np.linspace(x[0], x[-1], 500)  # generate interpolated data for plotting
    y = Gaussian(x_inter, *popt)

    y_lower = Gaussian(x_inter, *popt_lower)
    y_upper = Gaussian(x_inter, *popt_upper)


    int1, _ = integrate.quad(Gaussian, 0, max(x), args=(popt[0], popt[1], popt[2]))
    int2, _ = integrate.quad(Gaussian, min(x), 0, args=(popt[0], popt[1], popt[2]))

    int1_lower, _ = integrate.quad(Gaussian, 0, max(x), args=(popt_lower[0], popt_lower[1], popt_lower[2]))
    int2_lower, _ = integrate.quad(Gaussian, min(x), 0, args=(popt_lower[0], popt_lower[1], popt_lower[2]))

    int1_upper, _ = integrate.quad(Gaussian, 0, max(x), args=(popt_upper[0], popt_upper[1], popt_upper[2]))
    int2_upper, _ = integrate.quad(Gaussian, min(x), 0, args=(popt_upper[0], popt_upper[1], popt_upper[2]))

    A_T = (int1 - int2) / (int1 + int2)

    int1_err = (int1_upper - int1_lower)/2
    int2_err = (int2_upper - int2_lower)/2
    print(int1_err)
    print(int2_err)
    A_T_lower = (int1_lower - int2_lower) / (int1_lower + int2_lower)
    A_T_lower = A_T - A_T_lower
    A_T_upper = (int1_upper - int2_upper) / (int1_upper + int2_upper)
    A_T_upper = A_T_upper - A_T

    A_T_error = 1/(int1 + int2) * np.sqrt(int1_err**2 * (1-A_T)**2 + int2_err**2 * (1+A_T)**2)

    A = "$A=$" + str(round_to(popt[0], popterr[0])) + "$\\pm$" + str(round_to(popterr[0], popterr[0]))
    B = "$B=$" + str(round_to(popt[1], popterr[1])) + "$\\pm$" + str(round_to(popterr[1], popterr[1]))
    C = "$C=$" + str(round_to(popt[2], popterr[2])) + "$\\pm$" + str(round_to(popterr[2], popterr[2]))

    Chi = "$\chi^{2}/ndf =$" + str(round(chisqrRed, 3))

    label = A + "\n" + B + "\n" + C + "\n" + Chi

    plt.bar(x, bins, width=0.2, alpha=0.5)
    plt.ylim(0)
    plt.plot(x_inter, y_lower, color='grey')
    plt.plot(x_inter, y, color='red')
    plt.plot(x_inter, y_upper, color='grey')
    plt.title("$A_{T}=$" + str(round_to(A_T, A_T_error)) + "$\\pm$" + str(round_to(A_T_error, A_T_error)))
    plt.xlabel("$C_{T}$", fontsize=14)
    plt.ylabel("Weighted Number of Events", fontsize=14)
    plt.annotate(label, (1, 4))


def Yield(data, weights, title, x_axis):
    bins, edges = np.histogram(data, bins=100, weights=weights)  # bin data
    x = (edges[:-1] + edges[1:])/2  # get center of bins

    param_bounds = ([0, 0, 0], [max(bins), max(x), 1])  # set limits for the fit quantities

    popt, cov = curve_fit(Gaussian, x, bins, p0=[1, 1, 1], bounds=param_bounds)  # fit binned data to Gaussian function
    popterr = [cov[0, 0]**0.5, cov[1, 1]**0.5, cov[2, 2]**0.5]  # get the uncertianty in fit parameters

    popt_lower = [popt[0] - popterr[0], popt[1] - popterr[1], popt[2] - popterr[2]]  # get lower and upper limit of fit params
    popt_upper = [popt[0] + popterr[0], popt[1] + popterr[1], popt[2] + popterr[2]]

    # chi squared not needed
    #expected = Gaussian(x, *popt)
    #chisqr = np.sum(np.divide((bins - expected)**2, expected))
    #dof = len(bins) - 1
    #chisqrRed = chisqr/dof

    x_inter = np.linspace(x[0], x[-1], 500)  # generate interpolated data for plotting
    y = Gaussian(x_inter, *popt)

    y_lower = Gaussian(x_inter, *popt_lower)  # generate extremal fits
    y_upper = Gaussian(x_inter, *popt_upper)

    width = x[1] - x[0]  # width of bins
    integral = np.sqrt(2 * np.pi) * popt[0] * popt[2]/width  # is the yield
    error_fit = integral * ((popterr[0]/popt[0])**2 + (popterr[2]/popt[2])**2)**0.5  # is the error of the yield given the gaussian function is coorect
    error, _ = integrate.quad(Gaussian, popt[1], popt[1]+popt[2], args=(popt[0], popt[1], popt[2]))/width  # error in the yield given the probability of the events falls within 1 sigma (i.e. the actual error)

    # create text for table
    A = "|A          | $" + str(round_to(popt[0], popterr[0])) + " \pm " + str(round_to(popterr[0], popterr[0])) + "$"
    B = "|B          | $" + str(round_to(popt[1], popterr[1])) + " \pm " + str(round_to(popterr[1], popterr[1])) + "$"
    C = "|C          | $" + str(round_to(popt[2], popterr[2])) + " \pm " + str(round_to(popterr[2], popterr[2])) + "$"
    E = "|fit error  | " + str(round(error_fit, 2))
    F = "|yield      | " + str(round(integral, 2))
    G = "|yield error| " + str(round(error, 2))
    label = A + "\n" + B + "\n" + C + "\n" + E + "\n" + F + "\n" + G
    print("-----------------------")
    print("|quantity   | value")
    print("+-----------+----------")
    print(label)
    print("-----------------------")

    plt.bar(x, bins, width=width, alpha=0.5, edgecolor="black", linewidth=1)
    plt.ylim(0)
    plt.plot(x_inter, y_lower, color='grey')
    plt.plot(x_inter, y, color='red')
    plt.plot(x_inter, y_upper, color='grey')
    #plt.title(title + str(round(integral, 2)) + "$\\pm$" + str(round(error, 2)))
    plt.title(title)
    #plt.xlim(5.2, 5.4)
    plt.xlabel(x_axis, fontsize=14)
    plt.ylabel("Weighted Number of Events", fontsize=14)
    #plt.annotate(label, (0.7, 0.7), xycoords='axes fraction')
    return integral, error


#x = np.linspace(-10, 10, 500)
#y = Gaussian(x, 1, 0, 1)
#plt.plot(x, y)

particles, particlesbar, weights, weightsbar = PrepareData()
C_T = kin.Scalar_TP(kin.Vector_3(particles[3]), kin.Vector_3(particles[4]), kin.Vector_3(particles[1]))
C_Tbar = kin.Scalar_TP(kin.Vector_3(particlesbar[3]), kin.Vector_3(particlesbar[4]), kin.Vector_3(particlesbar[1]))
mB0 = kin.Mag_4(particles[0])/1000  # GeV conversion
mBbar0 = kin.Mag_4(particlesbar[0])/1000  # GeV conversion

_min = 5.2
_max = 5.35

mB0_l = mB0[C_T < 0]
mB0_u = mB0[C_T > 0]
weights_l = weights[C_T < 0]
weights_u = weights[C_T > 0]

#weights_l = weights_l[np.where(np.logical_and(mB0_l>=_min, mB0_l<=_max))]
#mB0_l = mB0_l[np.where(np.logical_and(mB0_l>=_min, mB0_l<=_max))]
#weights_u = weights_u[np.where(np.logical_and(mB0_u>=_min, mB0_u<=_max))]
#mB0_u = mB0_u[np.where(np.logical_and(mB0_u>=_min, mB0_u<=_max))]


mBbar0_l = mBbar0[-C_Tbar < 0]
mBbar0_u = mBbar0[-C_Tbar > 0]
weightsbar_l = weightsbar[-C_Tbar < 0]
weightsbar_u = weightsbar[-C_Tbar > 0]

#weightsbar_l = weightsbar_l[np.where(np.logical_and(mBbar0_l>=_min, mBbar0_l<=_max))]
#mBbar0_l = mBbar0_l[np.where(np.logical_and(mBbar0_l>=_min, mBbar0_l<=_max))]
#weightsbar_u = weightsbar_u[np.where(np.logical_and(mBbar0_u>=_min, mBbar0_u<=_max))]
#mBbar0_u = mBbar0_u[np.where(np.logical_and(mBbar0_u>=_min, mBbar0_u<=_max))]


ax = plt.subplot(221)
N_l, N_l_error = Yield(mB0_l, weights_l, "(a)", "$m_{B^{0}}(C_{T} < 0)$")

ax = plt.subplot(222)
N_u, N_u_error = Yield(mB0_u, weights_u, "(b)", "$m_{B^{0}}(C_{T} > 0)$")

ax = plt.subplot(223)
N_l_CP, N_l_error_CP = Yield(mBbar0_l, weightsbar_l, "(c)", "$m_{\\bar{B}^{0}}(-\\bar{C}_{T} < 0)$")

ax = plt.subplot(224)
N_u_CP, N_u_error_CP = Yield(mBbar0_u, weightsbar_u, "(d)", "$m_{\\bar{B}^{0}}(-\\bar{C}_{T} > 0)$")

A_T = (N_u - N_l) / (N_u + N_l)
A_T_error = 1/(N_u + N_l) * np.sqrt(N_u_error**2 * (1-A_T)**2 + N_l_error**2 * (1+A_T)**2)


A_Tbar = (N_u_CP - N_l_CP) / (N_u_CP + N_l_CP)
A_T_errorbar = 1/(N_u_CP + N_l_CP) * np.sqrt(N_u_error_CP**2 * (1-A_Tbar)**2 + N_l_error_CP**2 * (1+A_Tbar)**2)

A_CP = kin.A_CP((A_T, A_T_error), (A_Tbar, A_T_errorbar))

print("A_T = " + str((A_T, A_T_error)))
print("A_Tbar = " + str((A_Tbar, A_T_errorbar)))
print("A_CP = " + str(A_CP))


