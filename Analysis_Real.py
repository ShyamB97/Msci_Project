# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:35:59 2019

@author: Shyam Bhuller

@Description: Has code which analyses the LHCb data. It can calculate the asymmetries and yields
through various techniques.
"""
import Kinematic as kin  # vecotrised 4-vector kineamtics
import Plotter as pt  # generic plotter with consistent formatting and curve fitting
import DataManager as dm  # handles data opened from data files
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate
from math import log10, floor


"""Round x to the 1st significant figure of y"""
def round_to(x, y):
    return round(x, -int(floor(log10(abs(y)))))


"""Gaussan function used for fitting"""
def Gaussian(x, A, B, C):
    return A*np.exp(-0.5 * np.square((x-B)/C))


"""A function which Boosts particles into the rest frame of the parent."""
def BoostIntoRest(p):
    n = kin.Direction(p['p_0'])  # get the direction vector of p_0 in each event
    beta = kin.Beta(p['p_0'])  # calculates the speed of p_0
    G = kin.Gamma(beta)  # Lorentz factor

    Mats = []
    """For each event, computed the boost matrix and adds it into a list"""
    for i in range(len(beta)):
        mat = kin.Boost(G[i], beta[i], n[i])  # create the boost matrix
        Mats.append(mat)
    Mats = np.array(Mats)  # conver to a 3D matrix, 1st axis is the events, other 2 are the matrix


    B0 = kin.VectorMatMul(Mats, p['p_0'])  # matrix multiplies all events at once to boost p_0 into the rest frame
    D = kin.VectorMatMul(Mats, p['p_1'])  # ...
    Dbar = kin.VectorMatMul(Mats, p['p_2'])
    K = kin.VectorMatMul(Mats, p['p_3'])
    pi = kin.VectorMatMul(Mats, p['p_4'])
    return [B0, D, Dbar, K, pi]


"""Prepares real data so that it is returned as a list and is boosted into the COM frame."""
def PrepareData(name, cut):
    p, pbar, weights, weightsbar = dm.ReadRealData(name, cut)  # get the particle dictionaries and weights, splits regular to conjugate
    p = BoostIntoRest(p)  # Boosts particles into the COM frame
    pbar = BoostIntoRest(pbar)
    return p, pbar, weights, weightsbar


"""sorts list of data from different files by particle type rather than file"""
def RearrangeParticleLists(lst):
    tmp = []  # final list to return
    for i in range(5):
        tmp2 = []  # list of the same particle in the different event files
        for j in range(len(lst)):
            tmp2.append(lst[j][i])
        tmp.append(tmp2)
    return tmp



"""Prepares real data from multiple files and merges them together."""
def ReadRealDataMulti(names, cuts):
    global ps, pbars, ws, wbars
    ps = []
    pbars = []
    ws = []
    wbars = []
    """Get particle data in Rest frame of COM as well as weights"""
    for i in range(len(files)):
        p, pbar, w, wbar = PrepareData(files[i] + '.pkl', cuts[i])  # get run 1 data
        ps.append(p)
        pbars.append(pbar)
        ws.append(w)
        wbars.append(wbar)
    
    ps = RearrangeParticleLists(ps)  # switches the ordering of the list, see function for more
    pbars = RearrangeParticleLists(pbars)
    
    p = [np.concatenate(ps[x]) for x in range(5)]  # merge particle data from different files
    pbar = [np.concatenate(pbars[x]) for x in range(5)]
    w = np.concatenate(ws)  # merge weights from different data files
    wbar = np.concatenate(wbars)
    return p, pbar, w, wbar



"""Plots unweighted data and weighted data for a visual comparison"""
def CompWieght(data, x_axis, x_axisW, weights):
    ax = plt.subplot(121)  # left figure
    ax.set_title("(a)")  # set title
    pt.Histogram(data, axis=True, x_axis=x_axis, y_axis='Number of Events')  # plot unweighted histogram
    ax = plt.subplot(122)  # right figure
    ax.set_title("(b)")
    pt.Histogram(data, weights=weights, axis=True, x_axis=x_axisW, y_axis='Weighted Number of Events')  # plot weighted histogram


"""Plots the unweighted and weighted parent particle mass and fits the weighted data"""
def InvariantMassComp(particles, weights):
    inv_mass = particles[0][:, 0]  # get mass of p_0 (asssume already in rest frame)

    plt.subplot(121)  # left figure
    pt.Histogram(inv_mass/1000, axis=True, x_axis='$S(GeV)$', y_axis='number of events')

    plt.subplot(122)  # right figure
    mass, width = pt.BWCurveFit(inv_mass, weights=weights, fit_parameters=[0, 5, 1], legend=True, binWidth=0.01, binNum=50, x_axis='$S(GeV)$', y_axis='weighted number of events', axis=True)  # fits the weighted distributions and returns the fitted masss and decay width
    return mass, width


"""Calculates the A_T, by fitting a distribution to C_T histograms (not accurate)."""
def C_T_Analysis(particles, weights):
    C_T = kin.Scalar_TP(kin.Vector_3(particles[3]), kin.Vector_3(particles[4]), kin.Vector_3(particles[1]))  # calculate scalar triple product

    bins, edges = np.histogram(C_T/1E8, bins=15, weights=weights)  # generate binned data
    x = (edges[:-1] + edges[1:])/2  # gets center of bins

    popt, cov = curve_fit(Gaussian, x, bins, p0=[max(bins), 0, 2])  # fit Gaussian curve to binned data
    popterr = [cov[0, 0]**0.5, cov[1, 1]**0.5, cov[2, 2]**0.5]  # get fit parameter errors

    popt_lower = [popt[0] - popterr[0], popt[1] - popterr[1], popt[2] - popterr[2]]  # lower limit of the fit
    popt_upper = [popt[0] + popterr[0], popt[1] + popterr[1], popt[2] + popterr[2]]  # upper limit of the fit

    expected = Gaussian(x, *popt)  # calculate C_T bins from the fitt
    chisqr = np.sum((bins - expected)**2 / expected)  # get the Chi sqr
    dof = len(bins) - 1  # degrees of freedom of a Gaussian curve

    chisqrRed = chisqr/dof  # reduced chi squared

    x_inter = np.linspace(x[0], x[-1], 500)  # generate interpolated data for plotting
    y = Gaussian(x_inter, *popt)

    y_lower = Gaussian(x_inter, *popt_lower)  # lower bound of fitted curve
    y_upper = Gaussian(x_inter, *popt_upper)  # upper bound of fitted curve


    int1, _ = integrate.quad(Gaussian, 0, max(x), args=(popt[0], popt[1], popt[2]))  # caclaute finite integral (yield)
    int2, _ = integrate.quad(Gaussian, min(x), 0, args=(popt[0], popt[1], popt[2]))

    int1_lower, _ = integrate.quad(Gaussian, 0, max(x), args=(popt_lower[0], popt_lower[1], popt_lower[2]))  # calculate yield of lower bound
    int2_lower, _ = integrate.quad(Gaussian, min(x), 0, args=(popt_lower[0], popt_lower[1], popt_lower[2]))

    int1_upper, _ = integrate.quad(Gaussian, 0, max(x), args=(popt_upper[0], popt_upper[1], popt_upper[2]))  # calculate yield of upper bound
    int2_upper, _ = integrate.quad(Gaussian, min(x), 0, args=(popt_upper[0], popt_upper[1], popt_upper[2]))

    A_T = (int1 - int2) / (int1 + int2)  # calculate A_T from yields

    int1_err = (int1_upper - int1_lower)/2  # error of the yield
    int2_err = (int2_upper - int2_lower)/2

    print(int1_err)
    print(int2_err)

    A_T_lower = (int1_lower - int2_lower) / (int1_lower + int2_lower)  # smallest possible value of A_T given the errors
    A_T_lower = A_T - A_T_lower  # lower bound of A_T
    A_T_upper = (int1_upper - int2_upper) / (int1_upper + int2_upper)  # largest possible value of A_T given the errors
    A_T_upper = A_T_upper - A_T  # upper bound of A_T

    A_T_error = 1/(int1 + int2) * np.sqrt(int1_err**2 * (1-A_T)**2 + int2_err**2 * (1+A_T)**2)  # calculate propagated error

    A = "$A=$" + str(round_to(popt[0], popterr[0])) + "$\\pm$" + str(round_to(popterr[0], popterr[0]))  # string of A to print
    B = "$B=$" + str(round_to(popt[1], popterr[1])) + "$\\pm$" + str(round_to(popterr[1], popterr[1]))  # ...
    C = "$C=$" + str(round_to(popt[2], popterr[2])) + "$\\pm$" + str(round_to(popterr[2], popterr[2]))

    Chi = "$\chi^{2}/ndf =$" + str(round(chisqrRed, 3))  # chi square to print

    label = A + "\n" + B + "\n" + C + "\n" + Chi  # create the label

    plt.bar(x, bins, width=0.2, alpha=0.5)  # plot histogram
    plt.ylim(0)  # set ylimit of the plot
    plt.plot(x_inter, y_lower, color='grey')  # plot lower bound of fit
    plt.plot(x_inter, y, color='red')  # plot best fit
    plt.plot(x_inter, y_upper, color='grey')  # plot upper bound of fit
    plt.title("$A_{T}=$" + str(round_to(A_T, A_T_error)) + "$\\pm$" + str(round_to(A_T_error, A_T_error)))  # display A_T a sthe title
    plt.xlabel("$C_{T}$", fontsize=14)
    plt.ylabel("Weighted Number of Events", fontsize=14)
    plt.annotate(label, (1, 4))  # add the label as text on the plot


"""Calculate the Yield by integrating a fitted Gaussian curve to the invariant mass plots
for various C_T conditions"""
def Yield(data, weights, title, x_axis, Plot=True):
    bins, edges = np.histogram(data, bins=100, weights=weights)  # bin data
    x = (edges[:-1] + edges[1:])/2  # get center of bins

    param_bounds = ([0, 0, 0], [max(bins), max(x), 1])  # set limits for the fit quantities

    popt, cov = curve_fit(Gaussian, x, bins, p0=[1, 1, 1], bounds=param_bounds)  # fit binned data to Gaussian function
    popterr = [cov[0, 0]**0.5, cov[1, 1]**0.5, cov[2, 2]**0.5]  # get the uncertianty in fit parameters

    popt_lower = [popt[0] - popterr[0], popt[1] - popterr[1], popt[2] - popterr[2]]  # get lower and upper limit of fit params
    popt_upper = [popt[0] + popterr[0], popt[1] + popterr[1], popt[2] + popterr[2]]


    x_inter = np.linspace(x[0], x[-1], 500)  # generate interpolated data for plotting
    y = Gaussian(x_inter, *popt)

    y_lower = Gaussian(x_inter, *popt_lower)  # generate extremal fits
    y_upper = Gaussian(x_inter, *popt_upper)

    width = x[1] - x[0]  # width of bins
    integral = np.sqrt(2 * np.pi) * popt[0] * popt[2]/width  # is the yield
    error_fit = integral * ((popterr[0]/popt[0])**2 + (popterr[2]/popt[2])**2)**0.5  # is the error of the yield given the gaussian function is coorect
    error = integral**0.5  # error in yield given a poisson approximation

    """Create text for table, formatted to be used in Latex"""
    A = "|A          | $" + str(round_to(popt[0], popterr[0])) + " \pm " + str(round_to(popterr[0], popterr[0])) + "$"  # Gaussian amplitude
    B = "|B          | $" + str(round_to(popt[1], popterr[1])) + " \pm " + str(round_to(popterr[1], popterr[1])) + "$"  # x-offset
    C = "|C          | $" + str(round_to(popt[2], popterr[2])) + " \pm " + str(round_to(popterr[2], popterr[2])) + "$"  # standard deviation
    E = "|fit error  | " + str(round(error_fit, 2))  # error in the fit
    F = "|yield      | " + str(round(integral, 2))  # yield
    G = "|yield error| " + str(round(error, 2))  # yield error
    label = A + "\n" + B + "\n" + C + "\n" + E + "\n" + F + "\n" + G  # construct the rows of the table
    print("-----------------------")  # start of table
    print("|quantity   | value")  # headers
    print("+-----------+----------")  # divide the cell
    print(label)  # content of the table
    print("-----------------------")  # close the table

    if(Plot is True):
        plt.bar(x, bins, width=width, alpha=0.5, edgecolor="black", linewidth=1)  # plot Histogram
        plt.ylim(0)  # set y limit of plot to zero

        plt.plot(x_inter, y_lower, color='grey')  # lower bound fit
        plt.plot(x_inter, y, color='red')  # best fit
        plt.plot(x_inter, y_upper, color='grey')  # upper bound plot

        plt.title(title)
        plt.xlabel(x_axis, fontsize=14)
        plt.ylabel("Weighted Number of Events", fontsize=14)
    return (integral, error), (popt[0], popterr[0]), (popt[1], popterr[1]), (popt[2], popterr[2])


"""Calculate the asymmetries using the Yield() function"""
def Fit_Analysis(plot=True):
    p, pbar, w, wbar = PrepareData('Data_sig_tos_weights.pkl')  # get run 1 data
    p2, pbar2, w2, wbar2 = PrepareData('Data_sig_tos_weights-Run2.pkl')  # get run 2 data

    p = [np.concatenate((p[x], p2[x])) for x in range(5)]  # merge run 1 and run 2
    pbar = [np.concatenate((pbar[x], pbar2[x])) for x in range(5)]

    w = np.concatenate((w, w2))  # merge run 1 and run 2 weight lists
    wbar = np.concatenate((wbar, wbar2))

    C_T = kin.Scalar_TP(kin.Vector_3(p[3]), kin.Vector_3(p[4]), kin.Vector_3(p[1]))  # Scalar triple products
    C_Tbar = kin.Scalar_TP(kin.Vector_3(pbar[3]), kin.Vector_3(pbar[4]), kin.Vector_3(pbar[1]))  # conjugate Scalar triple products
    mB0 = kin.Mag_4(p[0])/1000  # GeV conversion
    mBbar0 = kin.Mag_4(pbar[0])/1000  # GeV conversion

    """Get masses and weights for various C_T conditions (and -C_Tbar)"""
    mB0_l = mB0[C_T < 0]
    mB0_u = mB0[C_T > 0]
    weights_l = w[C_T < 0]
    weights_u = w[C_T > 0]

    mBbar0_l = mBbar0[-C_Tbar < 0]
    mBbar0_u = mBbar0[-C_Tbar > 0]
    weightsbar_l = wbar[-C_Tbar < 0]
    weightsbar_u = wbar[-C_Tbar > 0]

    """Calculate the Yields and plot if set to true"""
    plt.subplot(221)
    N_l, _, _, _ = Yield(mB0_l, weights_l, "(a)", "$m_{B^{0}}(C_{T} < 0)$", plot)

    plt.subplot(222)
    N_u, _, _, _ = Yield(mB0_u, weights_u, "(b)", "$m_{B^{0}}(C_{T} > 0)$", plot)

    plt.subplot(223)
    N_l_CP, _, _, _ = Yield(mBbar0_l, weightsbar_l, "(c)", "$m_{\\bar{B}^{0}}(-\\bar{C}_{T} < 0)$", plot)

    plt.subplot(224)
    N_u_CP, _, _, _ = Yield(mBbar0_u, weightsbar_u, "(d)", "$m_{\\bar{B}^{0}}(-\\bar{C}_{T} > 0)$", plot)

    A_T = kin.A_T(N_u[0], N_l[0])  # calculate A_T from yields

    A_Tbar = kin.A_T(N_l_CP[0], N_u_CP[0])  # calculate conjugate asymmetry

    A_CP = kin.A_CP(A_T, A_Tbar)

    print("A_T = " + str(A_T))
    print("A_Tbar = " + str(A_Tbar))
    print("A_CP = " + str(A_CP))


"""Calculate parity asymmetry by summing the weights"""
def P_Asym(w, C_T):
    w_u = w[C_T > 0]  # get C_T condition
    w_l = w[C_T < 0]

    N_u = np.sum(w_u)
    print("yield_upper: " + str(N_u))
    N_l = np.sum(w_l)
    print("yield_lower: " + str(N_l))

    return kin.A_T(N_l, N_u)  # return the P asymmetry


"""Analyse the data by summing the weights to get the yields"""
def Sum_Analysis(names, cuts):
    global p, pbar, w, wbar
    p, pbar, w, wbar = ReadRealDataMulti(names, cuts)  # get particles data in the rest frame of COM for multiple event files, with conjugate decays tagged.

    C_T = kin.Scalar_TP(kin.Vector_3(p[3]), kin.Vector_3(p[4]), kin.Vector_3(p[1]))
    C_Tbar = -kin.Scalar_TP(kin.Vector_3(pbar[3]), kin.Vector_3(pbar[4]), kin.Vector_3(pbar[1]))


    A_T = P_Asym(w, C_T)  # get A_T by summing the weights to get the yield
    A_Tbar = P_Asym(wbar, C_Tbar)

    A_CP = kin.A_CP(A_T, A_Tbar)

    Asyms = [A_T, A_Tbar, A_CP]
    text = ["A_T   ", "A_Tbar", "A_CP  "]  # first column of the table
    """Print a table to display the asymmetries and errors. Formatted to use in Latex"""
    print("---------Sum all the weights--------")  # start of table
    """Prints each asymmetry value and error"""
    for i in range(len(Asyms)):
        Asym = Asyms[i]
        print(text[i] + ": $" + str(round(Asym[0], 3)) + " \pm " + str(round(Asym[1], 3)) + "$")
    print("------------------------------------")  # end of table

    print(np.sum(w))  # yield
    print(np.sum(wbar))  # conjugate yield
    print(np.sum(w) + np.sum(wbar))  # total yield



"""Anaylsis Asuuming the signal near the resonance contains no background"""
def NoWeightAnalysis(names, cuts, plot=True):
    p, pbar, _, _ = ReadRealDataMulti(names, cuts)

    C_T = kin.Scalar_TP(kin.Vector_3(p[3]), kin.Vector_3(p[4]), kin.Vector_3(p[1]))
    C_Tbar = -kin.Scalar_TP(kin.Vector_3(pbar[3]), kin.Vector_3(pbar[4]), kin.Vector_3(pbar[1]))
    m = kin.Mag_4(p[0])
    mbar = kin.Mag_4(pbar[0])

    _range = [5200 , 5400]  # range of masses near the resonance
    mRes = []
    C_TRes = []
    """gets all invariant masses near the resonance"""
    for i in range(len(m)):
        if m[i] > _range[0] and m[i] < _range[1]:
            mRes.append(m[i])
            C_TRes.append(C_T[i])

    mRes = np.array(mRes)  # convert to numpy arrays
    C_TRes = np.array(C_TRes)

    mbarRes = []
    C_TbarRes = []
    for i in range(len(mbar)):
        if mbar[i] > _range[0] and mbar[i] < _range[1]:
            mbarRes.append(mbar[i])
            C_TbarRes.append(C_Tbar[i])

    mbarRes = np.array(mbarRes)
    C_TbarRes = np.array(C_TbarRes)


    A_T = kin.TP_Amplitude(C_TRes)
    A_Tbar = kin.TP_Amplitude(C_TbarRes)

    A_CP = kin.A_CP(A_T, A_Tbar)

    """See Sum_Analysis() for details"""
    Asyms = [A_T, A_Tbar, A_CP]
    text = ["A_T   ", "A_Tbar", "A_CP  "]
    print("----Sum weights near B0 resonance---")
    for i in range(len(Asyms)):
        Asym = Asyms[i]
        print(text[i] + ": $" + str(round(Asym[0], 3)) + " \pm " + str(round(Asym[1], 3)) + "$")
    print("------------------------------------")

    """Plot the unweighted data near the resonance"""
    if plot is True:
        y_axis="Event Number"  #y label
        plt.subplot(221)  # first subplot in a 2x2 grid of figures
        data = mRes[C_TRes > 0] # plot this C_T condition
        pt.Histogram(data, bins=50, axis=True, x_axis="$m_{B^{0}}(C_{T} > 0)$", y_axis=y_axis)

        plt.subplot(222)
        data = mRes[C_TRes < 0]
        pt.Histogram(data, bins=50, axis=True, x_axis="$m_{B^{0}}(C_{T} < 0)$", y_axis=y_axis)

        plt.subplot(223)
        data = mbarRes[-C_TbarRes > 0]
        pt.Histogram(data, bins=50, axis=True, x_axis="$m_{\\bar{B}^{0}}(-\\bar{C}_{T} > 0)$", y_axis=y_axis)

        plt.subplot(224)
        data = mbarRes[-C_TbarRes < 0]
        pt.Histogram(data, bins=50, axis=True, x_axis="$m_{\\bar{B}^{0}}(-\\bar{C}_{T} < 0)$", y_axis=y_axis)


"""Main Body"""
files = ["tos_Run1", "tis_Run1", "tos_Run2", "tis_Run2"]
#files = ["Data_sig_tos_weights-Run1", "Data_sig_tis_weights-Run1", "Data_sig_tos_weights-Run2", "Data_sig_tis_weights-Run2"]
cuts = [0.9968, 0.9988, 0.9693, 0.9708]

#p, pbar, w, wbar = ReadRealDataMulti(files, cuts)

Sum_Analysis(files, cuts)
