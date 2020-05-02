"""
Created on 30 ‎October ‎2019

@author: Shyam Bhuller

@Description: Is a Multipurpose script used for Analysis For the Msci Project. Has no real Objective and
serves as a test bed for larger functions which I have create over the duration of the project.
"""
import Kinematic as kin  # vecotrised 4-vector kineamtics
import Plotter as pt  # generic plotter with consistent formatting and curve fitting
import DataManager as dm  # handles data opened from data files
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from math import log10, floor

"""Round x to the 1st significant figure of y"""
def round_to(x, y):
    return round(x, -int(floor(log10(abs(y)))))


"""Messes around with the order of the particle dictionary and removes the parent particle."""
def Rearrange(particles):
    for particle in particles:
        p = particles[particle]
        p = np.hsplit(p, 4)
        p = np.stack([p[3], p[0], p[1], p[2]], 1)[:, :, 0]
        particles[particle] = p
    return particles


"""Breit Wigner distribution with and ampltitude"""
def BW(x, A, M, T):
    # see https://en.wikipedia.org/wiki/Relativistic_Breit%E2%80%93Wigner_distribution for its definition
    gamma = np.sqrt(M**2 * (M**2 + T**2))  # formula is complex, so split it into multiple terms
    k = A * ((2 * 2**0.5)/np.pi) * (M * T * gamma)/((M**2 + gamma)**0.5)
    return k/((x**2 - M**2)**2 + (M*T)**2)


"""A linear sum of 3 Breit Wigner distributions, used to fit data with multiple resonances."""
def BWMulti(x, A1, M1, T1, A2, M2, T2, A3, M3, T3):
    return BW(x, A1, M1, T1) + BW(x, A2, M2, T2) + BW(x, A3, M3, T3)


"""This function tried using a Tensorflow event generator in python. No longer used"""
"""
def Phasedf(n):
    n = int(n)
    B0_Mass = 5279.4
    D0_Mass = 1864.84
    Kstar_Mass = 493.677
    Pi_Mass = 139.57018

    weights, particles = pp.nbody_decay(B0_Mass, [D0_Mass, D0_Mass, Kstar_Mass, Pi_Mass]).generate(n_events=n)

    # rearrange to used format for 4-vectors.
    for particle in particles:
        p = particles[particle]
        p = np.hsplit(p, 4)
        p = np.stack([p[3], p[0], p[1], p[2]], 1)[:, :, 0]
        particles[particle] = p
    B0 = np.transpose(np.reshape(np.repeat([B0_Mass, 0, 0, 0], n), [4, n]))
    particles = {'p_0': B0, 'p_1': particles['p_0'], 'p_2': particles['p_1'], 'p_3': particles['p_2'], 'p_4': particles['p_3']}
    return particles
"""


"""Computes dalitz plot of D -> K_s pi+ pi-, shows no features of the phasespace scene in real life so phase space is uniform.
No longer used."""
"""
def Uniform2():
    # second attempt to check if generated decays are uniform in phasespace

    D0_Mass = 1864.84
    Ks_Mass = 497.611
    Pi_Mass = 139.57018

    weights, particles = pp.nbody_decay(D0_Mass, [Ks_Mass, Pi_Mass, Pi_Mass]).generate(n_events=10000)
    particles = Rearrange(particles)
    m_Kspp = kin.Mag_4(particles["p_0"] + particles["p_1"]) # ks pi+
    m_Kspm = kin.Mag_4(particles["p_0"] + particles["p_2"]) # ks pi-

    plt.scatter(m_Kspp, m_Kspm, s=1)

    #plt.hist(m_Kspp, 50)
    #plt.hist(m_Kspm, 50)
"""


"""Check if you saved a file already, will contain scores. No longer used."""
"""
def Data_Save(repeats):
    samples = 1000
    data = []
    for i in range(repeats):
        score = 0
        particles = Phasedf(1E6)
        particles = [particles['p_0'], particles['p_1'], particles['p_2'], particles['p_3'], particles['p_4']]
        particles = np.array(particles)
        for j in range(int(1E6/samples)):
            part = particles[:, samples*j:samples*(j+1), :]
            TP = kin.Scalar_TP(part[1, :, 1:4], part[2, :, 1:4], part[3, :, 1:4])
            A_CP, error = kin.TP_Amplitude(TP)
            if abs(A_CP) > error:
                score += 1
        data.append(score)
        print("\r progress: "+str(round((i+1)/repeats * 100, 2)), end="")
    return data
"""


"""Fits Breit Wigner curve to invariant mass plots and returns resonance mass and lifetime in MeV"""
def BWCurve(E, plot=False):
    """E must be in units of GeV"""
    hist, bins = np.histogram(E, bins=50, density=1)  # gets bins invariant masses
    x = (bins[:-1] + bins[1:])/2  # center of bins

    popt, cov = curve_fit(BW, x, hist, p0=[1, 1])  # perform least squares fit of invariant mass

    x_inter = np.linspace(x[0], x[-1], 500)  # generate interpolated data for plotting
    y = BW(x_inter, *popt)

    _, p = stats.chisquare(hist, BW(x, *popt))  # caculate a chisquare test

    p = stats.norm.ppf(p)  # get goodness of fit in sigma

    """Plots histogram and fitted data as well as calculated fit values"""
    if(plot is True):
        plt.bar(x, hist, 0.01)  # plot histogram
        plt.plot(x_inter, y, color='r')  # plot fit
        plt.vlines(popt[0], min(y), max(y), linewidth=2, linestyle="--")  # plot invariant mass location
        plt.hlines(max(y)/2, -popt[1]/2 + popt[0], popt[1]/2 + popt[0], linewidth=2, linestyle="--")  # plot width
        plt.xlabel("$E_{K^{+}\pi^{-}}(GeV)$", fontsize=14)
        plt.ylabel("Normlaised count", fontsize=14)

    popt *= 1000  # convert GeV to MeV
    cov *= 1000
    return [popt[0], cov[0, 0]], [popt[1], cov[1, 1]]


"""Checks how A_CP varies with sample size"""
def CPConvergenceTest():
    value = []
    error = []
    eventFiles = dm.GetFileNames('\Samples')  # get filenames
    eventFiles_CP = dm.GetFileNames('\Samples_CP')

    for i in range(len(eventFiles)):
        p = dm.AmpGendf(eventFiles[i], False)  # generate particle data
        pbar = dm.AmpGendf(eventFiles_CP[i], True) # generate CP particle data

        C_T = kin.Scalar_TP(kin.Vector_3(p['p_3']), kin.Vector_3(p['p_4']), kin.Vector_3(p['p_1']))  # calcualtes scalar triple product
        C_Tbar = -kin.Scalar_TP(kin.Vector_3(pbar['p_3']), kin.Vector_3(pbar['p_4']), kin.Vector_3(pbar['p_1']))# -sign for parity flip

        A_T = kin.TP_Amplitude(C_T)  # calculate parity asymmetries
        A_Tbar = kin.TP_Amplitude(C_Tbar)

        A_CP = kin.A_CP(A_T, A_Tbar)  # calculate A_CP
        value.append(A_CP[0])
        error.append(A_CP[1])

    pt.ErrorPlot([np.linspace(1, 10, len(eventFiles)), value], axis=True, y_error=error, x_axis="Number of Events ($10^{5}$)", y_axis="$\mathcal{A}_{CP}$")  # plots data


"""Checks if the seed of the generator significantly affects A_CP"""
def Seed_test():
    fileNames = dm.GetFileNames('\seed_test')  # get filenames
    events = fileNames[0:5]  # split the dataset in half
    events_CP = fileNames[5:10]  # make this half CP data

    value = []
    error = []
    for i in range(5):
        p = dm.AmpGendf(events[i], False)  # generate particle data
        pbar = dm.AmpGendf(events_CP[i], True)

        C_T = kin.Scalar_TP(kin.Vector_3(p['p_3']), kin.Vector_3(p['p_4']), kin.Vector_3(p['p_1']))  # calcualtes scalar triple product
        C_Tbar = -kin.Scalar_TP(kin.Vector_3(pbar['p_3']), kin.Vector_3(pbar['p_4']), kin.Vector_3(pbar['p_1']))

        A_T = kin.TP_Amplitude(C_T)  # calculate parity asymmetries
        A_Tbar = kin.TP_Amplitude(C_Tbar)

        A_CP = kin.A_CP(A_T, A_Tbar)  # calculate A_CP
        value.append(A_CP[0])
        error.append(A_CP[1])

    pt.ErrorPlot([np.linspace(1, 5, 5), value], axis=True, y_error=error, x_axis="Iteration", y_axis="$\mathcal{A}_{CP}$")  # plots data


"""Same as CPCOnvergenceTest, but does for all asymmetries and is compatible for analysing
much larger sample sizes and for events with random seeds."""
def Convergence_test2():
    direc = "\Samples"  # direectory to look for
    direc_CP = direc + "_CP"  # directory for conjugate events
    samples = ["1K", "10K", "100K", "1000K", "10000K"]  # folder names


    A_Ts = []
    A_Tbars = []
    A_CPs = []
    """Will go through each folder and will calculate the asymmetry for a given event size.
    Does this for each seed and merges the data into one list"""
    for i in range(len(samples)):
        filenames = dm.GetFileNames(direc+"\\"+samples[i])  # get ROOT file for the particular sample
        filenames_CP = dm.GetFileNames(direc_CP+"\\"+samples[i]+"_CP")  # .. conjugate sample
        C_T = []
        C_Tbar = []
        """Opens the jth files for the regular and conjugate sample and computed C_T"""
        for j in range(len(filenames)):
            p = dm.AmpGendf(filenames[j], False)  # gets data from the file
            pbar = dm.AmpGendf(filenames_CP[j], True)  # gets conjugate data
            tmp = kin.Scalar_TP(kin.Vector_3(p["p_3"]), kin.Vector_3(p["p_4"]), kin.Vector_3(p["p_1"]))  # C_T
            C_T.append(tmp)  # add to the list
            tmp = -kin.Scalar_TP(kin.Vector_3(pbar["p_3"]), kin.Vector_3(pbar["p_4"]), kin.Vector_3(pbar["p_1"]))  # -C_Tbar
            C_Tbar.append(tmp)


        C_T = np.hstack(C_T)  # merges the data from each individual seed
        C_Tbar = np.hstack(C_Tbar)

        A_T = kin.TP_Amplitude(C_T)
        A_Tbar = kin.TP_Amplitude(C_Tbar)

        A_CPs.append(kin.A_CP(A_T, A_Tbar))  # add asymmetries to a list to save
        A_Ts.append(A_T)
        A_Tbars.append(A_Tbar)
        print(i)

    np.save("Convergence_test/A_T", A_Ts)  # save calculated data into a file for later use
    np.save("Convergence_test/A_Tbar", A_Tbars)
    np.save("Convergence_test/A_CP", A_CPs)


"""Plots asymmetries for the data calulated by Convergence_test2"""
def PlotConvergence():
    global A_T, A_Tbar, A_CP
    x = [1E3, 1E4, 1E5, 1E6, 1E7]  # number of events per sample
    A_T = np.load("Convergence_Test/A_T.npy")  # load the numpy data
    A_Tbar = np.load("Convergence_Test/A_Tbar.npy")
    A_CP = np.load("Convergence_Test/A_CP.npy")

    Asyms = [A_T, A_Tbar, A_CP]  # list of data to plot
    labels = ["$A_{T}$", "$\\bar{A}_{T}$", "$\\mathcal{A}_{CP}$"]  # y labels

    loc = 131  # initial figure location
    j = 0  # which figure to plot to - 1
    """Will plot each asymmetry in a figure"""
    for Asym in Asyms:
        plt.subplot(loc+j)  # assign figure
        mean = [Asym[i][0] for i in range(len(Asym))]  # get the mean values
        error = [Asym[i][1] for i in range(len(Asym))]  # get the errors
        pt.ErrorPlot((x, mean), y_error=error, x_axis="Number of Events", axis=True, y_axis=labels[j])  # plot with error bars
        plt.xscale("log")  # switch to a log scale on x
        j+=1  # move on the the next figure


"""Main Body"""
"""
p = dm.AmpGendf("Dto4_Body.root")

pbar = dm.AmpGendf("CP_test.root", CP=True)

C_T = kin.Scalar_TP(kin.Vector_3(p['p_3']), kin.Vector_3(p['p_4']), kin.Vector_3(p['p_1']))
C_Tbar = -kin.Scalar_TP(kin.Vector_3(pbar['p_3']), kin.Vector_3(pbar['p_4']), kin.Vector_3(pbar['p_1']))

A_T = kin.TP_Amplitude(C_T)
A_Tbar = kin.TP_Amplitude(C_Tbar)

A_CP = kin.A_CP(A_T, A_Tbar)

print(A_T)
print(A_Tbar)
print(A_CP)
"""
PlotConvergence()