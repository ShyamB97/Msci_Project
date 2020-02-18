"""
Created on %(date)

@author: % Shyam Bhuller
"""
import Kinematic as kin  # my own module
import StatTools as st  # my own module
import Plotter as pt  # my own module
import DataManager as dm  # my own module
import uproot
#import phasespace as pp
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import stats
from scipy.optimize import curve_fit
from math import log10, floor
from scipy.special import factorial
import glob


def round_to(x, y):
    return round(x, -int(floor(log10(abs(y)))))


def Rearrange(particles):
    for particle in particles:
        p = particles[particle]
        p = np.hsplit(p, 4)
        p = np.stack([p[3], p[0], p[1], p[2]], 1)[:, :, 0]
        particles[particle] = p
    return particles


def BW(x, A, M, T):
    gamma = np.sqrt(M**2 * (M**2 + T**2))
    k = A * ((2 * 2**0.5)/np.pi) * (M * T * gamma)/((M**2 + gamma)**0.5)
    return k/((x**2 - M**2)**2 + (M*T)**2)


def BWMulti(x, A1, M1, T1, A2, M2, T2, A3, M3, T3):#, M4, T4, M5, T5):
    return BW(x, A1, M1, T1) + BW(x, A2, M2, T2) + BW(x, A3, M3, T3)# + BW(x, M4, T4) + BW(x, M5, T5)

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


"""computes dalitz plot of D -> K_s pi+ pi-, shows no features of the phasespace scene in real life so phase space is uniform"""
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


"""Fits Breit Wigner curve to invariant mass plots and returns resonance mass and lifetime in MeV"""
def BWCurve(E, plot=False):
    """E must be in units of GeV"""
    hist, bins = np.histogram(E, bins=50, density=1)
    x = (bins[:-1] + bins[1:])/2  # center of bins

    popt, cov = curve_fit(BW, x, hist, p0=[1, 1])

    x_inter = np.linspace(x[0], x[-1], 500)
    y = BW(x_inter, *popt)

    _, p = stats.chisquare(hist, BW(x, *popt))

    p = stats.norm.ppf(p)

    if(plot is True):
        plt.bar(x, hist, 0.01)
        plt.plot(x_inter, y, color='r')
        plt.vlines(popt[0], min(y), max(y), linewidth=2, linestyle="--")
        plt.hlines(max(y)/2, -popt[1]/2 + popt[0], popt[1]/2 + popt[0], linewidth=2, linestyle="--")
        plt.xlabel("$E_{K^{+}\pi^{-}}(GeV)$", fontsize=14)
        plt.ylabel("Normlaised count", fontsize=14)

    popt *= 1000
    cov *= 1000
    return [popt[0], cov[0, 0]], [popt[1], cov[1, 1]]


"""Check if you saved a file already, will contain scores"""
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


"""Checks how A_CP varies with sample size"""
def CPConvergenceTest():
    value = []
    error = []
    eventFiles = dm.GetFileNames('\samples')
    eventFiles_CP = dm.GetFileNames('\samplesCP')
    for i in range(10):
        p = dm.AmpGendf(eventFiles[i], False)
        pbar = dm.AmpGendf(eventFiles_CP[i], True)

        C_T = kin.Scalar_TP(kin.Vector_3(p['p_3']), kin.Vector_3(p['p_4']), kin.Vector_3(p['p_1']))
        C_Tbar = -kin.Scalar_TP(kin.Vector_3(pbar['p_3']), kin.Vector_3(pbar['p_4']), kin.Vector_3(pbar['p_1']))

        A_T = kin.TP_Amplitude(C_T)
        A_Tbar = kin.TP_Amplitude(C_Tbar)

        A_CP = kin.A_CP(A_T, A_Tbar)
        value.append(A_CP[0])
        error.append(A_CP[1])

    pt.ErrorPlot([np.linspace(1, 10, 10), value], axis=True, y_error=error, x_axis="Number of Events ($10^{5}$)", y_axis="$\mathcal{A}_{CP}$")


"""Checks if the seed of the generator significantly affects A_CP"""
def Seed_test():
    fileNames = dm.GetFileNames('\seed_test')
    events = fileNames[0:5]
    events_CP = fileNames[5:10]

    value = []
    error = []
    for i in range(5):
        p = dm.AmpGendf(events[i], False)
        pbar = dm.AmpGendf(events_CP[i], True)

        C_T = kin.Scalar_TP(kin.Vector_3(p['p_3']), kin.Vector_3(p['p_4']), kin.Vector_3(p['p_1']))
        C_Tbar = -kin.Scalar_TP(kin.Vector_3(pbar['p_3']), kin.Vector_3(pbar['p_4']), kin.Vector_3(pbar['p_1']))

        A_T = kin.TP_Amplitude(C_T)
        A_Tbar = kin.TP_Amplitude(C_Tbar)

        A_CP = kin.A_CP(A_T, A_Tbar)
        value.append(A_CP[0])
        error.append(A_CP[1])

    pt.ErrorPlot([np.linspace(1, 5, 5), value], axis=True, y_error=error, x_axis="Iteration", y_axis="$\mathcal{A}_{CP}$")


def PWaveAmp_test():
    fileNames = dm.GetFileNames('\Amp_test')
    rel_Amp = []
    for i in range(len(fileNames)):
        string = fileNames[i][16:-5]
        num = float(string)
        rel_Amp.append(num)

    zipped_1 = zip(rel_Amp, fileNames)
    fileNames = [x for _, x in sorted(zipped_1)]
    rel_Amp = sorted(rel_Amp)

    value = []
    error = []
    for i in range(len(rel_Amp)):
        p = dm.AmpGendf(fileNames[i], False)
        C_T = kin.Scalar_TP(kin.Vector_3(p['p_3']), kin.Vector_3(p['p_4']), kin.Vector_3(p['p_1']))
        A_T = kin.TP_Amplitude(C_T)
        value.append(A_T[0])
        error.append(A_T[1])

    pt.ErrorPlot([rel_Amp, value], axis=True, y_error=error, x_axis="relative P-wave amplitudes", y_axis="$A_{T}$")


"""Main Body"""
#p = dm.AmpGendf('B_times_test2.root', False)
#pbar = dm.AmpGendf('B_0.1M_S10.root', True)

#C_T = kin.Scalar_TP(kin.Vector_3(p['p_3']), kin.Vector_3(p['p_4']), kin.Vector_3(p['p_1']))
#C_Tbar = -kin.Scalar_TP(kin.Vector_3(pbar['p_3']), kin.Vector_3(pbar['p_4']), kin.Vector_3(pbar['p_1']))

#A_T = kin.TP_Amplitude(C_T)
#A_Tbar = kin.TP_Amplitude(C_Tbar)

#A_CP = kin.A_CP(A_T, A_Tbar)

#sigma = abs(A_CP[0]/A_CP[1])

samples = np.array([1E4, 4E4, 9E4, 16E4, 25E4, 36E4, 49E4, 64E4, 81E4])
times = np.array([34, 86, 172, 301, 466, 699, 876, 1130, 1461])

grad, _ = np.polyfit(samples, times, 1)

samplestoTest = np.linspace(int(1E2), int(1E3), 10)**2

time_take = (grad *samplestoTest)/60
total_time = np.sum(time_take)
plt.plot(samples, times, marker='o', markersize=5)

"""
intervals = [1, 3, 5]
labels = []
for i in range(len(intervals)):
    samples = st.Search_CP(3E6, 1E7, 10, A_T, A_Tbar, intervals[i])
    samples = samples/int(1E6)
    label = str(intervals[i]) + '$\\sigma$| events = ' + str(samples) + '$\\times10^{6}$'
    labels.append(label)

plt.legend(labels, fontsize=14)
plt.xlabel("MC $\mathcal{A}_{CP}$ distribution", fontsize=14)
plt.ylabel("weighted events", fontsize=14)
plt.tight_layout()
"""




