# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:38:04 2019

@author: bhull
"""

import Kinematic as kin  # my own module
import numpy as np
from numpy import random_intel  # This is super awesome! use instead of np.random
import winsound
import matplotlib.pyplot as plt
from scipy import stats

"""Will generate MC data for a value and its uncertianty for a gaussian distribution"""
def Generate_MC(value, error, n):
    n = int(n)
    dist = random_intel.normal(value, error , n)
    return dist

def MC_Sim_Amplitude(lower, upper, n):
    n = int(n)
    means = [lower, upper]

    dist_l = random_intel.normal(means[0], np.sqrt(lower) , n)
    dist_u = random_intel.normal(means[1], np.sqrt(upper), n)
    #dist_l = random_intel.poisson(means[0], n)
    #dist_u = random_intel.poisson(means[1], n)

    dist = (dist_u - dist_l) / (dist_u + dist_l)
    A = np.mean(dist)
    error = np.std(dist)
    return dist, A, error

def NormaliseCheck(P, confidence):
    _, p = stats.normaltest(P)

    # if p > confidence distribution  isn't gaussian
    if(len(np.argwhere(p > confidence)) > 0):
        print('one of the diestibutions isnt gaussian.')
    else:
        print('can be modelled as as gaussian for condifence:', confidence)
    return p


"""Uses monte carlo simulation to get uncertainty and value of TP amplitude"""
def MC_TP_Amplitude(phi, n):
    lower = []
    upper = []

    for i in range(len(phi)):
        if phi[i] < 0:
            lower.append(phi[i])
        else:
            upper.append(phi[i])

    dist, A, error = MC_Sim_Amplitude(len(lower), len(upper), n)
    return dist, A, error


"""Estimate minimum amount of asymmetry in order to see P violation"""
def MC_Sim_BestCase(min_, max_, iter_, samples, confidence=1, n=int(1E7)):
    diff = np.linspace(min_, max_, iter_)
    lower = samples/2 - diff
    upper = samples - lower
    print('mean | error')
    for i in range(len(diff)):
        dist, A, error = MC_Sim_Amplitude(lower[i], upper[i], n)
        print(A, '|' , error)
        if((confidence * error) < abs(A)):
            break

    if len(dist) > 0:
        plt.hist(dist, 50, label='$A_{Tmin}$', density=True)
        print('difference needed in', samples, 'samples is at least:', diff[i])


"""will sample through generated data to get a rough order of magnitude of the sample number needed to see P violation for a given amount"""
def MC_Min(min_, max_, iter_, A_CT, confidence=1, n=int(1E7)):
    samples = np.linspace(min_, max_, iter_)
    lower =  0.5 * samples * (1 - A_CT)
    upper = 0.5 * samples * (1+ A_CT)
    print('mean | error')

    for i in range(len(samples)):
        dist, A, error = MC_Sim_Amplitude(lower[i], upper[i], n)
        print(A, '|' , error)
        if((confidence * error) < abs(A)):
            break

    return i, A, samples[i], dist


"""Search algorithm for MC_Min"""
def Search(_min, _max, _iter, A_CT, confidence=1, n=int(1E7), maxSearch=10):
    print('working...')
    win = False
    search = 1
    while win == False:
        if(search == maxSearch):
            print('maximum search limit reached, convergence failed or increase maxSearch.')
            break

        score, A_CT_sim, samplesRequired, dist = MC_Min(_min, _max, _iter, A_CT, confidence, n)
        score += 1
        if score == _iter:
            win == False
            print('upper bound needs to be pushed up')
            _min = _max
            _max *= 10
            search += 1
        if score < _iter:
            win = True
    print('done!')
    winsound.MessageBeep()
    weights = np.ones_like(dist)/float(len(dist))
    plt.hist(dist, 50, label='$A_{Tideal}$', weights=weights)
    print('events needed to see asymmetry', A_CT_sim, 'with significance is', samplesRequired)


"""will sample through generated data to get a rough order of magnitude of the sample number needed to see CP violation for a given amount"""
def MC_Min_CP(min_, max_, iter_, A_T, A_Tbar, confidence=1, n=int(1E7)):
    samples = np.linspace(min_, max_, iter_)
    l =  0.5 * samples * (1 - A_T[0])
    u = 0.5 * samples * (1+ A_T[0])
    lbar =  0.5 * samples * (1 - A_Tbar[0])
    ubar = 0.5 * samples * (1+ A_Tbar[0])

    print('mean | error')

    for i in range(len(samples)):
        dist, _, _ = MC_Sim_Amplitude(l[i], u[i], n)
        distbar, _, _ = MC_Sim_Amplitude(lbar[i], ubar[i], n)
        A_CP = 0.5 * (dist - distbar)
        mean = np.mean(A_CP)
        error = np.std(A_CP)
        print(mean, '|' , error)
        if((confidence * error) < abs(mean)):
            break

    return i, mean, samples[i], dist


"""Search algorithm for MC_Min_CP"""
def Search_CP(_min, _max, _iter, A_T, A_Tbar, confidence=1, n=int(1E7), maxSearch=10):
    print('working...')
    win = False
    search = 1
    while win == False:
        if(search == maxSearch):
            print('maximum search limit reached, convergence failed or increase maxSearch.')
            break

        score, A_CP, samplesRequired, dist = MC_Min_CP(_min, _max, _iter, A_T, A_Tbar, confidence, n)
        score += 1
        if score == _iter:
            win == False
            print('upper bound needs to be pushed up')
            _min = _max
            _max *= 10
            search += 1
        if score < _iter:
            win = True
    print('done!')
    winsound.MessageBeep()
    weights = np.ones_like(dist)/float(len(dist))
    plt.hist(dist, 50, weights=weights)
    print('events needed to see asymmetry', A_CP, 'with significance is', samplesRequired)
    return samplesRequired