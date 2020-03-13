# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:38:04 2019

@author: Shyam Bhuller

@Desciption: Contains all code involved in analysis to do with Sensitivity studies
"""
import numpy as np
import winsound
import matplotlib.pyplot as plt
from scipy import stats


"""Will generate a Gaussian distribution for a value and its uncertianty"""
def Generate_MC(value, error, n):
    return np.random.normal(value, error , int(n))


"""Calculate A_T by randmoly sampling N(C_T > 0) and N(C_T < 0) and generating a distribution of A_T from this"""
def MC_Sim_A_T(lower, upper, n):
    means = [lower, upper]

    dist_l = Generate_MC(means[0], np.sqrt(lower) , n)  # get n random samples in Gaussain distribution
    dist_u = Generate_MC(means[1], np.sqrt(upper), n)

    dist = (dist_u - dist_l) / (dist_u + dist_l)  # calculate A_T distribution
    A_T = np.mean(dist)  # mean value of A_T
    error = np.std(dist)  # error of A_T
    return dist, A_T, error


"""Check if a particular dataset can be modelled as a Gaussain"""
def NormaliseCheck(data, confidence):
    _, p = stats.normaltest(data)  # generate p value of normal distribution for the data

    # if p > confidence distribution  isn't gaussian
    if(len(np.argwhere(p > confidence)) > 0):
        print('one of the diestibutions isnt gaussian.')
    else:
        print('can be modelled as as gaussian for condifence:', confidence)
    return p


"""Calcualte P asymmetry byt randomly sampling data from a distribution"""
def MC_A_T(C_T, n):
    lower = []  # C_T < 0
    upper = []  # C_T > 0

    """Get the data for the C_T conditions"""
    for i in range(len(C_T)):
        if C_T[i] < 0:
            lower.append(C_T[i])
        else:
            upper.append(C_T[i])

    dist, A_T, error = MC_Sim_A_T(len(lower), len(upper), n)  # calculate A_T by random sampling
    return dist, A_T, error


"""Estimate minimum amount of asymmetry in order to see P violation"""
def MC_Sim_BestCase(min_, max_, iter_, samples, confidence=1, n=int(1E7)):
    diff = np.linspace(min_, max_, iter_)  # create sets of asymmetry values to try
    lower = samples/2 - diff  # get C_T < 0 for the various A_Ts
    upper = samples - lower  # get C_T > 0

    print('mean | error')
    """Goes through each A_T and checks if A_T shows significant asymmetry"""
    for i in range(len(diff)):
        dist, A_T, error = MC_Sim_A_T(lower[i], upper[i], n)  # calculate distributions of A_T by random sampling
        print(A_T, '|' , error)
        """if the uncertainty up to a confidence is less than the mean value, we have found P violation"""
        if((confidence * error) < abs(A_T)):
            break
        
    """Plot distriubution for visual example of the asymmetry needed for the given sample size"""
    if len(dist) > 0:
        plt.hist(dist, 50, label='$A_{Tmin}$', density=True)
        print('difference needed in', samples, 'samples is at least:', diff[i])


"""will sample through generated data to get a rough order of magnitude of the sample number needed to see P violation for a given amount"""
def MC_Min_A_T(min_, max_, iter_, A_T, confidence=1, n=int(1E7)):
    samples = np.linspace(min_, max_, iter_)  # get a range of sample numbers to try out
    lower =  0.5 * samples * (1 - A_T)  # get C_T < 0
    upper = 0.5 * samples * (1+ A_T)  # get C_T > 0
    print('mean | error')
    """Goes through each sample size and checks if A_T shows significant asymmetry"""
    for i in range(len(samples)):
        dist, mean, error = MC_Sim_A_T(lower[i], upper[i], n)
        print(mean, '|' , error)
        if((confidence * error) < abs(mean)):
            break

    return i, mean, samples[i], dist


"""Search algorithm for MC_Min, it will search through a range of samples until either the maximum search limit
is reached, or significant asymmetry is found. Does so by adjusting the range of values to try."""
def Search_A_T(_min, _max, _iter, A_CT, confidence=1, n=int(1E7), maxSearch=10):
    print('working...')
    win = False  # we havn't found asymmetry yet
    search = 1  # this is the first search we are trying
    """While no asymmetry has been found, keep looking"""
    while win == False:
        """Dont want to run the code forever"""
        if(search == maxSearch):
            print('maximum search limit reached, convergence failed or increase maxSearch.')
            break

        score, A_CT_sim, samplesRequired, dist = MC_Min_A_T(_min, _max, _iter, A_CT, confidence, n)  # calls MC_min and gets the results, see function for more detail
        score += 1  # add the score
        """if all the samples we gave showed no asymmetry, we need to push the upper bound"""
        if score == _iter:
            win == False
            print('upper bound needs to be pushed up')
            _min = _max  # set minimum sample to the previous maximum
            _max *= 10  # det the maximum sample to 10x the previous
            search += 1  # one search has been completed
        """if the score is less than the number of samples we tried, it means we found some asymmetry so we can stop"""
        if score < _iter:
            win = True
    print('done!')
    winsound.MessageBeep()  # notify me when its done
    weights = np.ones_like(dist)/float(len(dist))  # calculate weights such that bins of the distribution are equal to that of dist
    plt.hist(dist, 50, label='$A_{Tideal}$', weights=weights)  # plot the ideal value of A_T to find asymmetry
    print('events needed to see asymmetry', A_CT_sim, 'with significance is', samplesRequired)


"""will sample through generated data to get a rough order of magnitude of the sample number needed to see CP violation for a given amount"""
def MC_Min_CP(min_, max_, iter_, A_T, A_Tbar, confidence=1, n=int(1E7)):
    samples = np.linspace(min_, max_, iter_)  # get range of samples to try
    l =  0.5 * samples * (1 - A_T[0])  # get C_T < 0
    u = 0.5 * samples * (1+ A_T[0])  # get C_T > 0
    lbar =  0.5 * samples * (1 - A_Tbar[0])  # get -C_Tbar < 0
    ubar = 0.5 * samples * (1+ A_Tbar[0])  # # get -C_Tbar > 0

    print('mean | error')
    """Generates samples of P asymmretries and uses them to calculate the A_CP distribution for each sample"""
    for i in range(len(samples)):
        dist, _, _ = MC_Sim_A_T(l[i], u[i], n)  # get A_T distribution
        distbar, _, _ = MC_Sim_A_T(lbar[i], ubar[i], n)  # get A_Tbar distribution
        A_CP = 0.5 * (dist - distbar)  # calcualte A_CP distribution
        mean = np.mean(A_CP)  # get the mean value
        error = np.std(A_CP)  # get the uncertainty
        print(mean, '|' , error)
        """Check if A_CP asymmetry is significant"""
        if((confidence * error) < abs(mean)):
            break

    return i, mean, samples[i], dist


"""Search algorithm for MC_Min_CP, almost identical to Search_A_T. See Search A_T for details"""
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