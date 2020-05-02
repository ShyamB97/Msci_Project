# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:21:57 2020

@author: Shyam Bhuller

@Description: Can be used to create histograms of the CM variables for a given sample of data. Uses Custom buiilt modules to do so.
"""
import Kinematic as kin  # vecotrised 4-vector kineamtics
import Plotter as pt  # generic plotter with consistent formatting and curve fitting
import DataManager as dm  # handles data opened from data files
import numpy as np


"""Gets data from a file created by an event generator"""
def RetrieveData(generator, filename, CP=False):
    if generator == 'AmpGen':
        particles = dm.AmpGendf(filename, CP)
    if generator == 'MINT':  # MINT data no longer used
        particles = dm.MINTdf()
    return particles


"""Generates 5 of the 34 possible parameters to calculate the LIPS. splits data into parity even/odd bins"""
def DalitzParameters(particles):
    C_Tl, C_Tu = kin.Segment_C_T(particles)  # splits data into parity odd-even
    """Helicity Angles"""
    cos_p_1_l = kin.HelicityAngle(C_Tl[1]+C_Tl[2], C_Tl[0], C_Tl[1])  # first helcity angle for C_T < 0
    cos_p_1_u = kin.HelicityAngle(C_Tu[1]+C_Tu[2], C_Tu[0], C_Tu[1])  # .. C_T > 0
    cos_p_1 = [cos_p_1_l, cos_p_1_u] # group data set

    cos_p_3_l = kin.HelicityAngle(C_Tl[3]+C_Tl[4], C_Tl[0], C_Tl[3])  # second helicity angle
    cos_p_3_u = kin.HelicityAngle(C_Tu[3]+C_Tu[4], C_Tu[0], C_Tu[3])
    cos_p_3 = [cos_p_3_l, cos_p_3_u]

    """Invariant masses"""
    m_12_l = kin.Mag_4(C_Tl[1] + C_Tl[2])  # invariant mass of p1 and p2 for C_T < 0
    m_12_u = kin.Mag_4(C_Tu[1] + C_Tu[2])  # ...
    m_12 = [m_12_l, m_12_u]

    m_34_l = kin.Mag_4(C_Tl[3] + C_Tl[4])
    m_34_u = kin.Mag_4(C_Tu[3] + C_Tu[4])
    m_34 = [m_34_l, m_34_u]

    """Decay plane angle"""
    phi_l = kin.Decay_Plane_Angle(*C_Tl)  # decay plane angle for C_T < 0
    phi_u = kin.Decay_Plane_Angle(*C_Tu)  # .. C_T > 0
    phi = [phi_l, phi_u]

    return [cos_p_1, cos_p_3, m_12, m_34, phi]


"""Used to generate DalitzParameters for datasets too large to be computed at once."""
def MultiSampleDalitzParameters(particles, split=10):
    data = dm.SplitEvents(particles, split)  # splits events into smaller sets
    parameters = []
    progress = 0

    """Calcualte CM variables"""
    for d in data:
        progress += 1
        print(progress/len(data) * 100)
        params = DalitzParameters(d)  # calulate statistics for the data set
        parameters.append(params)

    new_list = []
    """Merges each CM variable and C_T calulated for each data set"""
    for i in range(5):
        subset = []
        for j in range(len(parameters)):
            subset.append(parameters[j][i])
        new_list.append(subset)

    final_data = []
    """Puts the calculated values in a single list"""
    for i in range(5):
        subset = np.array(new_list[i])  # converts list into an array
        subset = list(subset.flatten('F'))  # flattens this list into a matrix, columns are differnet CM variables for different C_T
        lower = subset[:int(len(subset)/2)]  # gets C_T < 0 states
        lower = np.concatenate(lower).ravel()  # merge columns into one column
        upper = subset[int(len(subset)/2):]  # # gets C_T > 0 states
        upper = np.concatenate(upper).ravel()
        subset = [lower, upper]  # creates a list of uppr and lower values for the single CM variable
        final_data.append(subset)
    return final_data

"""Will generate plots to analyse the data provided and use in reports"""
def GeneratePlots(data, request, x_axis='', y_axis='Number of Events', alpha=0.5, lines=True, single=True, fit_parameters=[1, 1, 1]):
    if request == 'Helicity' or request == 'TP Angle':
        pt.Histogram_Multi(data, legend=True, axis=True, labels=['$C_{T} < 0$','$C_{T} > 0$'], x_axis=x_axis, y_axis=y_axis, alpha=alpha)

    if request == 'Mass':
        results = pt.BWMultiFit(data, legend=True, axis=True, labels=['$C_{T} < 0$','$C_{T} > 0$'], x_axis=x_axis, y_axis=y_axis, lines=lines, single=single, fit_parameters=fit_parameters)
        return results

#particles = RetrieveData('AmpGen', 'Dto4_Body.root', False)  # get data
particles = dm.GenerateDataFrames("\P45_A0.75_1M", False)
particles = dm.MergeData(particles, 10)
particles = {'p_0': particles[0], 'p_1': particles[1], 'p_2': particles[2], 'p_3': particles[3], 'p_4': particles[4]}
data = MultiSampleDalitzParameters(particles, 1000)  # calculate CM variables

# Use GeneratePlot() in a console or write stuff here.