# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:21:57 2020

@author: bhull
"""
import Kinematic as kin  # my own module
import StatTools as st  # my own module
import Plotter as pt  # my own module
import DataManager as dm  # my own module
import numpy as np


"""Gets  data from file !ADD FILENAME SUPPORT!"""
def RetrieveData(generator, filename):
    if generator == 'AmpGen':
        particles = dm.AmpGendf(filename)
    if generator == 'MINT':
        particles = dm.MINTdf()
    return particles


"""Generates 5 of the 34 possible parameters to calculate the LIPS. splits data into parity even/odd bins"""
def DalitzParameters(particles):
    C_Tl, C_Tu = kin.Segment_TP(particles)  # splits data into parity odd-even

    # Helicity angles
    cos_p_1_l = kin.HelicityAngle(C_Tl[1]+C_Tl[2], C_Tl[0], C_Tl[1])
    cos_p_1_u = kin.HelicityAngle(C_Tu[1]+C_Tu[2], C_Tu[0], C_Tu[1])
    cos_p_1 = [cos_p_1_l, cos_p_1_u] # group data set

    cos_p_3_l = kin.HelicityAngle(C_Tl[3]+C_Tl[4], C_Tl[0], C_Tl[3])
    cos_p_3_u = kin.HelicityAngle(C_Tu[3]+C_Tu[4], C_Tu[0], C_Tu[3])
    cos_p_3 = [cos_p_3_l, cos_p_3_u]

    # Invariant masses
    m_12_l = kin.Mag_4(C_Tl[1] + C_Tl[2])
    m_12_u = kin.Mag_4(C_Tu[1] + C_Tu[2])
    m_12 = [m_12_l, m_12_u]

    m_34_l = kin.Mag_4(C_Tl[3] + C_Tl[4])
    m_34_u = kin.Mag_4(C_Tu[3] + C_Tu[4])
    m_34 = [m_34_l, m_34_u]

    # Decay plane angle
    phi_l = kin.Decay_Plane_Angle(*C_Tl)
    phi_u = kin.Decay_Plane_Angle(*C_Tu)
    phi = [phi_l, phi_u]

    return [cos_p_1, cos_p_3, m_12, m_34, phi]


"""Used to generate DalitzParameters for datasets too large to be computed at once."""
def MultiSampleDalitzParameters(particles):
    data = dm.SplitEvents(particles, 10)
    parameters = []
    progress = 0
    for d in data:
        progress += 1
        print(progress/len(data) * 100)
        params = DalitzParameters(d)
        parameters.append(params)

    new_list = []
    for i in range(5):
        subset = []
        for j in range(len(parameters)):
            subset.append(parameters[j][i])
        new_list.append(subset)

    final_data = []
    for i in range(5):
        subset = np.array(new_list[i])
        subset = list(subset.flatten('F'))
        lower = subset[:int(len(subset)/2)]
        lower = np.concatenate(lower).ravel()
        upper = subset[int(len(subset)/2):]
        upper = np.concatenate(upper).ravel()
        subset = [lower, upper]
        final_data.append(subset)
    return final_data

"""Will generate plots to analyse the data provided and use in reports (Hopefully)"""
def GeneratePlots(data, request, x_axis='', y_axis='Number of Events', alpha=0.5, lines=True, single=True, fit_parameters=[1, 1, 1]):
    if request == 'Helicity' or request == 'TP Angle':
        # want to depriciate function if possible.
        pt.Histogram_Multi(data, legend=True, axis=True, labels=['$C_{T} < 0$','$C_{T} > 0$'], x_axis=x_axis, y_axis=y_axis, alpha=alpha)

    if request == 'Mass':
        pt.BWMultiFit(data, legend=True, axis=True, labels=['$C_{T} < 0$','$C_{T} > 0$'], x_axis=x_axis, y_axis=y_axis, lines=lines, single=single, fit_parameters=fit_parameters)


particles = RetrieveData('AmpGen', 'B-4Body.root')
data = MultiSampleDalitzParameters(particles)

