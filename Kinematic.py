# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:39:04 2019

@author: bhull
"""

import numpy as np
import warnings
# General convension / where possible make all inputs 4Vectors and change as neccessary within this script
"""Gets the 3-Vector component of a 4-Vector"""
def Vector_3(v4):
    return v4[:, 1:4]


"""Simulatiously multiplies 4D matrices to respective 4D vectors"""
def VectorMatMul(Mats, vector_4):
    out = np.matmul(Mats, np.transpose(vector_4))
    vec = []
    for i in range(np.size(Mats, 0)):
        v = out[i, :, i]
        vec.append(v)

    vec = np.array(vec)
    return vec


"""calculates the scalar product of a set of vectors, 3 or 4 dimensional"""
def Scalar(vector, dimension=3):
    if dimension not in range(3, 5):
        warnings.warn("dimesion can only be 3 or 4, set to 3 by default.")
        dimension = 3
    if np.size(vector, 1) == 4:
        v_3 = vector[:, 1:4]
        s_3 = np.sum(v_3 * v_3, 1)
        if dimension == 4:
            return (vector[:, 0]*vector[:, 0])  - s_3
        else:
            return s_3
    else:
        return np.sum(vector * vector, 1)


"""Computes the Scalar triple product for 3D vectors"""
def Scalar_TP(v1, v2, v3):
    return np.sum(np.cross(v1, v2, axis=1) * v3, 1)


"""direction of particles motion"""
def Direction(vector_4):
    p_3 = vector_4[:, 1:4]
    p_mag = Mag_3(p_3)
    index = np.array(np.where(p_mag == 0))
    p_mag[index, :] = 1
    n = p_3 / p_mag
    return n


"""3-momentum magnitude"""
def Mag_3(vector):
    if np.size(vector, 1) == 4:
        vector_3 = vector[:, 1:4]
    else:
        vector_3 = vector
    mag = np.sqrt(np.sum(np.square(vector_3), 1))
    return np.repeat(mag[:, np.newaxis], 3, 1)

"""4-momentum magnitude"""
def Mag_4(vector):
    if np.size(vector, 1) != 4:
        Warning("input vectors are the wrong dimension.")
    magsqr = Scalar(vector, 4)
    return np.sqrt(magsqr)


"""Particle velocity"""
def Beta(vector_4):
    E = vector_4[:, 0]
    p_mag = Mag_3(vector_4)[:, 0]
    return p_mag/E


"""Lorentz factor"""
def Gamma(beta):
    return 1/np.sqrt(1-np.square(beta))


"""Generates Lorentz boost matrix from particles motion"""
def Boost(g, beta, n):
    gb = g * beta * np.array(n)

    P_m = (g - 1) * (np.swapaxes(np.array([n]), 0, 1) * np.array([n]))

    B = [[g, -gb[0], -gb[1], -gb[2]],
         [-gb[0], 1 + P_m[0, 0], P_m[0, 1], P_m[0, 2]],
         [-gb[1], P_m[1, 0], 1 + P_m[1, 1], P_m[1, 2]],
         [-gb[2], P_m[2, 0], P_m[2, 1], 1 + P_m[2, 2]]]
    return np.array(B)


"""Generates angle between decay planes of p1p2 and p3p4 for 4 body decays."""
def Decay_Plane_Angle(P, p1, p2, p3, p4):

    """Generate Boost matrices"""
    n = Direction(P)
    A = Beta(P)
    G = Gamma(A)

    Mats = []
    for i in range(len(A)):
        mat = Boost(G[i], A[i], n[i])
        Mats.append(mat)
    Mats = np.array(Mats)

    """Apply boost matrices"""
    p1p = VectorMatMul(Mats, p1)
    p2p = VectorMatMul(Mats, p2)
    p3p = VectorMatMul(Mats, p3)
    p4p = VectorMatMul(Mats, p4)

    #p1 = D0
    #p2 = Dbar0
    #p3 = Kstar
    #p4 = pi
    """calculate TP"""
    z = (p1p[:, 1:4] + p2p[:, 1:4]) / Mag_3(p1p + p2p) # z plane

    n_12 = np.cross(p1p[:, 1:4], p2p[:, 1:4], axis=1)
    n_12 = n_12/Mag_3(n_12)

    n_34 = np.cross(p3p[:, 1:4], p4p[:, 1:4], axis=1)
    n_34 = n_34/Mag_3(n_34)

    sin_phi = np.sum(np.cross(n_12, n_34, axis=1) * z, 1)

    return np.arcsin(sin_phi)


"""Calculate Helicity angles of particles in a decay plane"""
def HelicityAngle(referenceParticle, parent, particle):
    n = Direction(referenceParticle)  # direction of 3d momenta
    beta = Beta(referenceParticle)  # speed of rf
    gamma = Gamma(beta)  # lorentz factor

    # MAKE BOOST MATRIX
    Mats = []
    for i in range(len(beta)):
        mat = Boost(gamma[i], beta[i], n[i])
        Mats.append(mat)
    Mats = np.array(Mats)

    p_1 = VectorMatMul(Mats, parent)[:,1:4] # boost particles
    p_2 = VectorMatMul(Mats, particle)[:,1:4]

    Mag_1 = Mag_3(p_1)[:, 0]
    Mag_2 = Mag_3(p_2)[:, 0]
    cos = np.sum(p_1 * p_2, axis=1)/(Mag_1 * Mag_2)  # generate helicity angles
    return -cos


"""calculates P violating amplitude from TP and its propagated uncertainty
(assuming poisson distributed uncertainties in number of decays)"""
def TP_Amplitude(data):
    lower = []
    upper = []

    for i in range(len(data)):
        if data[i] < 0:
            lower.append(data[i])
        else:
            upper.append(data[i])

    lower = len(lower)
    upper = len(upper)
    A = (upper - lower)/(upper + lower)
    e_l = np.sqrt(lower)
    e_u = np.sqrt(upper)

    e_A = 1/(upper + lower) * np.sqrt(e_u**2 * (1-A)**2 + e_l**2 * (1+A)**2)
    return A, e_A


"""Splits the ROOT data frame into two lists containing data for differing parities"""
def Segment_TP(dataFrame):
    C_T = Scalar_TP(Vector_3(dataFrame['p_1']), Vector_3(dataFrame['p_2']), Vector_3(dataFrame['p_3']))# triple prodcut momenta
    C_Tl0 = []
    C_Tu0 = []

    indexl = np.array(np.where(C_T < 0)).T  # which events have -ve C-T
    indexu = np.array(np.where(C_T > 0)).T  # which events have +ve C-T

    # Split dataframe
    for i in dataFrame:
        particle = dataFrame[i]
        pl0 = []
        pu0 = []

        # get C_T < 0
        for j in range(len(indexl)):
            pl0.append(particle[indexl[j], :])

        # get C_T < 0
        for j in range(len(indexu)):
            pu0.append(particle[indexu[j], :])

        pl0 = np.reshape(np.array(pl0), [len(indexl), 4])  # reshape to correct format
        pu0 = np.reshape(np.array(pu0), [len(indexu), 4])
        C_Tl0.append(pl0)
        C_Tu0.append(pu0)
    return C_Tl0, C_Tu0


"""Calcualtes CP sensitive quantity ans its error"""
def A_CP(A_T, A_Tbar):
    A_CP = 0.5 * (A_T[0] - A_Tbar[0])
    A_CP_error = ((0.5*A_T[1])**2 + (0.5*A_Tbar[1])**2)**0.5
    return A_CP, A_CP_error






