# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:39:04 2019

@author: Shyam Bhuller

@Description: Contains various functions used to calculate various kinematic quantities and asymmetry values,
prodominantly through vectorised calculations, to speed up computation times.
"""

import numpy as np
import warnings

# General convension / where possible make all inputs 4Vectors and change as neccessary within this script

"""Gets the 3-Vector component of a 4-Vector"""
def Vector_3(v4):
    return v4[:, 1:4]


"""Simulatiously multiplies 4D matrices to respective 4D vectors.
This function will not work for arrays larger then 10^5 because of memory limitations,
to split the data if Needed, see DataManager.SPlitEvents."""
def VectorMatMul(Mats, vector_4):
    out = np.matmul(Mats, np.transpose(vector_4))  # performs matrix multiplication, need to transpose 4-Vectors to match the shapes
    vec = []
    """Matrix mulitpliation is across multiple dimensions, so we need to project the right axis of vectors which relate
    to typical matrix mulitplications of 2D matrices"""
    for i in range(np.size(Mats, 0)):
        v = out[i, :, i]  # get the projection trough the second dimension of the matrix, these are the correct 4-vectors
        vec.append(v)

    vec = np.array(vec)  # convert the list back into a 4-vector
    return vec


"""Calculates the scalar product of a set of vectors, 3 or 4 dimensional.
In both case the dot product definition for both dimensions is used."""
def Scalar(vector, dimension=3):
    """Dimension number can only be 3 or 4, set to 3 by default"""
    if dimension not in range(3, 5):
        warnings.warn("dimesion can only be 3 or 4, set to 3 by default.")
        dimension = 3
    
    """Compute dot product depending on which dimension, for A 4-vector"""
    if np.size(vector, 1) == 4:
        v_3 = vector[:, 1:4]  # gets 3 vector
        s_3 = np.sum(v_3 * v_3, 1)  # comuptes scalar product in 3D for the 4-vector
        if dimension == 4:
            return (vector[:, 0]*vector[:, 0])  - s_3  # computes 4D scalar product if needed
        else:
            return s_3  # returns the 3D scalar product of the 4-vector
    else:
        return np.sum(vector * vector, 1)  # returns the scalar product of a 3-vector


"""Computes the Scalar triple product for 3D vectors"""
def Scalar_TP(v1, v2, v3):
    return np.sum(np.cross(v1, v2, axis=1) * v3, 1)  # does a cross product, then a dot product


"""direction of particles motion, will get it from a 4-vector"""
def Direction(vector_4):
    p_3 = vector_4[:, 1:4]  # gets 3 vector
    p_mag = Mag_3(p_3)  # gets magnitude
    index = np.array(np.where(p_mag == 0))  # get any elements which equal zero
    p_mag[index, :] = 1  # set zero value elements to 1, to prevent dividing by zero
    n = p_3 / p_mag  # get direction
    return n


"""3-momentum magnitude"""
def Mag_3(vector):
    """if a 4-vector is given, it will calculate the magnitiude of the 3 vector compnent only"""
    if np.size(vector, 1) == 4:
        vector_3 = vector[:, 1:4]
    else:
        vector_3 = vector
    mag = np.sqrt(np.sum(np.square(vector_3), 1))  # gets vector magnitude
    return np.repeat(mag[:, np.newaxis], 3, 1)  # returns the magnitude copied 3 times, one for each dimention. This is useful for vectorised calculations.


"""4-momentum magnitude"""
def Mag_4(vector):
    if np.size(vector, 1) != 4:
        Warning("input vectors are the wrong dimension.")  # will only work for 4-vectors
    magsqr = Scalar(vector, 4)  # compute scalar product
    return np.sqrt(magsqr)  # return square root i.e. magnitude


"""Particle velocity as a fraction of c"""
def Beta(vector_4):
    E = vector_4[:, 0]  # energy of the particle
    p_mag = Mag_3(vector_4)[:, 0]  # momentum magnitude
    return p_mag/E  # beta i.e. v/c


"""Lorentz factor"""
def Gamma(beta):
    return 1/np.sqrt(1-np.square(beta))


"""Generates Lorentz boost matrix from particles motion"""
def Boost(g, beta, n):
    # Due to the symmetry of the boost matrix, we can define the time-like
    # compnents as a 4-vector and project various components.
    # the same can be done for the spacial elements, but this time can be repesented by a generic matrix
    gb = g * beta * np.array(n)  # time like components in a vector i.e. matrix with either row or columns at zero

    P_m = (g - 1) * (np.swapaxes(np.array([n]), 0, 1) * np.array([n]))  # spacial compnent i.e. the roation part of the lorentz boost matrix

    # construct the matrix, for full definition see https://en.wikipedia.org/wiki/Lorentz_transformation
    # under section 'Proper Tranformations'
    B = [[g, -gb[0], -gb[1], -gb[2]],
         [-gb[0], 1 + P_m[0, 0], P_m[0, 1], P_m[0, 2]],
         [-gb[1], P_m[1, 0], 1 + P_m[1, 1], P_m[1, 2]],
         [-gb[2], P_m[2, 0], P_m[2, 1], 1 + P_m[2, 2]]]
    return np.array(B)  # convert to array for Vectorised calculations


"""Generates angle between decay planes of p1p2 and p3p4 for 4 body decays. Does this for all events simulatniously"""
def Decay_Plane_Angle(P, p1, p2, p3, p4):

    """Generate Boost matrices, We need to make sure we are in the COM frame i.e. rest frame of parent particle"""
    n = Direction(P)  # direction of the parent particles
    beta = Beta(P)  # speeds of the parent particles
    G = Gamma(beta)  # Loretnz factors of the parent particles

    Mats = []
    """For each event, it generates the Boost matrix to Boost particles into COM frame"""
    for i in range(len(beta)):
        mat = Boost(G[i], beta[i], n[i])  # create boost matrix
        Mats.append(mat)  # add it to the list
    Mats = np.array(Mats)  # convert to 3D array, 1st axis is events, other two store the row and colummns of the matrix

    """Apply boost matrices"""
    p1p = VectorMatMul(Mats, p1)  # simulatniously matrix multiply the events by the repsective boost matrix
    p2p = VectorMatMul(Mats, p2)  # ...
    p3p = VectorMatMul(Mats, p3)
    p4p = VectorMatMul(Mats, p4)

    """calculate TP"""
    z = (p1p[:, 1:4] + p2p[:, 1:4]) / Mag_3(p1p + p2p) # z plane

    n_12 = np.cross(p1p[:, 1:4], p2p[:, 1:4], axis=1)
    n_12 = n_12/Mag_3(n_12)  # get the normal of the p1p2 decay plane

    n_34 = np.cross(p3p[:, 1:4], p4p[:, 1:4], axis=1)
    n_34 = n_34/Mag_3(n_34)  # # get the normal of the p3p4 decay plane

    sin_phi = np.sum(np.cross(n_12, n_34, axis=1) * z, 1)  # calcualte the angle between the normals of the plane

    return np.arcsin(sin_phi)  # return the angle, ranging from -pi to pi


"""Calculate Helicity angles of particles in a decay plane"""
def HelicityAngle(referenceParticle, parent, particle):
    """Generate Boost matrix in frame of reference particle i.e. the resonance of p1p2 or p3pp4 or p1p3 etc."""
    n = Direction(referenceParticle)  # direction of reference particle (rp)
    beta = Beta(referenceParticle)  # speed of rp
    gamma = Gamma(beta)  # lorentz factor of rp

    """For each event, it generates the Boost matrix to Boost particles into rp frame"""
    Mats = []
    for i in range(len(beta)):
        mat = Boost(gamma[i], beta[i], n[i])
        Mats.append(mat)
    Mats = np.array(Mats)

    p_1 = VectorMatMul(Mats, parent)[:,1:4]  # boost parent particle
    p_2 = VectorMatMul(Mats, particle)[:,1:4]  # boost secondary particle to make an angle with

    Mag_1 = Mag_3(p_1)[:, 0]  # get maginitudes
    Mag_2 = Mag_3(p_2)[:, 0]
    cos = np.sum(p_1 * p_2, axis=1)/(Mag_1 * Mag_2)  # generate cosine of the helicity angles
    return -cos  # need to return the -ve as per convention


"""calculates P violating amplitude from a scalar triple prouct and its propagated uncertainty
(assuming poisson distributed uncertainties in number of decays)"""
def TP_Amplitude(TP):
    lower = []  # TP < 0
    upper = []  # TP > 0

    """splits data by sign of TP"""
    for i in range(len(TP)):
        if TP[i] < 0:
            lower.append(TP[i])
        else:
            upper.append(TP[i])

    lower = len(lower)  # we need the number of events only
    upper = len(upper)
    
    return A_T(lower, upper)  # see A_T function for more


"""Splits the ROOT data frame into two lists containing data for differing parities."""
def Segment_C_T(dataFrame):
    C_T = Scalar_TP(Vector_3(dataFrame['p_1']), Vector_3(dataFrame['p_2']), Vector_3(dataFrame['p_3'])) # triple prodcut of 3-momenta
    C_Tl = []  # C_T < 0
    C_Tu = []  # C_T > 0

    indexl = np.array(np.where(C_T < 0)).T  # get appropriate events indicies
    indexu = np.array(np.where(C_T > 0)).T  # ...

    """Will split a particle data frame by the C_T conditions"""
    for i in dataFrame:
        particle = dataFrame[i]  # get particle momenta
        pl = []
        pu = []

        """get all events which satisfy C_T < 0 for the ith particle"""
        for j in range(len(indexl)):
            pl.append(particle[indexl[j], :])

        """get all events which satisfy C_T > 0 for the ith particle"""
        for j in range(len(indexu)):
            pu.append(particle[indexu[j], :])

        pl = np.reshape(np.array(pl), [len(indexl), 4])  # reshape to correct format
        pu = np.reshape(np.array(pu), [len(indexu), 4])  # ...
        C_Tl.append(pl)
        C_Tu.append(pu)
    return C_Tl, C_Tu  # returns the particle dictionary as lists instead


"""Calculates CP sensitive quantity and its error"""
def A_CP(A_T, A_Tbar):
    A_CP = 0.5 * (A_T[0] - A_Tbar[0])  # calculate A_T
    A_CP_error = 0.5 * ((A_T[1])**2 + (A_Tbar[1])**2)**0.5  # calculate propagated error, derived from partial derivitive of A_CP formula
    return A_CP, A_CP_error


"""Calcualtes P sensitive quantity and its error, provided N(C_T > 0) and N(C_T < 0) has been calculated already"""
def A_T(lower, upper):
    A_T = (upper - lower)/(upper + lower)  # calculate parity sensitive quantitiy
    e_l = np.sqrt(lower)  # get poisson error of the samples
    e_u = np.sqrt(upper)  # ...
    #e_A_T = 1/(upper + lower) * np.sqrt(e_u**2 * (1-A_T)**2 + e_l**2 * (1+A_T)**2)  # calculate propagated error, formula dervied from partial derivities of A_T formula
    e_A_T = 2/((upper+lower)**2) * np.sqrt((lower * e_u)**2 + (upper * e_l)**2)
    return A_T, e_A_T
