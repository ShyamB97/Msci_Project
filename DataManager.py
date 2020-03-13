# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:24:13 2020

@author: Shyam Bhuller

@Desciption: Code which handles the importing of data from various file types.
Will also manipulate particle dictionaries i.e. split or merge multiple data sets.
"""

import uproot
from natsort import natsorted, ns
import numpy as np
import os
import pandas as pd


"""Will reuturn the ROOT filenames in a given directory"""
def GetFileNames(directory='\samples'):
    eventFiles = []
    path = os.getcwd() + directory  # get current direcroty plus folder we want to search
    
    """Goes through each file in the directory and subdirectories and gets the names of any ROOT file"""
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".root"):
                path_file = os.path.join(directory[1:],file)  # create the path_file string
                #print(path_file)
                eventFiles.append(path_file)
    return eventFiles


"""Opens ROOT file generated by MINT and stores particle data into a dictionary (depriciated)"""
def MINTdf(filename='GeneratedMC_1.root', CP=False):
    tree = uproot.open('GeneratedMC_1.root')['DalitzEventList']  # open 4-vecotrs in the root file
    df = tree.pandas.df()  # turn into a dataframe
    df_n = np.array(df)  # turn into a numpy array

    """If we want the C conjugate, invert the particle kinematics. So create a mask to apply to the numpy array"""
    if CP is True:
        mask = [[1, -1, -1, -1]] * np.size(df, 0)  # create C conjugate mask (invert 3 momenta)
    if CP is False:
        mask = [[1, 1, 1, 1]] * np.size(df, 0)  # create default mask
    mask = np.array(mask)

    """data for each particles (MeV) where a row is a 4 vector"""
    p_0 = df_n[:, 1:5] * mask
    p_1 = df_n[:, 6:10] * mask
    p_2 = df_n[:, 11:15] * mask
    p_3 = df_n[:, 16:20] * mask
    p_4 = df_n[:, 21:25] * mask
    particles = {'p_0': p_0, 'p_1': p_1, 'p_2': p_2, 'p_3': p_3, 'p_4': p_4}  # this is a particle dictionary containing  the 4-vectors of the intial particle p_0 and the final particles
    return particles


"""Opens ROOT file generated by AmpGen and stores particle data into a dictionary"""
def AmpGendf(filename='output.root', CP=False):
    tree = uproot.open(filename)['DalitzEventList']  # open 4-vecotrs in the root file
    df = tree.pandas.df()
    df_n = np.array(df) * 1000  # GeV to MeV
    df_n = df_n[:, 0:16]  # get only the 4-vecotrs (not interested in weights)

    """Generate Mask depending on if we want the C conjugate"""
    if CP is True:
        mask = [[1, -1, -1, -1]] * np.size(df, 0)
    if CP is False:
        mask = [[1, 1, 1, 1]] * np.size(df, 0)
    mask = np.array(mask)

    """data for each particles (MeV) where a row is a 4 vector"""
    p_1 = df_n[:, 0:4] * mask
    p_2 = df_n[:, 4:8] * mask
    p_3 = df_n[:, 8:12] * mask
    p_4 = df_n[:, 12:16] * mask
    p_0 = p_1 + p_2 + p_3 + p_4  # get 4-vectors of parent particle
    particles = {'p_0': p_0, 'p_1': p_1, 'p_2': p_2, 'p_3': p_3, 'p_4': p_4}
    return particles


"""If an eventfile is too large to interpret using vetorised computations, this will split the data set into smaller ones"""
def SplitEvents(particles, number=10):
    events = np.size(particles['p_0'], 0)
    segments= int(events/number)  # gets the array length of each segment
    
    p = []
    """Will cycle through each particle, split the data evenly and add it to the list p"""
    for i in particles:
        particle = particles[i]  # get particle 4-vectors
        for j in range(number):
            p.append(particle[j*segments: segments*(j+1), :])  # splits the particle 4-vectors

    data = []
    """Will go throguh each split particle segment in p and construct the required number of particle dictionaries"""
    for i in range(number):
        p_0 = p[0 + i]  # parent particle of the ith dictionary
        p_1 = p[1*number + i]  # firt final state particle of the ith dictionary
        p_2 = p[2*number + i]  # ...
        p_3 = p[3*number + i]
        p_4 = p[4*number + i]
        subset = {'p_0': p_0, 'p_1': p_1, 'p_2': p_2, 'p_3': p_3, 'p_4': p_4}  # return in the dictionary format
        data.append(subset)
    return data


"""Get mutiple dataframes in a list from a directory"""
def GenerateDataFrames(directory, CP):
    fileNames = GetFileNames(directory)  # get the filenames to open
    fileNames = natsorted(fileNames, alg=ns.IGNORECASE)  # open file names, sorted by the naming convention of the file

    """Produce dataframes for each file"""
    data = []
    for i in range(len(fileNames)):
        p = AmpGendf(fileNames[i], CP)  # get the particle dictionary
        data.append(p)
    return data


"""Merges dataframes into a single list of values (preserves order)"""
def MergeData(lst, dfs=10):
    data_full = []
    for i in range(5):
        particle = []
        name = 'p_' + str(i)  # get particle name
        """For each dataframe, merge the particles 4-vectors into one list"""
        for j in range(dfs):
            particle.append(lst[j][name])
        data_full.append(np.vstack(particle))  # merge list of numpy arrays row-wise
    return data_full


"""Used to construct numpy 4-Vectors from the real data set. Splits into regular and conjugate decays"""
def ConstructParticle(particleName, df, tags):
    component_name = ['_PE', '_PX', '_PY', '_PZ']  # 4-vectors needed look like paritclName_PE etc.
    component = []
    """Gets each 4-vector component of the perticle and adds it to the list"""
    for i in range(len(component_name)):
        component.append(df[particleName + component_name[i]].to_numpy())  # get pickle dataframe and convert into numpy array
    component = np.vstack(component).T  # stack valuees column wise and transpose, to get it into the same format as the other data
    p = component[tags > 0]  # gets regular decays from the K tags
    pbar = component[tags < 0]  # gets conjugate decays
    return p, pbar


"""Reads the Real Data (stored in a pickle file) and returns particles dictionary and the signal weightings"""
def ReadRealData(name='Data_sig_tos_weights.pkl'):
    df = pd.read_pickle(name)  # read pickle file
    opt_cut = 0.9979  # global efficiency of data
    df = df[df.NN_weights > opt_cut]  # get data with NN_weights greater then the efficiency
    df = df.drop_duplicates(subset = ['runNumber', 'eventNumber'], keep = 'first')  # remove duplicates

    totalWeights = df.sWeights.to_numpy()  # get weights of the events
    Ktags = df["K_Kst0_ID"].to_numpy()  # get K tags, distinguishes regular decays from conjugate decays

    p_0 = ConstructParticle('B0', df, Ktags)  # get particle 4-Vectors, splits into regualr and conjugate decays
    p_1 = ConstructParticle('D0', df, Ktags)  # ...
    p_2 = ConstructParticle('D0bar', df, Ktags)
    p_3 = ConstructParticle('K_Kst0', df, Ktags)
    p_4 = ConstructParticle('Pi_Kst0', df, Ktags)

    particles = {'p_0': p_0[0], 'p_1': p_1[0], 'p_2': p_2[0], 'p_3': p_3[0], 'p_4': p_4[0]}  # create paritcle dictionary
    particlesbar = {'p_0': p_0[1], 'p_1': p_1[1], 'p_2': p_2[1], 'p_3': p_3[1], 'p_4': p_4[1]}  # create conjugate particle dictionary

    weights = totalWeights[Ktags > 0]  # get weights of regular decays
    weightsbar = totalWeights[Ktags < 0]  # get weights of conjugate decays
    return particles, particlesbar, weights, weightsbar