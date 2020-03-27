# -*- coding: utf-8 -*-
"""
Created on Wed March 25 12:35:59 2019

@author: Shyam Bhuller
"""
import uproot
import numpy as np
import pandas as pd
import Plotter as pt
import matplotlib.pyplot as plt


"""Opens lHCb data files and puts the 4-vectors into a root file compatible with AMPGEN's fitter"""
def CreateBranches(selection="Both"):
    files = ["tos_Run1", "tis_Run1", "tos_Run2", "tis_Run2"]  # file names
    cuts = [0.9968, 0.9988, 0.9693, 0.9708]  # NNweights to cut from, is the global efficiency
    
    in_names = ["D0", "D0bar", "K_Kst0", "Pi_Kst0"]  # branch names in LHCb data
    in_comps = ['_PE', '_PX', '_PY', '_PZ']  # 4 momenta compnent names for LHCb data
    
    out_names = ["_1_D0", "_2_Dbar0", "_3_K~", "_4_pi#"]  # branch names for the fitter file
    out_comps = ["_E", "_Px", "_Py", "_Pz"]  # 4 momenta compnent names for the fitter file
    
    branches = {}
    """For Loop creates a template of the TTree which, will be populated with LHCb data"""
    for name in out_names:
        for comp in out_comps:
            branches.update({name+comp: np.float})  # key will accept a np.float type but its not specified yet
    
    particles_list = []
    weights_list = []
    """Will open each file and create root branches to add to the fitter file"""
    for i in range(len(files)):
        particles = {}  # particle dictionray
        df = pd.read_pickle(files[i]+".pkl")  # read pickle file
        df = df[df.NN_weights > cuts[i]]  # get data with NN_weights greater then the efficiency
        df = df.drop_duplicates(subset = ['runNumber', 'eventNumber'], keep = 'first')  # remove duplicates

        """Checks if regular, CP or both decays need to be added"""
        if selection != "Both":
            Ktags = df["K_Kst0_ID"].to_numpy()  # get K tags, distinguishes regular decays from conjugate decays
            if selection == "CP":
                df = df[Ktags < 0]
            if selection == "Regular":
                df = df[Ktags > 0]

        """creates Branches of particle 4 momenta components to be added to the TTree"""
        for j in range(len(out_names)):
            for k in range(len(out_comps)):
                particles.update({out_names[j]+out_comps[k]: df[in_names[j]+in_comps[k]].to_numpy()/1000})  # convert to GeV to help the fitter

        particles_list.append(particles)
        weights_list.append(df.sWeights.to_numpy())  # get inWeights to be applied to the data
    
    
    branches = {}
    """Adds the particle data to branches"""
    for i in range(len(out_names)):
        for j in range(len(out_comps)):
            tmp = np.concatenate([particles_list[x][out_names[i]+out_comps[j]] for x in range(len(particles_list))])  # concantenate to merge all LHCb file data into one
            branches.update({out_names[i]+out_comps[j]: tmp})  # add branch
    
    tmp = np.concatenate([weights_list[x] for x in range(len(weights_list))])  # merge inWeights from all data together
    branches.update({"weight": tmp})  # add inWeights to the TTree
    return branches


"""Converts LHCb data files into a root file compatible with the AMPGEN fitter"""
def ConvertFile(name, selection):
    file = uproot.recreate(name)  # open root file to make, recreate will overwrite existing file
    branches = CreateBranches(selection)  # get the data from LHCb data in the correct format

    branches_tmp = {}
    """Create a template branch which represents a TTree in a root file"""
    for branch in branches:
        branches_tmp.update({branch: np.float})  # key will accept a np.float type but its not specified yet

    file.newtree("DalitzEventList", branches_tmp, "DalitzEventList")  # create a tree and add a branch to store the data in
    file["DalitzEventList"].extend(branches)  # add branches to ROOT file
    file.close()


"""Opens Root Histogram and returns the bin center and frequencies"""
def Hist(hist):
    edges = hist.edges
    centers = (edges[1:] + edges[:-1])/2
    values = hist.values
    return [centers, values]


"""Plots fits for resonant mass projections from the AMPGEN fitter output. Contains only histograms."""
def PlotFitProjections(file_name, plotData=True):
    file = uproot.open(file_name)  # open root file

    names = [*dict(file)]  # branch names
    names = [names[x].decode("utf-8")[:-2] for x in range(len(names))]  # convert names from bytes to string

    model_names = names[:int(len(names)/2)]  # get model histogram names

    if plotData is False:
        data_names = names[int(len(names)/2):]  # get data histogram names

    particle_names = ["$m_{D^{0}\\bar{D}^{0}}$", "$m_{D^{0}K^{+}}$", 
                      "$m_{D^{0}\pi^{-}}$", "$m_{\\bar{D}^{0}K^{+}}$",
                      "$m_{\\bar{D}^{0}\pi^{-}}$", "$m_{K^{+}\pi^{-}}$",
                      "$m_{D^{0}\\bar{D}^{0}K^{+}}$", "$m_{D^{0}\\bar{D}^{0}\pi^{-}}$",
                      "$m_{D^{0}K^{+}\pi^{-}}$", "$m_{\\bar{D}^{0}K^{+}\pi^{-}}$"]  # x_labels

    """Plots the data and respective fit calcualted by AMPGEN"""
    for i in range(len(data_names)):
        plt.subplot(3, 4, i+1)  # define subplot in figure
        model = Hist(file[model_names[i]])  # get data from histogram
        model[0] = np.sqrt(model[0])  # turn invarnat mass squared to invariant mass

        if plotData is True:
            data = Hist(file[data_names[i]])
            data[0] = np.sqrt(data[0])

        pt.ErrorPlot(model, alpha=0.5, markersize=2.5)  # plot fitted data
        pt.ErrorPlot(data, y_error=np.sqrt(data[0]), marker=None, capsize=2, alpha=0.5, x_axis=particle_names[i]+"(GeV)", axis=True)  # plot data with errorbars


"""Main Body"""
#ConvertFile("LHCb_run.root", "Both")  # use to format data for AMPGEN
#PlotFitProjections("Fitter\\No_CP_mod.root")  # use to plot results