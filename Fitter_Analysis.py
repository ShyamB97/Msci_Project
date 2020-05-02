# -*- coding: utf-8 -*-
"""
Created on Wed March 25 12:35:59 2019

@author: Shyam Bhuller
"""
import uproot
import numpy as np
import DataManager as dm
import Plotter as pt
import matplotlib.pyplot as plt


"""Opens lHCb data files and puts the 4-vectors into a root file compatible with AMPGEN's fitter"""
def CreateBranches(files, cuts):
    out_names = ["_1_D0", "_2_Dbar0", "_3_K~", "_4_pi#"]  # branch names for the fitter file
    out_comps = ["_E", "_Px", "_Py", "_Pz"]  # 4 momenta compnent names for the fitter file

    particles_list = []
    weights_list = []
    for i in range(len(files)):
        p, pbar, w, wbar = dm.ReadRealData(files[i]+".pkl", cuts[i])
        mask = np.array([[1, -1, -1, -1]] * np.size(pbar["p_0"], 0))

        """Invert the parity"""
        for x in pbar:
            tmp = pbar[x]
            pbar[x] = tmp * mask

        """C transformation"""
        p_1 = pbar["p_1"]
        p_2 = pbar["p_2"]
        pbar["p_2"] = p_1  # switch D0 with D0bar, need this for CP transformation
        pbar["p_1"] = p_2

        lst = []
        tags = ["p_1", "p_2", "p_3", "p_4"]
        for i in range(len(tags)):
            tmp = np.concatenate((p[tags[i]]/1000, pbar[tags[i]]/1000))
            lst.append(tmp)

        dicts = {}
        for i in range(len(out_names)):
            for j in range(len(out_comps)):
                    comp = lst[i][:, j]
                    dicts.update({out_names[i]+out_comps[j]: comp})
        particles_list.append(dicts)
        weights_list.append(np.concatenate((w, wbar)))


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
def ConvertFile(name, files, cuts):
    file = uproot.recreate(name)  # open root file to make, recreate will overwrite existing file
    branches = CreateBranches(files, cuts)  # get the data from LHCb data in the correct format

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
def PlotFitProjections(file_name, plotData=True, fit_color="red"):
    file = uproot.open(file_name)  # open root file

    names = [*dict(file)]  # branch names
    names = [names[x].decode("utf-8")[:-2] for x in range(len(names))]  # convert names from bytes to string

    model_names = names[:int(len(names)/2)]  # get model histogram names

    if plotData is True:
        data_names = names[int(len(names)/2):]  # get data histogram names

    particle_names = ["$m_{D^{0}\\bar{D}^{0}}$", "$m_{D^{0}K^{+}}$", 
                      "$m_{D^{0}\pi^{-}}$", "$m_{\\bar{D}^{0}K^{+}}$",
                      "$m_{\\bar{D}^{0}\pi^{-}}$", "$m_{K^{+}\pi^{-}}$",
                      "$m_{D^{0}\\bar{D}^{0}K^{+}}$", "$m_{D^{0}\\bar{D}^{0}\pi^{-}}$",
                      "$m_{D^{0}K^{+}\pi^{-}}$", "$m_{\\bar{D}^{0}K^{+}\pi^{-}}$"]  # x_labels

    """Plots the data and respective fit calcualted by AMPGEN"""
    for i in range(len(model_names)):
        plt.subplot(3, 4, i+1)  # define subplot in figure
        model = Hist(file[model_names[i]])  # get data from histogram
        model[0] = np.sqrt(model[0])  # turn invarnat mass squared to invariant mass

        pt.ErrorPlot(model, alpha=0.5, markersize=2.5, color=fit_color)  # plot fitted data

        if plotData is True:
            data = Hist(file[data_names[i]])
            data[0] = np.sqrt(data[0])
            pt.ErrorPlot(data, y_error=np.sqrt(data[0]), marker=None, capsize=2, alpha=0.5, x_axis=particle_names[i]+"(GeV)", axis=True, y_axis="Number of Events")  # plot data with errorbars


"""Main Body"""
files = ["tos_Run1", "tis_Run1", "tos_Run2", "tis_Run2"]  # file names
cuts = [0.9968, 0.9988, 0.9693, 0.9708]  # NNweights to cut from, is the global efficiency

#ConvertFile("Fitter\LHCb_run.root", files, cuts)  # use to format data for AMPGEN
PlotFitProjections("Fitter\\all.root", True, "red")  # use to plot results
PlotFitProjections("Fitter\\all_mod.root", False, "black")  # use to plot results