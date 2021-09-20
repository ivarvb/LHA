#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:41:35 2021

@author: oscar
"""

import multiprocessing
from multiprocessing import Pool, Manager, Process, Lock

import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from skimage.feature.texture import local_binary_pattern

#from  PyRadiomics import *

from datetime import datetime

from Util import *
from Description import *

class LBP(Description):
    def __init__(self, arg):
        super().__init__(arg)
        self.columns = []
        

    def computeLBPHistograms(self, icc, imageName, positions, lbpTiles, maskTiles, nBins, isPluera, tilepercentage):
        #print(tilepercentage)
        rowvector = []
        for position, lbpTile, mTile in zip(positions, lbpTiles, maskTiles):

            # if not True in mTile:
            if mTile.shape[0] * mTile.shape[1] <= 0:
                print("fuck ", mTile.shape[0] * mTile.shape[1])

            if np.sum(mTile == True) <= ((mTile.shape[0] * mTile.shape[1]) * (tilepercentage)):
                continue

            # compute the histogram
            xnBins = int(lbpTile.max() + 1)
            histogramPleura, _ = np.histogram(lbpTile[np.where(mTile == True) ], bins=xnBins, range=(0, xnBins))
            rowvector.append([imageName] + position.T.tolist() + [icc] + histogramPleura.tolist() + [isPluera])

            icc+=1

        return rowvector


    def process(self, arg):
        imageName = arg["imageName"]
        inputdir = arg["inputdir"]
        #boundaryDataSet = arg["boundaryDataSet"]
        targetSet = arg["targetSet"]
        imagedir = arg["imagedir"]
        masksdir = arg["masksdir"]
        maskstilesdir = arg["maskstilesdir"]
        tile_size = arg["parameters"]["tile_size"]
        radius = arg["parameters"]["radius"]
        tilepercentage = arg["tilepercentage"]

        
        #df = arg["df"]
        

        print(imageName)

        fimage = inputdir + '/' + imagedir + '/' + targetSet + '/' + imageName
        fimagmask_pleura_path = inputdir + "/" + masksdir + "/pleura/"
        fimagmask_nonpleura_path = inputdir + "/" + masksdir + "/non_pleura/"

        """
        fimage = inputdir + imagedir+'/images_cleaned/' + targetSet + '/' + imageName        
        fimagmask_pleura_path = inputdir + "/boundary_masks/" + boundaryDataSet + "/pleura/"
        fimagmask_nonpleura_path = inputdir + "/boundary_masks/" + boundaryDataSet + "/non_pleura/"
        """

        #print(fimagmask_pleura_path + imageName)
        inputImage = cv2.imread(fimage, cv2.IMREAD_GRAYSCALE)
        pleuraMask = cv2.imread(fimagmask_pleura_path + imageName, cv2.IMREAD_GRAYSCALE) > 0
        nonPleuraMask = cv2.imread(fimagmask_nonpleura_path + imageName, cv2.IMREAD_GRAYSCALE) > 0

        # local Binary Pattern (LBP)
        #radius = 3
        nPoints = 8 * radius

        # Compute LBP fro the whole input image
        lbp = local_binary_pattern(inputImage, nPoints, radius, method='uniform')
        nBins = int(lbp.max() + 1)

        # split masks into tiles   
        pleuraTiles, positions = Util.splitImage(pleuraMask, tile_size) # get positions just one here because it is the same
        nonPleuraTiles, _ = Util.splitImage(nonPleuraMask, tile_size)
        lbpTiles, _ = Util.splitImage(lbp, tile_size)

        icc = 1
        pleuraDataset = self.computeLBPHistograms(icc, imageName, positions, lbpTiles, pleuraTiles, nBins, "pleura", tilepercentage)
        icc = len(pleuraDataset)+1
        nonPleuraDataset = self.computeLBPHistograms(icc, imageName, positions, lbpTiles, nonPleuraTiles, nBins, "nopleura", tilepercentage)
   
        return pleuraDataset + nonPleuraDataset
        #return [pleuraDataset, nonPleuraDataset]



    def execute(self):       
        inputdir = self.arg["inputdir"]
        outputdir = self.arg["outputdir"]

        #boundaryDataSet = self.arg["boundaryDataSet"]
        targetSet = self.arg["targetSet"]
        imagedir = self.arg["imagedir"]
        file = self.arg["file"]
        imagedir = self.arg["imagedir"]
        masksdir = self.arg["masksdir"]
        maskstilesdir = self.arg["maskstilesdir"]
        tile = self.arg["parameters"]["tile_size"]

        tilepercentage = self.arg["tilepercentage"]


        #infocsv =  inputdir + '/'+masksdir+'/'+boundaryDataSet+'/'+tiledir+"/"+'info.csv'
        #df_rois = pd.read_csv(infocsv)

        arg = []
        for imageName in os.listdir(inputdir + '/'+imagedir+'/' + targetSet):
            dat = self.arg.copy()
            #df_filter = df_rois[(df_rois.image == imageName)]
            dat["imageName"] = imageName
            dat["tilepercentage"] = tilepercentage
            
            #dat["df"] = df_filter
            arg.append(dat)

        dataset = []
       
        #ncpus = 15
        ncpus = multiprocessing.cpu_count()

        pool = Pool(processes=ncpus)
        rr = pool.map(self.process, arg)
        pool.close()

        for rs in rr:
            if len(dataset)==0:
                dataset = rs
            else:
                dataset = dataset + rs

        """ 
        for d in arg:
            rs =  process(d)
            if dataset.empty:
                dataset = pd.concat([rs[0], rs[1]])
            else:
                dataset = pd.concat([dataset, rs[0], rs[1]])
            print(dataset)
        """ 
        #for d in dataset:
        #    print(len(d))
        #print("dataset", dataset)

        self.columns = ["LBP"+str(i+1) for i in range(len(dataset[0])-7)]
        xcolumns = ["image","loc1","loc2","loc3","loc4","idseg"]+self.columns+["target"]
        df = pd.DataFrame(data=dataset)
        df.columns = xcolumns
        df.to_csv(outputdir+"/"+file, index=False)

