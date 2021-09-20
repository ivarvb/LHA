#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Ivar
"""
import multiprocessing

from multiprocessing import Pool, Manager, Process, Lock
#from sourcecode.src.vx.pclas.Description import Description

import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from skimage.feature.texture import local_binary_pattern
import time
import sys
import random

from  Util import *
from  PyRadiomics import *
#from  Description import *

class ROI:
    def __init__(self, arg):
        self.arg = arg
        #self.columns = []

    @staticmethod
    def getLabel(arg):
        d = []
        for k, v in arg.items():
            if type(v) == list:
                s = [str(a) for a in v]
                s = ",".join(s)
                d.append(str(k)+":["+str(s)+"]")
            else:
                d.append(str(k)+":"+str(v))

        d = "{"+" ".join(d)+"}"
        return d


    @staticmethod
    def makeRegions(icc, smask, nameimg, positions, maskTiles, isPluera, tilepercentage):
        """
        ..........        
        """
        rowvector = []
        ind = []
        for position, mTile in zip(positions, maskTiles):

            # if not True in mTile:
            if mTile.shape[0] * mTile.shape[1] <= 0:
                print("fuck ", mTile.shape[0] * mTile.shape[1])

            if np.sum(mTile == True) <= ((mTile.shape[0] * mTile.shape[1]) * (tilepercentage)):
                continue

            pt = position.T
            imageTile = smask[pt[0]:pt[1], pt[2]:pt[3]]
            imageTile[np.where(mTile == True)] = icc
            indices = np.where(smask == icc)
            
            ind.append(indices)
            rowvector.append([nameimg] + [tilepercentage]+ position.T.tolist() + [icc] + [isPluera])
            icc+=1
        return rowvector, ind


    def process(seff, arg):
        imageName = arg["imageName"]
        inputdir = arg["inputdir"]
        outputdir = arg["outputdir"]
        imagedir = arg["imagedir"]
        boundaryDataSet = arg["boundaryDataSet"]
        targetSet = arg["targetSet"]
        masksdir = arg["masksdir"]
        maskstilesdir = arg["maskstilesdir"]
        tile_size = arg["parameters"]["tile_size"]

        tilepercentage = arg["tilepercentage"]


        base = os.path.basename(imageName)
        base = os.path.splitext(base)
        imgoname = base[0]

        print(imageName)

        fimage = inputdir + '/'+imagedir+'/' + targetSet + '/' + imageName
        
        fimagmask_pleura_path = inputdir + "/" + masksdir + "/" + boundaryDataSet + "/pleura/"
        fimagmask_nonpleura_path = inputdir + "/" + masksdir + "/" + boundaryDataSet + "/non_pleura/"

        #inputImage = cv2.imread(fimage, cv2.IMREAD_GRAYSCALE)
        #get mask and convert them to boolean
        #print(fimagmask_pleura_path + imageName)
        pleuraMask = cv2.imread(fimagmask_pleura_path + imageName, cv2.IMREAD_GRAYSCALE) > 0
        nonPleuraMask = cv2.imread(fimagmask_nonpleura_path + imageName, cv2.IMREAD_GRAYSCALE) > 0


        # split masks into tiles   
        pleuraTiles, positions = Util.splitImage(pleuraMask, tile_size) # get positions just one here because it is the same
        nonPleuraTiles, _ = Util.splitImage(nonPleuraMask, tile_size)

        smask = np.zeros(pleuraMask.shape, int)
        icc = 1
        pleuraDataset, ind_pleura = ROI.makeRegions(icc, smask, imageName, positions, pleuraTiles, "pleura", tilepercentage)
        icc = len(ind_pleura)+1
        nonPleuraDataset, ind_nonpleura = ROI.makeRegions(icc, smask, imageName, positions, nonPleuraTiles, "nopleura", tilepercentage)
        
        del smask


        #################
        #################
        # save

        index_roid = ind_pleura + ind_nonpleura
        
        #image = sitk.ReadImage(fimage, sitk.sitkFloat32)
        image = sitk.ReadImage(fimage)
        image_mask = PyRadiomics.make_mask(image, index_roid)
        PyRadiomics.write_mask(image_mask, outputdir+"/"+imgoname+".nrrd")
        
        #sitk.WriteImage(sitk.LabelToRGB(image_mask), outputdir+"/"+imgoname+".png")
        #################
        #################

        im_size = np.array(image.GetSize())[::-1]
        ma_arr = np.zeros(im_size, dtype=int)
        ma_arr.fill(255)
        rgb_img = np.stack((ma_arr,)*3, axis=-1)

        for r in index_roid:
            rgb_img[r] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        
        #print(rgb_img)

        #rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(outputdir+"/"+imgoname+".png", rgb_img)

        return pleuraDataset, nonPleuraDataset


    def execute(self):
        inputdir = self.arg["inputdir"]
        outputdir = self.arg["outputdir"]
        imagedir = self.arg["imagedir"]
        boundaryDataSet = self.arg["boundaryDataSet"]
        targetSet = self.arg["targetSet"]
        #tile_size = self.arg["tile_size"]
        #radius = self.arg["radius"]

        name = self.arg["name"]
        label = self.arg["label"]
        file = self.arg["file"]
        
        Util.makedir(outputdir)

        print("BEGIN: ")
        arg = []
        for targ in targetSet:
            for imageName in os.listdir(inputdir + '/'+imagedir+'/' + targ):
                #print(self.arg)
                dat = self.arg.copy()
                dat["targetSet"] = targ
                dat["imageName"] = imageName
                arg.append(dat)
        ncpus = multiprocessing.cpu_count()-1

        dataset = pd.DataFrame()
       
        pool = Pool(processes=7)
        rr = pool.map(self.process, arg)
        pool.close()
        
        for rs in rr:
            if len(dataset)==0:
                dataset = rs[0]+rs[1]
            else:
                dataset = dataset + rs[0] + rs[1]
            #print(dataset)

        columns = ["image","tilepercentage","loc1","loc2","loc3","loc4","idseg","target"]

        df = pd.DataFrame(data=dataset)
        df.columns = columns

        df.to_csv(self.arg["outputdir"]+"/"+self.arg["file"], index=False)
        print("END: ")
    
    @staticmethod
    def start(inputdir, outputdir, template):
        traintest = ["train","test"]
        for r in template:
            for erode in r["erode"]:
                r["inputdir"] = inputdir
                r["outputdir"] = outputdir+"/erode_radius_"+str(erode)+"/"+str(r["tilepercentage"])+"/"+str(r["parameters"]["tile_size"])
                r["boundaryDataSet"] = "erode_radius_"+str(erode)
                r["boundaryDataSet_id"] = str(erode)
                r["targetSet"] = traintest
                r["label"] = ROI.getLabel(r["parameters"])
                r["file"] = "info.csv"

                c = r.copy()

                lbpo = ROI(c)
                lbpo.execute()

if __name__ == "__main__":
    print(sys.argv[1])
    with open(sys.argv[1], mode='r') as jsond:
        argsdat = ujson.load(jsond)
        for args in argsdat:
            inputdir = args["inputdir"]
            outputdir = args["outputdir"]
            template = []
            for tile_size in args["tilesize"]:
                template.append({
                    "name":args["name"],
                    "imagedir":args["grayscaledir"],
                    "masksdir":args["masksdir"],
                    "maskstilesdir":args["maskstilesdir"],
                    "parameters":{"tile_size":tile_size},
                    "tilepercentage":args["tilepercentage"],
                    "erode":args["erode"]
                    })
            ROI.start(inputdir, outputdir, template)
