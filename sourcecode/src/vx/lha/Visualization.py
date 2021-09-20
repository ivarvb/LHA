#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import pandas as pd
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

import sys
import os

import SimpleITK as sitk

import imageio

import multiprocessing
from multiprocessing import Pool, Manager, Process, Lock


from Util import *

class Visualization:
    def __init__(self, arg):
        self.arg = arg

    # one image
    def process(self, arg):
        imageName = arg["imageName"]
        datasetdir = arg["datasetdir"]
        ouputdir = arg["outputdir"]

        targetSet = arg["targetSet"]
        imagegrayscaledir = arg["grayscaledir"]
        imagesdir = arg["imagesdir"]
        masksdir = arg["masksdir"]
        
        tile = arg["classifier"][arg["clsr"]]["tile_size"]
        radius = arg["classifier"][arg["clsr"]]["radius"]
        
        tilepercentage = arg["tilepercentage"]        

        base = os.path.splitext(imageName)
        imageNameO = base[0]

        fimageGT = datasetdir + '/'+arg["maskstilesdir"]+"/"+str(tilepercentage)+'/'+str(tile)+'/'+imageNameO+".nrrd"
        fimageGT_png = datasetdir + '/'+arg["maskstilesdir"]+"/"+str(tilepercentage)+'/'+str(tile)+'/'+imageNameO+".png"
        fimageGT_png_out = imageNameO+"_tile.png"
        
        
        fimageGT_out_path = arg["outputdir"]+"/"+arg["clsr"]+"_"+str(tile)+"_"+str(radius)
        fimageGT_out_GT = fimageGT_out_path+"/"+imageNameO+"_GT.png"
        fimageGT_out_GT_tile = fimageGT_out_path+"/"+imageNameO+"_GT_tile.png"
        fimageGT_out_pred = fimageGT_out_path+"/"+imageNameO+"_predicted.png"

        

        #fimageGT_out = arg["inputdir"]+"/"+arg["predictedfile"]

        fimage = datasetdir + '/'+imagesdir+'/'+imageName
        fimage_gray = datasetdir + '/' + imagegrayscaledir + '/' + targetSet + '/' + imageName
        fimagmask_pleura_path = datasetdir + "/" + masksdir + "/pleura/"
        fimagmask_nonpleura_path = datasetdir + "/" + masksdir + "/non_pleura/"

        print(fimage)
        #print(fimage)
        #print(fimagmask_pleura_path)
        #print(fimagmask_nonpleura_path)

        rgb_img_GT = cv2.imread(fimage)
        inputImageGray = cv2.imread(fimage_gray, cv2.IMREAD_GRAYSCALE)
        pleuraMask = cv2.imread(fimagmask_pleura_path + imageName, cv2.IMREAD_GRAYSCALE) > 0
        nonPleuraMask = cv2.imread(fimagmask_nonpleura_path + imageName, cv2.IMREAD_GRAYSCALE) > 0
        
        #print(rgb_img_GT)
        
        #rgb_img_GT = np.stack((inputImage,)*3, axis=-1)
        rgb_img_GT_tile = np.stack((inputImageGray,)*3, axis=-1)
        rgb_img_pred = np.stack((inputImageGray,)*3, axis=-1)
        #print(rgb_img_GT)



        maskGT = sitk.ReadImage(fimageGT)
        maskGT_array = sitk.GetArrayFromImage(maskGT)
        #lsif = sitk.LabelShapeStatisticsImageFilter()
        #lsif.Execute(maskGT)
        #labels = lsif.GetLabels()

        red, green = [255,0,0], [0,255,0]


        #print("df",arg["df"])
        for index, row in arg["df"].iterrows():
            index = np.where(maskGT_array == row["idseg"])        
            
            color = red
            if row['target']=="pleura":
                color = green
            rgb_img_GT_tile[index] = color

            color = red
            if row['target_predicted']=="pleura":
                color = green
            rgb_img_pred[index] = color


        rgb_img_GT[pleuraMask] = green
        rgb_img_GT[nonPleuraMask] = red
        
        if os.path.exists(fimageGT_png):
            cptext = "cp -r "+fimageGT_png+" "+fimageGT_out_path+"/"+fimageGT_png_out
            #print(cptext)
            os.popen(cptext)

        rgb_img_GT = cv2.cvtColor(rgb_img_GT, cv2.COLOR_RGB2BGR)
        cv2.imwrite(fimageGT_out_GT, rgb_img_GT)

        rgb_img_GT_tile = cv2.cvtColor(rgb_img_GT_tile, cv2.COLOR_RGB2BGR)
        cv2.imwrite(fimageGT_out_GT_tile, rgb_img_GT_tile)

        rgb_img_pred = cv2.cvtColor(rgb_img_pred, cv2.COLOR_RGB2BGR)
        cv2.imwrite(fimageGT_out_pred, rgb_img_pred)


        """
        for l in labels:
            index = np.where(mask_array == int(l))        
            rgb_img[index] = colors[int(l)]
        #sitk.WriteImage(sitk.GetImageFromArray(spmap), "./superpixels/"+str(side)+"/"+row["id_image"]+".nrrd", True)
        """
    def execute(self):
        datasetdir = self.arg["datasetdir"]
        imagedir = self.arg["grayscaledir"]
        #tile = self.arg["tile_size"]

        arg = []
        for clsr in self.arg["classifier"]:
            fimageGT_out_path = self.arg["outputdir"]+"/"+clsr+"_"+str(self.arg["classifier"][clsr]["tile_size"])+"_"+str(self.arg["classifier"][clsr]["radius"])
            Util.makedir(fimageGT_out_path)


            #infocsv = datasetdir + '/'+self.arg["maskstilesdir"]+'/'+str(tile)+"/"+'info.csv'
            predictedfile = self.arg["classifier"][clsr]["predictedfile"]
            inffocsv_predicted = self.arg["inputdir"]+"/"+predictedfile
            df_rois = pd.read_csv(inffocsv_predicted)
                    
            images = self.arg["classifier"][clsr]["images"]
            for traintest in ["test"]:
                if len(images)>0:
                    for imageName in os.listdir(datasetdir + '/'+imagedir +'/' + traintest):
                        if imageName in images:
                            dat = self.arg.copy()
                            df_filter = df_rois[(df_rois.image == imageName)]
                            dat["clsr"] = clsr
                            dat["imageName"] = imageName
                            dat["targetSet"] = traintest
                            dat["df"] = df_filter
                            arg.append(dat)                    
                else:
                    for imageName in os.listdir(datasetdir + '/'+imagedir +'/' + traintest):
                        dat = self.arg.copy()
                        df_filter = df_rois[(df_rois.image == imageName)]
                        dat["clsr"] = clsr
                        dat["imageName"] = imageName
                        dat["targetSet"] = traintest
                        dat["df"] = df_filter
                        arg.append(dat)

        #for dar in arg:
        #    self.process(dar)

        #"""
        ncpus = multiprocessing.cpu_count()-1
        pool = Pool(processes=ncpus)
        rr = pool.map(self.process, arg)
        pool.close()
        #"""


if __name__ == "__main__":
    with open(sys.argv[1], mode='r') as jsond:
        #print (jsdata)
        args = ujson.load(jsond)
        #print(args)
        for arg in args:
            obj = Visualization(arg)
            obj.execute()





