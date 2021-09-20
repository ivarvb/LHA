

import sys
import os


import cv2
import pandas as pd
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

import sys

import SimpleITK as sitk


import multiprocessing
from multiprocessing import Pool, Manager, Process, Lock


from Util import *


class Metric:

    def __init__ (self, arg):
        self.arg = arg

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
       
        
        fimage = datasetdir + '/'+imagesdir+'/'+imageName
        fimagmask_pleura_path = datasetdir + "/" + masksdir + "/pleura/"
        fimagmask_nonpleura_path = datasetdir + "/" + masksdir + "/non_pleura/"

        print(fimage)
       
        pleuraMask = cv2.imread(fimagmask_pleura_path + imageName, cv2.IMREAD_GRAYSCALE) > 0
        nonPleuraMask = cv2.imread(fimagmask_nonpleura_path + imageName, cv2.IMREAD_GRAYSCALE) > 0
          

        img_GT_pleura = np.zeros((pleuraMask.shape[0], pleuraMask.shape[1]), np.int8)
        img_GT_nonpleura = np.zeros((pleuraMask.shape[0], pleuraMask.shape[1]), np.int8)

        img_pred_pleura = np.zeros((pleuraMask.shape[0], pleuraMask.shape[1]), np.int8)
        img_pred_nonpleura = np.zeros((pleuraMask.shape[0], pleuraMask.shape[1]), np.int8)


        maskGT = sitk.ReadImage(fimageGT)
        maskGT_array = sitk.GetArrayFromImage(maskGT)

        for index, row in arg["df"].iterrows():
            index = np.where(maskGT_array == row["idseg"])        

            if row['target_predicted']=="pleura":
                img_pred_pleura[index] = 1
            elif row['target_predicted']=="nopleura":
                img_pred_nonpleura[index] = 1

        img_GT_pleura[pleuraMask] = 1
        img_GT_nonpleura[nonPleuraMask] = 1
        #print(img_GT_pleura, img_GT_nonpleura)

        result = [imageName, arg["clsr"] ]
        
        dice_p = Metric.dice(img_GT_pleura, img_pred_pleura, k=1)
        dice_np = Metric.dice(img_GT_nonpleura, img_pred_nonpleura, k=1)
        result += [dice_p, dice_np]
        
        jaccard_p = Metric.jaccard(img_GT_pleura, img_pred_pleura, k=1)
        jaccard_np = Metric.jaccard(img_GT_nonpleura, img_pred_nonpleura, k=1)
        result += [jaccard_p, jaccard_np]

        #print(result)
        return result

    def execute(self):
        datasetdir = self.arg["datasetdir"]
        imagedir = self.arg["grayscaledir"]
        #tile = self.arg["tile_size"]
        tile_size = 0

        arg = []
        for clsr in self.arg["classifier"]:
            fimageGT_out_path = self.arg["outputdir"]+"/"+clsr+"_"+str(self.arg["classifier"][clsr]["tile_size"])+"_"+str(self.arg["classifier"][clsr]["radius"])
            Util.makedir(fimageGT_out_path)


            #infocsv = datasetdir + '/'+self.arg["maskstilesdir"]+'/'+str(tile)+"/"+'info.csv'
            predictedfile = self.arg["classifier"][clsr]["predictedfile"]
            inffocsv_predicted = self.arg["inputdir"]+"/"+predictedfile
            df_rois = pd.read_csv(inffocsv_predicted)
                    
            images = self.arg["classifier"][clsr]["images"]
            tile_size = self.arg["classifier"][clsr]["tile_size"]
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

        datacsv = []
        """ 
        for dar in arg:
            dd = self.process(dar)
            datacsv.append(dd)
        """

        
        ncpus = multiprocessing.cpu_count()-1
        pool = Pool(processes=ncpus)
        rr = pool.map(self.process, arg)
        pool.close()
        for r in rr:
            #print(r)
            datacsv.append(r)
        
        print(datacsv)

        columnnames = ["image", "classifier", "dice_pleura", "dice_nopleura", "jaccard_pleura", "jaccard_nopleura"]
        df = pd.DataFrame(data=datacsv)
        df.columns = columnnames
        df.to_csv(self.arg["outputdir"]+"/metrics_"+str(tile_size)+".csv", index=False)


    @staticmethod        
    def dice(gt, seg, k=1):
        #k=1
        dice = np.sum(seg[gt==k])*2.0 / (np.sum(seg) + np.sum(gt))
        return dice

    def jaccard(gt, seg, k=1):
        #k=1
        inter = np.sum(seg[gt==k])
        jacc = inter / ( (np.sum(seg) + np.sum(gt))-inter )
        return jacc


if __name__ == "__main__":    
    with open(sys.argv[1], mode='r') as jsond:
        #print (jsdata)
        args = ujson.load(jsond)
        #print(args)
        for arg in args:
            obj = Metric(arg)
            obj.execute()


