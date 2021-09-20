#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Ivar
"""

from multiprocessing import Pool, Manager, Process, Lock

import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from skimage.feature.texture import local_binary_pattern
import time
import sys

from  PyRadiomics import *
from  Description import *


class RadiomicsFeatures():
    def __init__(self, image, mask):
        self.image = image
        self.mask = mask
        self.columns = ['original_firstorder_10Percentile', 'original_firstorder_90Percentile', 'original_firstorder_Energy', 'original_firstorder_Entropy', 'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis', 'original_firstorder_Maximum', 'original_firstorder_MeanAbsoluteDeviation', 'original_firstorder_Mean', 'original_firstorder_Median', 'original_firstorder_Minimum', 'original_firstorder_Range', 'original_firstorder_RobustMeanAbsoluteDeviation', 'original_firstorder_RootMeanSquared', 'original_firstorder_Skewness', 'original_firstorder_TotalEnergy', 'original_firstorder_Uniformity', 'original_firstorder_Variance', 'original_glcm_Autocorrelation', 'original_glcm_ClusterProminence', 'original_glcm_ClusterShade', 'original_glcm_ClusterTendency', 'original_glcm_Contrast', 'original_glcm_Correlation', 'original_glcm_DifferenceAverage', 'original_glcm_DifferenceEntropy', 'original_glcm_DifferenceVariance', 'original_glcm_Id', 'original_glcm_Idm', 'original_glcm_Idmn', 'original_glcm_Idn', 'original_glcm_Imc1', 'original_glcm_Imc2', 'original_glcm_InverseVariance', 'original_glcm_JointAverage', 'original_glcm_JointEnergy', 'original_glcm_JointEntropy', 'original_glcm_MCC', 'original_glcm_MaximumProbability', 'original_glcm_SumAverage', 'original_glcm_SumEntropy', 'original_glcm_SumSquares', 'original_glrlm_GrayLevelNonUniformity', 'original_glrlm_GrayLevelNonUniformityNormalized', 'original_glrlm_GrayLevelVariance', 'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunEmphasis', 'original_glrlm_LongRunHighGrayLevelEmphasis', 'original_glrlm_LongRunLowGrayLevelEmphasis', 'original_glrlm_LowGrayLevelRunEmphasis', 'original_glrlm_RunEntropy', 'original_glrlm_RunLengthNonUniformity', 'original_glrlm_RunLengthNonUniformityNormalized', 'original_glrlm_RunPercentage', 'original_glrlm_RunVariance', 'original_glrlm_ShortRunEmphasis', 'original_glrlm_ShortRunHighGrayLevelEmphasis', 'original_glrlm_ShortRunLowGrayLevelEmphasis', 'original_glszm_GrayLevelNonUniformity', 'original_glszm_GrayLevelNonUniformityNormalized', 'original_glszm_GrayLevelVariance', 'original_glszm_HighGrayLevelZoneEmphasis', 'original_glszm_LargeAreaEmphasis', 'original_glszm_LargeAreaHighGrayLevelEmphasis', 'original_glszm_LargeAreaLowGrayLevelEmphasis', 'original_glszm_LowGrayLevelZoneEmphasis', 'original_glszm_SizeZoneNonUniformity', 'original_glszm_SizeZoneNonUniformityNormalized', 'original_glszm_SmallAreaEmphasis', 'original_glszm_SmallAreaHighGrayLevelEmphasis', 'original_glszm_SmallAreaLowGrayLevelEmphasis', 'original_glszm_ZoneEntropy', 'original_glszm_ZonePercentage', 'original_glszm_ZoneVariance', 'original_ngtdm_Busyness', 'original_ngtdm_Coarseness', 'original_ngtdm_Complexity', 'original_ngtdm_Contrast', 'original_ngtdm_Strength', 'original_gldm_DependenceEntropy', 'original_gldm_DependenceNonUniformity', 'original_gldm_DependenceNonUniformityNormalized', 'original_gldm_DependenceVariance', 'original_gldm_GrayLevelNonUniformity', 'original_gldm_GrayLevelVariance', 'original_gldm_HighGrayLevelEmphasis', 'original_gldm_LargeDependenceEmphasis', 'original_gldm_LargeDependenceHighGrayLevelEmphasis', 'original_gldm_LargeDependenceLowGrayLevelEmphasis', 'original_gldm_LowGrayLevelEmphasis', 'original_gldm_SmallDependenceEmphasis', 'original_gldm_SmallDependenceHighGrayLevelEmphasis', 'original_gldm_SmallDependenceLowGrayLevelEmphasis']       

    def process(self, arg):
        imageName = arg["imageName"]
        inputdir = arg["inputdir"]

        targetSet = arg["targetSet"]
        imagedir = arg["imagedir"]

        maskstilesdir = arg["maskstilesdir"]
        tile = arg["parameters"]["tile_size"]

        label = arg['label']


        #base = os.path.basename(imageName)
        #base = os.path.splitext(base)
        #imgoname = base[0]
        #fimage = inputdir + '/'+imagedir+'/' + targetSet + '/' + imageName
        #fmask = inputdir + '/'+maskstilesdir+'/' + str(tile) +"/"+ imgoname + ".nrrd"
        #image = sitk.ReadImage(fimage, sitk.sitkFloat32)
        #image_mask = sitk.ReadImage(fmask)

        image = self.image
        image_mask = self.mask

        #tic = time.time()        
        fenames = []
        #for index, row in df.iterrows():
        

        fenames, fevals = PyRadiomics.extraction(image, image_mask, label)
        fe = [arg['image'], arg['loc1'], arg['loc2'], arg['loc3'], arg['loc4'], label]+fevals+[arg['target']]

        #toc = time.time()
        #print("time",(toc - tic), imageName)

        if len(self.columns)==0:
            self.columns = fenames
            print(self.columns)
        return [fe]


class RAD(Description):
    def __init__(self, arg):
        super().__init__(arg)
        #self.columns = []
        #self.columns = ['original_firstorder_10Percentile', 'original_firstorder_90Percentile', 'original_firstorder_Energy', 'original_firstorder_Entropy', 'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis', 'original_firstorder_Maximum', 'original_firstorder_MeanAbsoluteDeviation', 'original_firstorder_Mean', 'original_firstorder_Median', 'original_firstorder_Minimum', 'original_firstorder_Range', 'original_firstorder_RobustMeanAbsoluteDeviation', 'original_firstorder_RootMeanSquared', 'original_firstorder_Skewness', 'original_firstorder_TotalEnergy', 'original_firstorder_Uniformity', 'original_firstorder_Variance', 'original_glcm_Autocorrelation', 'original_glcm_ClusterProminence', 'original_glcm_ClusterShade', 'original_glcm_ClusterTendency', 'original_glcm_Contrast', 'original_glcm_Correlation', 'original_glcm_DifferenceAverage', 'original_glcm_DifferenceEntropy', 'original_glcm_DifferenceVariance', 'original_glcm_Id', 'original_glcm_Idm', 'original_glcm_Idmn', 'original_glcm_Idn', 'original_glcm_Imc1', 'original_glcm_Imc2', 'original_glcm_InverseVariance', 'original_glcm_JointAverage', 'original_glcm_JointEnergy', 'original_glcm_JointEntropy', 'original_glcm_MCC', 'original_glcm_MaximumProbability', 'original_glcm_SumAverage', 'original_glcm_SumEntropy', 'original_glcm_SumSquares', 'original_glrlm_GrayLevelNonUniformity', 'original_glrlm_GrayLevelNonUniformityNormalized', 'original_glrlm_GrayLevelVariance', 'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunEmphasis', 'original_glrlm_LongRunHighGrayLevelEmphasis', 'original_glrlm_LongRunLowGrayLevelEmphasis', 'original_glrlm_LowGrayLevelRunEmphasis', 'original_glrlm_RunEntropy', 'original_glrlm_RunLengthNonUniformity', 'original_glrlm_RunLengthNonUniformityNormalized', 'original_glrlm_RunPercentage', 'original_glrlm_RunVariance', 'original_glrlm_ShortRunEmphasis', 'original_glrlm_ShortRunHighGrayLevelEmphasis', 'original_glrlm_ShortRunLowGrayLevelEmphasis']
        self.columns = ['original_firstorder_10Percentile', 'original_firstorder_90Percentile', 'original_firstorder_Energy', 'original_firstorder_Entropy', 'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis', 'original_firstorder_Maximum', 'original_firstorder_MeanAbsoluteDeviation', 'original_firstorder_Mean', 'original_firstorder_Median', 'original_firstorder_Minimum', 'original_firstorder_Range', 'original_firstorder_RobustMeanAbsoluteDeviation', 'original_firstorder_RootMeanSquared', 'original_firstorder_Skewness', 'original_firstorder_TotalEnergy', 'original_firstorder_Uniformity', 'original_firstorder_Variance', 'original_glcm_Autocorrelation', 'original_glcm_ClusterProminence', 'original_glcm_ClusterShade', 'original_glcm_ClusterTendency', 'original_glcm_Contrast', 'original_glcm_Correlation', 'original_glcm_DifferenceAverage', 'original_glcm_DifferenceEntropy', 'original_glcm_DifferenceVariance', 'original_glcm_Id', 'original_glcm_Idm', 'original_glcm_Idmn', 'original_glcm_Idn', 'original_glcm_Imc1', 'original_glcm_Imc2', 'original_glcm_InverseVariance', 'original_glcm_JointAverage', 'original_glcm_JointEnergy', 'original_glcm_JointEntropy', 'original_glcm_MCC', 'original_glcm_MaximumProbability', 'original_glcm_SumAverage', 'original_glcm_SumEntropy', 'original_glcm_SumSquares', 'original_glrlm_GrayLevelNonUniformity', 'original_glrlm_GrayLevelNonUniformityNormalized', 'original_glrlm_GrayLevelVariance', 'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunEmphasis', 'original_glrlm_LongRunHighGrayLevelEmphasis', 'original_glrlm_LongRunLowGrayLevelEmphasis', 'original_glrlm_LowGrayLevelRunEmphasis', 'original_glrlm_RunEntropy', 'original_glrlm_RunLengthNonUniformity', 'original_glrlm_RunLengthNonUniformityNormalized', 'original_glrlm_RunPercentage', 'original_glrlm_RunVariance', 'original_glrlm_ShortRunEmphasis', 'original_glrlm_ShortRunHighGrayLevelEmphasis', 'original_glrlm_ShortRunLowGrayLevelEmphasis', 'original_glszm_GrayLevelNonUniformity', 'original_glszm_GrayLevelNonUniformityNormalized', 'original_glszm_GrayLevelVariance', 'original_glszm_HighGrayLevelZoneEmphasis', 'original_glszm_LargeAreaEmphasis', 'original_glszm_LargeAreaHighGrayLevelEmphasis', 'original_glszm_LargeAreaLowGrayLevelEmphasis', 'original_glszm_LowGrayLevelZoneEmphasis', 'original_glszm_SizeZoneNonUniformity', 'original_glszm_SizeZoneNonUniformityNormalized', 'original_glszm_SmallAreaEmphasis', 'original_glszm_SmallAreaHighGrayLevelEmphasis', 'original_glszm_SmallAreaLowGrayLevelEmphasis', 'original_glszm_ZoneEntropy', 'original_glszm_ZonePercentage', 'original_glszm_ZoneVariance', 'original_ngtdm_Busyness', 'original_ngtdm_Coarseness', 'original_ngtdm_Complexity', 'original_ngtdm_Contrast', 'original_ngtdm_Strength', 'original_gldm_DependenceEntropy', 'original_gldm_DependenceNonUniformity', 'original_gldm_DependenceNonUniformityNormalized', 'original_gldm_DependenceVariance', 'original_gldm_GrayLevelNonUniformity', 'original_gldm_GrayLevelVariance', 'original_gldm_HighGrayLevelEmphasis', 'original_gldm_LargeDependenceEmphasis', 'original_gldm_LargeDependenceHighGrayLevelEmphasis', 'original_gldm_LargeDependenceLowGrayLevelEmphasis', 'original_gldm_LowGrayLevelEmphasis', 'original_gldm_SmallDependenceEmphasis', 'original_gldm_SmallDependenceHighGrayLevelEmphasis', 'original_gldm_SmallDependenceLowGrayLevelEmphasis']       


    def process(self, arg):
        imageName = arg["imageName"]
        inputdir = arg["inputdir"]
        #boundaryDataSet = arg["boundaryDataSet"]
        targetSet = arg["targetSet"]
        imagedir = arg["imagedir"]
        masksdir = arg["masksdir"]
        maskstilesdir = arg["maskstilesdir"]
        tile = arg["parameters"]["tile_size"]
        
        tilepercentage = arg["tilepercentage"]

        df = arg["df"]

        base = os.path.basename(imageName)
        base = os.path.splitext(base)
        imgoname = base[0]

        fimage = inputdir + '/'+imagedir+'/' + targetSet + '/' + imageName
        fmask = inputdir+'/'+maskstilesdir+'/'+str(tilepercentage)+"/"+str(tile)+"/"+ imgoname+".nrrd"
        
        image = sitk.ReadImage(fimage, sitk.sitkFloat32)
        image_mask = sitk.ReadImage(fmask)

        fe = []
        tic = time.time()        
        fenames = []
        for index, row in df.iterrows():
            label = row['idseg']
            fenames, fevals = PyRadiomics.extraction(image, image_mask, label)
            fe.append([row['image'], row['loc1'], row['loc2'], row['loc3'], row['loc4'], label]+fevals+[row['target']])
            #print(fenames)
        toc = time.time()
        print("time",(toc - tic), imageName)

        if len(self.columns)==0:
            self.columns = fenames
            print(self.columns)
        return fe

    """
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
        

        infocsv =  inputdir + '/'+maskstilesdir+'/'+str(tile)+"/"+'info.csv'
        df_rois = pd.read_csv(infocsv)
        ##print(df_rois.head())

        dataset = []
        
        for imageName in os.listdir(inputdir + '/'+imagedir+'/' + targetSet):
            df_filter = df_rois[(df_rois.image == imageName)]
            
            base = os.path.basename(imageName)
            base = os.path.splitext(base)
            imgoname = base[0]

            fimage = inputdir + '/'+imagedir+'/' + targetSet + '/' + imageName
            fmask = inputdir + '/'+maskstilesdir+'/' + str(tile) +"/"+ imgoname + ".nrrd"           
            image = sitk.ReadImage(fimage, sitk.sitkFloat32)
            image_mask = sitk.ReadImage(fmask)

            tic = time.time()        
            arg = []
            for index, row in df_filter.iterrows():
                dat = self.arg.copy()
                dat["imageName"] = imageName
                #dat["image_itk"] = image
                #dat["image_mask_itk"] = image_mask
                
                dat["label"] = row['idseg']
                dat["image"] = row['image']
                dat["loc1"] = row['loc1']
                dat["loc2"] = row['loc2']
                dat["loc3"] = row['loc3']
                dat["loc4"] = row['loc4']
                dat["target"] = row['target']
                arg.append(dat)
                #fenames, fevals = PyRadiomics.extraction(image, image_mask, label)
            
            rad = RadiomicsFeatures(image, image_mask)
            #ncpus = 15
            #ncpus = 10
            ncpus = multiprocessing.cpu_count()-1
            #ncpus = multiprocessing.cpu_count()
            pool = Pool(processes=ncpus)
            #rr = pool.map(self.process, arg)
            rr = pool.map(rad.process, arg)
            pool.close()
            for rs in rr:
                if len(dataset)==0:
                    dataset = rs
                else:
                    dataset = dataset + rs
            toc = time.time()
            print("time",(toc - tic), imageName)
            #print("dataset",dataset)
            columns = rad.columns

        #columns = ['original_firstorder_10Percentile', 'original_firstorder_90Percentile', 'original_firstorder_Energy', 'original_firstorder_Entropy', 'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis', 'original_firstorder_Maximum', 'original_firstorder_MeanAbsoluteDeviation', 'original_firstorder_Mean', 'original_firstorder_Median', 'original_firstorder_Minimum', 'original_firstorder_Range', 'original_firstorder_RobustMeanAbsoluteDeviation', 'original_firstorder_RootMeanSquared', 'original_firstorder_Skewness', 'original_firstorder_TotalEnergy', 'original_firstorder_Uniformity', 'original_firstorder_Variance', 'original_glcm_Autocorrelation', 'original_glcm_ClusterProminence', 'original_glcm_ClusterShade', 'original_glcm_ClusterTendency', 'original_glcm_Contrast', 'original_glcm_Correlation', 'original_glcm_DifferenceAverage', 'original_glcm_DifferenceEntropy', 'original_glcm_DifferenceVariance', 'original_glcm_Id', 'original_glcm_Idm', 'original_glcm_Idmn', 'original_glcm_Idn', 'original_glcm_Imc1', 'original_glcm_Imc2', 'original_glcm_InverseVariance', 'original_glcm_JointAverage', 'original_glcm_JointEnergy', 'original_glcm_JointEntropy', 'original_glcm_MCC', 'original_glcm_MaximumProbability', 'original_glcm_SumAverage', 'original_glcm_SumEntropy', 'original_glcm_SumSquares', 'original_glrlm_GrayLevelNonUniformity', 'original_glrlm_GrayLevelNonUniformityNormalized', 'original_glrlm_GrayLevelVariance', 'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunEmphasis', 'original_glrlm_LongRunHighGrayLevelEmphasis', 'original_glrlm_LongRunLowGrayLevelEmphasis', 'original_glrlm_LowGrayLevelRunEmphasis', 'original_glrlm_RunEntropy', 'original_glrlm_RunLengthNonUniformity', 'original_glrlm_RunLengthNonUniformityNormalized', 'original_glrlm_RunPercentage', 'original_glrlm_RunVariance', 'original_glrlm_ShortRunEmphasis', 'original_glrlm_ShortRunHighGrayLevelEmphasis', 'original_glrlm_ShortRunLowGrayLevelEmphasis']
        #self.columns = self.makecolumns()
        

        #columns = self.columns
        

        #columns = list(self.columns)
        #print("columns save", columns)
        xcolumns = ["image","loc1","loc2","loc3","loc4","idseg"]+columns+["target"]
        df = pd.DataFrame(data=dataset)
        df.columns = xcolumns
        df.to_csv(outputdir+"/"+file, index=False)
    """
    
if __name__ == "__main__":
    pass
