#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:41:35 2021

@author: oscar
"""
import multiprocessing
import sys
import ujson
from multiprocessing import Pool, Manager, Process, Lock


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from Util import *
from Factory import *

class Description:
    def __init__(self, arg):
        self.arg = arg
        self.columns = []
        #manager = Manager()
        #self.columns = manager.list()
        #self.arg["parameters"]


    def makecolumns(self, arg):
        return []   

    def process(self):
        pass

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
        

        infocsv =  inputdir + '/'+maskstilesdir+'/'+str(tilepercentage)+"/"+str(tile)+"/"+'info.csv'
        df_rois = pd.read_csv(infocsv)
        ##print(df_rois.head())
        arg = []
        for imageName in os.listdir(inputdir + '/'+imagedir+'/' + targetSet):
            dat = self.arg.copy()
            df_filter = df_rois[(df_rois.image == imageName)]
            dat["imageName"] = imageName
            dat["df"] = df_filter
            arg.append(dat)

        dataset = []
        
        #ncpus = 15
        #ncpus = multiprocessing.cpu_count()-1
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
        
        
        #columns = ['original_firstorder_10Percentile', 'original_firstorder_90Percentile', 'original_firstorder_Energy', 'original_firstorder_Entropy', 'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis', 'original_firstorder_Maximum', 'original_firstorder_MeanAbsoluteDeviation', 'original_firstorder_Mean', 'original_firstorder_Median', 'original_firstorder_Minimum', 'original_firstorder_Range', 'original_firstorder_RobustMeanAbsoluteDeviation', 'original_firstorder_RootMeanSquared', 'original_firstorder_Skewness', 'original_firstorder_TotalEnergy', 'original_firstorder_Uniformity', 'original_firstorder_Variance', 'original_glcm_Autocorrelation', 'original_glcm_ClusterProminence', 'original_glcm_ClusterShade', 'original_glcm_ClusterTendency', 'original_glcm_Contrast', 'original_glcm_Correlation', 'original_glcm_DifferenceAverage', 'original_glcm_DifferenceEntropy', 'original_glcm_DifferenceVariance', 'original_glcm_Id', 'original_glcm_Idm', 'original_glcm_Idmn', 'original_glcm_Idn', 'original_glcm_Imc1', 'original_glcm_Imc2', 'original_glcm_InverseVariance', 'original_glcm_JointAverage', 'original_glcm_JointEnergy', 'original_glcm_JointEntropy', 'original_glcm_MCC', 'original_glcm_MaximumProbability', 'original_glcm_SumAverage', 'original_glcm_SumEntropy', 'original_glcm_SumSquares', 'original_glrlm_GrayLevelNonUniformity', 'original_glrlm_GrayLevelNonUniformityNormalized', 'original_glrlm_GrayLevelVariance', 'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunEmphasis', 'original_glrlm_LongRunHighGrayLevelEmphasis', 'original_glrlm_LongRunLowGrayLevelEmphasis', 'original_glrlm_LowGrayLevelRunEmphasis', 'original_glrlm_RunEntropy', 'original_glrlm_RunLengthNonUniformity', 'original_glrlm_RunLengthNonUniformityNormalized', 'original_glrlm_RunPercentage', 'original_glrlm_RunVariance', 'original_glrlm_ShortRunEmphasis', 'original_glrlm_ShortRunHighGrayLevelEmphasis', 'original_glrlm_ShortRunLowGrayLevelEmphasis']
        #self.columns = self.makecolumns()
        columns = self.columns
        #columns = list(self.columns)
        #print("columns save", columns)
        xcolumns = ["image","loc1","loc2","loc3","loc4","idseg"]+columns+["target"]
        df = pd.DataFrame(data=dataset)
        df.columns = xcolumns
        df.to_csv(outputdir+"/"+file, index=False)

    @staticmethod
    def start(inputdir, outputdir, template):       
        result = []
        traintest = ["test","train"]
        #traintest = ["whole"]
        for r in template:
            #e = r["parameters"]["erode"]
            
            row = {tt:{} for tt in traintest}
            row["name"] = r["name"]
            for targetSet in traintest:
                r["inputdir"] = inputdir
                r["outputdir"] = outputdir
                #if r["tiledir"]!="":
                #    r["tiledir"] = str(r["tiledir"])+"/"

                #r["boundaryDataSet"] = "erode_radius_"+str(e)
                #r["boundaryDataSet_id"] = str(e)
                r["targetSet"] = targetSet
                r["label"] = "{"+Util.getLabel(r["parameters"])+"}"
                r["file"] = Util.getFileName(r)

                c = r.copy()
                    
                row[targetSet] = c
                lbpo = Factory.descriptor(c["name"], c)
                lbpo.execute()

            result.append(row) 
        
        Util.write(outputdir+"/featureinfo.json",result)
        print("Complete: Feature extraction")

if __name__ == "__main__":

    with open(sys.argv[1], mode='r') as jsond:
        #print (jsdata)
        argsdata = ujson.load(jsond)
        #print(argsdata)

        for args in argsdata:
            inputdir = args["inputdir"]
            outputdir = args["outputdir"]
            Util.makedir(outputdir)
            Util.write(outputdir+"/config.json",args)
            template = []
            
            descriptor = args["descriptor"]
            for parameters in args["parameters"]:
                template.append({
                    "name":descriptor,
                    "imagedir":args["grayscaledir"],
                    "masksdir":args["masksdir"],
                    "maskstilesdir":args["maskstilesdir"],
                    "tilepercentage":args["tilepercentage"],
                    "parameters":parameters,
                })
            #print(template)
            Description.start(inputdir, outputdir, template)

        #ds_LBP("dataset_2")
        #ds_RAD("dataset_2")
