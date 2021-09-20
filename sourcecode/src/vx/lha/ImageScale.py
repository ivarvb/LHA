# Required Libraries
import cv2
import os
import numpy as np
from os import listdir
from os.path import isfile, join
from pathlib import Path
import argparse
import numpy

from Util import *
def image_scale(inputdir, outputdir, scale):
    Util.makedir(outputdir)
    for imageName in os.listdir(inputdir):
        img = cv2.imread(inputdir +"/"+ imageName)
    
        # Define a resizing Scale
        # To declare how much to resize
        resize_scaling = scale
        resize_width = int(img.shape[1] * resize_scaling/100)
        resize_hieght = int(img.shape[0] * resize_scaling/100)
        resized_dimentions = (resize_width, resize_hieght)
    
        # Create resized image using the calculated dimentions
        resized_image = cv2.resize(img, resized_dimentions,
                                interpolation=cv2.INTER_AREA)
    
        # Save the image in Output Folder
        cv2.imwrite(outputdir+"/"+imageName, resized_image)
    
    print("Images resized Successfully")

if __name__ == "__main__":    
    # 200 300 400
    # SVM
    pathm = "/mnt/sda6/software/projects/data/lha/dataset_3/build/csv/LBP+RADV2"


    """ 
    inputdir = pathm+"/visualization/SVCRBF_200_10"
    outputdir = pathm+"/visualization_scale/SVCRBF_200_10"
    scale = 20
    image_scale(inputdir, outputdir, scale)

    inputdir = pathm+"/visualization/SVCRBF_300_10"
    outputdir = pathm+"/visualization_scale/SVCRBF_300_10"
    scale = 20
    image_scale(inputdir, outputdir, scale)

    inputdir = pathm+"/visualization/SVCRBF_400_10"
    outputdir = pathm+"/visualization_scale/SVCRBF_400_10"
    scale = 20
    image_scale(inputdir, outputdir, scale)

    inputdir = pathm+"/visualization/SVCRBF_500_10"
    outputdir = pathm+"/visualization_scale/SVCRBF_500_10"
    scale = 20
    image_scale(inputdir, outputdir, scale)
    """


    # GBD
    inputdir = pathm+"/visualization/XGBC_200_10"
    outputdir = pathm+"/visualization_scale/XGBC_200_10"
    scale = 20
    image_scale(inputdir, outputdir, scale)

    inputdir = pathm+"/visualization/XGBC_300_10"
    outputdir = pathm+"/visualization_scale/XGBC_300_10"
    scale = 20
    image_scale(inputdir, outputdir, scale)

    inputdir = pathm+"/visualization/XGBC_400_10"
    outputdir = pathm+"/visualization_scale/XGBC_400_10"
    scale = 20
    image_scale(inputdir, outputdir, scale)

    inputdir = pathm+"/visualization/XGBC_500_10"
    outputdir = pathm+"/visualization_scale/XGBC_500_10"
    scale = 20
    image_scale(inputdir, outputdir, scale)




