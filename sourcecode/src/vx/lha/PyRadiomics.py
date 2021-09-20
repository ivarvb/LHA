#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Ivar
"""

import SimpleITK as sitk
import numpy as np
import radiomics
from radiomics.featureextractor import RadiomicsFeatureExtractor

import numpy as np
import SimpleITK as sitk

class PyRadiomics:
    @staticmethod
    def extraction(image, image_mask, label):

        settings = {}
        #settings['binWidth'] = 25

        settings['verbose'] = False
        settings['distances']  = [1, 2, 5, 10]
        extractor = RadiomicsFeatureExtractor(**settings)
        #binWidth=20

        # Enable a filter (in addition to the 'Original' filter already enabled)
        #extractor.enableInputImageByName('LoG')
        #extractor.enableInputImages(wavelet= {'level': 2})


        #extractor.enableAllFeatures()
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName('firstorder')#19
        extractor.enableFeatureClassByName('glcm')#24
        extractor.enableFeatureClassByName('glrlm')#16   
        extractor.enableFeatureClassByName('glszm')#16
        extractor.enableFeatureClassByName('ngtdm')#5
        extractor.enableFeatureClassByName('gldm')#14


        
        #extractor.enableFeatureClassByName('glszm')
        #extractor.enableFeatureClassByName('gldm')
        #extractor.enableFeatureClassByName('ngtdm')



        result = extractor.execute(image, image_mask, label=label)
        #result = extractor.execute(image, image_mask)
        features_names = []
        #features_values = {}
        features_values = []
        
        for k,v in result.items():
            if k.startswith('original_'): 
                features_names.append(k)
                features_values.append(v.tolist())
        #print("RAD size features", len(features_values))
                #features_values.append(v.tolist()[0])
        """ 
        for k,v in features_values.items():
            print(k, v)
        """
        return features_names, features_values


    @staticmethod
    def make_mask(image, labels):
        # Load the image
        #image = sitk.ReadImage(imagefi, sitk.sitkFloat32)

        # Build full mask
        im_size = np.array(image.GetSize())[::-1]
        #ma_arr = np.ones(im_size)
        ma_arr = np.zeros(im_size, dtype=int)
        
        #for r in range(labels):


        for r in range(len(labels)):
            node = labels[r]
            for i in range(len(node[0])):
                #print(r, i)
                x = node[0][i]
                y = node[1][i]
                ma_arr[x][y] = (r+1)
        
        
        """ 
        iss = {}
        for y in range(len(ma_arr)):
            for x in range(len(ma_arr[0])):
                if ma_arr[y][x]>0:
                    #print("(",ma_arr[y][x],")", end='')
                    iss[ma_arr[y][x]] = 0
        ddd = [key for key in sorted(iss)]
        print()
        print(ddd, len(labels), np.max(ma_arr))
        print()
        """


        ma = sitk.GetImageFromArray(ma_arr)
        ma.CopyInformation(image)

        #sitk.WriteImage(ma, 'mask_mae.nrrd', True)  # True specifies it can use compression
        #sitk.WriteImage(ma, imagefo, True)  # True specifies it can use compression

        return ma

    @staticmethod
    def write_mask(mask, imagefo):
        #sitk.WriteImage(ma, 'mask_mae.nrrd', True)  # True specifies it can use compression
        sitk.WriteImage(mask, imagefo, True)  # True specifies it can use compression


    @staticmethod
    def execute(imagefi, imagefo, labels):
        # Load the image
        image = sitk.ReadImage(imagefi, sitk.sitkFloat32)
        image_mask = PyRadiomics.make_mask(image, labels)
        PyRadiomics.write_mask(image_mask, imagefo)

        vfeatures = []
        for r in range(len(labels)):
            rsn, rsv = PyRadiomics.extraction(image, image_mask, r+1)
            vfeatures.append(rsv)
            #print("rs", rsn, rsv)
        return vfeatures 

def main():
    # Load the image
    image = sitk.ReadImage('B 2009 8854 A_1x.tiff', sitk.sitkFloat32)
    image_mask = sitk.ReadImage('mask_mae.nrrd')
    label=5
    fe = PyRadimics.extraction(image, image_mask, label)
    for k,v in fe.items():
        print(k, v)


if __name__ == "__main__":
    main()







""" 
    # Load the image
    image = sitk.ReadImage('B 2009 8854 A_1x.tiff', sitk.sitkFloat32)

    # Build full mask
    im_size = np.array(image.GetSize())[::-1]
    ma_arr = np.zeros(im_size)
    #ma_arr = np.ones(im_size)
    ma_arr[0][0] = 0
    ma_arr[0][1] = 2
    ma_arr[0][2] = 2
    ma_arr[0][3] = 2
    ma_arr[1][0] = 2
    for i in range(2,20):
        for j in range(2,20):
            ma_arr[i][j] = 5
    for i in range(20,40):
        for j in range(20,40):
            ma_arr[i][j] = 4
    print(ma_arr)

    ma = sitk.GetImageFromArray(ma_arr)
    ma.CopyInformation(image)


    sitk.WriteImage(ma, 'mask_mae.nrrd', True)  # True specifies it can use compression



    image_mask = sitk.ReadImage('mask_mae.nrrd')
    label=5

    #extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=20, sigma=[1, 
    #             2, 3], verbose=True)
    extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=20, distances  = [1, 2, 5, 10])
    # Disable all feature classes, save firstorder
    #extractor.disableAllFeatures()
    #extractor.enableFeatureClassByName('glrlm')


    #extractor.enableAllFeatures()

    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('glsz')
    extractor.enableFeatureClassByName('gldm')
    extractor.enableFeatureClassByName('ngtdm')



    result = extractor.execute(image, image_mask, label=label)
    #result = extractor.execute(image, image_mask)
    features_names = []
    features_values = {}
    for k,v in result.items():
        if k.startswith('original_'): 
            features_names.append(k)
            #features_values+=[v]
            features_values[k]= v.tolist()
            print(k)

    for k,v in features_values.items():
        print(k, v)
 """