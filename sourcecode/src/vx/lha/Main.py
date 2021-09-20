#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Ivar
"""

from Description import *
from Classification import *


if __name__ == "__main__":
    inputdir =  "../../../../data/LHA/dataset_1"
    outputdir = inputdir+"/csv/exp/"+Util.now()
    
    template = [
        {
            "name":"RAD",
            "imagedir":"images_cleaned",
            "maskdir":"seg/seg_window",
            "masksubsetdir":"100",
            "parameters":{"tile_size":100},
            "erode":[30]
        },
        {
            "name":"RAD",
            "imagedir":"images_cleaned",
            "maskdir":"seg/seg_window",
            "masksubsetdir":"200",
            "parameters":{"tile_size":200},
            "erode":[30]
        },
        {
            "name":"RAD",
            "imagedir":"images_cleaned",
            "maskdir":"seg/seg_window",
            "masksubsetdir":"300",
            "parameters":{"tile_size":300},
            "erode":[30]
        },








        {
            "name":"LBP",
            "imagedir":"images_cleaned",
            "maskdir":"masks",
            "masksubsetdir":"100",
            "parameters":{"tile_size":100, "radius":5},
            "erode":[30]
        },
        {
            "name":"LBP",
            "imagedir":"images_cleaned",
            "maskdir":"masks",
            "masksubsetdir":"200",
            "parameters":{"tile_size":200, "radius":5},
            "erode":[30]
        },
        {
            "name":"LBP",
            "imagedir":"images_cleaned",
            "maskdir":"masks",
            "masksubsetdir":"300",
            "parameters":{"tile_size":300, "radius":5},
            "erode":[30]
        },








        {
            "name":"LBP",
            "imagedir":"images_cleaned",
            "maskdir":"masks",
            "masksubsetdir":"100",
            "parameters":{"tile_size":100, "radius":10},
            "erode":[30]
        },
        {
            "name":"LBP",
            "imagedir":"images_cleaned",
            "maskdir":"masks",
            "masksubsetdir":"200",
            "parameters":{"tile_size":200, "radius":10},
            "erode":[30]
        },
        {
            "name":"LBP",
            "imagedir":"images_cleaned",
            "maskdir":"masks",
            "masksubsetdir":"300",
            "parameters":{"tile_size":300, "radius":10},
            "erode":[30]
        },
    ]

    inputdir =  "../../../../data/LHA/dataset_2"
    outputdir = inputdir+"/csv/exp/"+Util.now()


    Description.start(inputdir, outputdir, template)
    Classification.start(outputdir, outputdir)
    
    print("Complete in {}".format(outputdir))
    
    