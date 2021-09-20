import sys
import os
import pandas as pd
import numpy as np

from Util import *

if __name__ == "__main__":    
    with open(sys.argv[1], mode='r') as jsond:
        #print (jsdata)
        args = ujson.load(jsond)
        #print(args)
        #outputdir = inputdira<-inputdirb
        for arg in args:
            if arg["type"] == "simple":
                Util.curvePlot(arg)             
            if arg["type"] == "fromcvsfile":
                Util.curvePlotFromCSV(arg)
            if arg["type"] == "fromdir":
                Util.curvePlotFromDIR(arg)


