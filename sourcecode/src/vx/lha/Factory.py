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

#from  PyRadiomics import *

from datetime import datetime

from Util import *
from RAD import *
from LBP import *
              
class Factory:
    @staticmethod
    def descriptor(name, arg):
        model = None
        if name=="LBP":
            model = LBP(arg)
        elif name=="RAD":
            model = RAD(arg)
            
        return model

    @staticmethod
    def classifier(name, arg):
        model = None
        if name=="asdfasdfnlasdflas":
            model = LBP(arg)
        return model



