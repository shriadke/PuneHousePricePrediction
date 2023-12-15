import os
import numpy as np
import pandas as pd
import pickle
import json
import math
import re

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import TargetEncoder

#Consider mean area in case of a range
def get_area(area):
    if str(area).isdigit():
        return float(area)
    else:
        s = [float(s) for s in re.findall(r'-?\d+\.?\d*', str(area))]
        if len(s) == 0:
            return np.NAN
        else:
            return np.mean(s)

def get_bedroom_size(prop_type):
    if "bhk" in str(prop_type):
        s = [float(s) for s in re.findall(r'-?\d+\.?\d*', prop_type)]
        if len(s) == 0:
            return np.NAN
        else:
            return sum(s)
    return np.NAN

def get_township_size(ts_area):

    if not np.isnan(ts_area):
        if 0 < int(ts_area) <= 25:
            return "small"
        elif 25 < int(ts_area) <= 250:
            return "medium"
        elif 250 < int(ts_area) < 25000:
            return "large"
    return "unknown"

def get_ts_size_map():
    return {
            "unknown" : 0,
            "small" : 1,
            "medium":2,
            "large":3
        }

def get_loc_trend_map():
    return {
            "bavdhan" : 6.5 , "mahalunge" : 15.2 , "balewadi" :  8.3 , "ravet" : 3.6  , "baner" : 14.7  , "kharadi" :  11.5 , "koregaon park" :  13.7 , "keshav nagar" :  -2.4 , 
            "kirkatwadi sinhagad road" :  2.4 , "akurdi" :  -0.7 , "tathawade" : 4.0  , "hadapsar" :  25.2 , "kiwale" : 5.7  , "kalyani nagar" : 24.5  , "pisoli" :  -4.0 , "manjri" : 0.0  ,
            "handewadi" : 2.1  , "mundhwa" : 0.0  , "nibm" : 1.0  , "bt kawade rd" : 3.6  , "undri" : 2.5  , "karvenagar" : 2.0  , "magarpatta" : 12.1  , "hinjewadi" : 8.0  , "vimannagar" : 17.0  , 
            "wadgaon sheri" : 38.4  , "susgaon" : 3.1  , "mohammadwadi" : 4.3  , "dhanori" : 4.3  , "lonavala" : 0  , "talegoan" : 0 
        }

def get_val_tag_map():
    return {
            0    : "unknown",
            1    : "affordable",
            2    : "midrange",
            3    : "premium"
        }

def get_area_idx_dict():
    return {
            "bavdhan" : 3 , "mahalunge" : 3 , "balewadi" :  3 , "ravet" : 1  , "baner" : 3  , "kharadi" :  3 , "koregaon park" :  3 , "keshav nagar" :  2 , 
            "kirkatwadi sinhagad road" :  1 , "akurdi" :  2 , "tathawade" : 2  , "hadapsar" :  3 , "kiwale" : 1  , "kalyani nagar" : 3  , "pisoli" :  1 , "manjri" : 2  ,
            "handewadi" : 1  , "mundhwa" : 2  , "nibm" : 3  , "bt kawade rd" : 2  , "undri" : 2  , "karvenagar" : 3  , "magarpatta" : 3  , "hinjewadi" : 2  , "vimannagar" : 3  , 
            "wadgaon sheri" : 3  , "susgaon" : 2  , "mohammadwadi" : 3  , "dhanori" : 2  , "lonavala" : 0  , "talegoan" : 0 
        }

def get_loc_tag_map():
    loc_tag_map = {}
    val_tag_map = get_val_tag_map()
    for key,val in get_area_idx_dict().items():
        loc_tag_map[key] = val_tag_map[val]
    return loc_tag_map