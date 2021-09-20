import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from Util import *


def join_by_columns(df1, df2, keys):
    return pd.concat([df1, df2], ignore_index=True)

def join_csv(data):
    da = data["da"]
    db = data["db"]
    dc = data["dc"]
    files = data["files"]
    Util.makedir(dc)
    #da<-db
    for file in files:
        fa = file["fa"]
        fb = file["fb"]
        #if file.endswith(".csv"):
        print(fa, fb)
        if  os.path.exists(da+"/"+fa) and os.path.exists(db+"/"+fb):
            dfa = pd.read_csv(da+"/"+fa)
            dfb = pd.read_csv(db+"/"+fb)
            ra, ca = dfa.shape
            rb, cb = dfb.shape
            if ra == rb:
                keys = ["image","loc1","loc2","loc3","loc4","idseg","target"]
                result = pd.merge(dfa, dfb, how="left", on=keys)
                result.to_csv(dc+"/"+fa, index=False)

    os.popen('cp '+da+"/featureinfo.json"+' '+dc+"/featureinfo.json")

def join_features(arg):
    da,db,dc =  arg["inputdira"], arg["inputdirb"], arg["outputdir"]
    features_a = Util.read(da+"/featureinfo.json")
    features_b = Util.read(db+"/featureinfo.json")
    files = []
    for fa in features_a:
        for fb in features_b:
            files.append({"fa":fa["train"]["file"], "fb":fb["train"]["file"]})
            files.append({"fa":fa["test"]["file"], "fb":fb["test"]["file"]})

    data={
            "da":da,
            "db":db,
            "dc":dc,
            "files":files
        }
    join_csv(data)


def join_datasets(arg):
    dsa,dsb,dsc =  arg["inputdira"], arg["inputdirb"], arg["outputdir"]
    
    keys = ["image","loc1","loc2","loc3","loc4","idseg","target"]
    Util.makedir(dsc)
    features_a = Util.read(dsa+"/featureinfo.json")
    for fa in features_a:
        fatrain_name, fatest_name = fa["train"]["file"], fa["test"]["file"]
        fbtrain_name, fbtest_name = fa["train"]["file"], fa["test"]["file"]

        fatrain, fatest = dsa+"/"+fatrain_name, dsa+"/"+fatest_name
        fbtrain, fbtest = dsb+"/"+fbtrain_name, dsb+"/"+fbtest_name

        if os.path.exists(fatrain) and os.path.exists(fatest) and  os.path.exists(fbtrain) and os.path.exists(fbtest):
            dfa = pd.concat([pd.read_csv(fatrain), pd.read_csv(fatest)], ignore_index=True)
            dfb = pd.concat([pd.read_csv(fbtrain), pd.read_csv(fbtest)], ignore_index=True)
            dfc = pd.concat([dfa, dfb], ignore_index=True)
            
            #dfc = pd.merge(dfa, dfb, how="left", on=keys)
            y = dfc["target"]

            dfc_train, dfc_test, _, _ = train_test_split(
                                    dfc, y, stratify=y, test_size=0.3, random_state=7)

            dfc_train.to_csv(dsc+"/"+fatrain_name, index=False)
            dfc_test.to_csv(dsc+"/"+fatest_name, index=False)
            
    os.popen('cp '+dsa+"/featureinfo.json"+' '+dsc+"/featureinfo.json")

def join_resample(arg):
    dsa,dsb,dsc =  arg["inputdira"], arg["inputdirb"], arg["outputdir"]

    keys = ["image","loc1","loc2","loc3","loc4","idseg","target"]
    Util.makedir(dsc)
    features_a = Util.read(dsa+"/featureinfo.json")
    for fa in features_a:
        fatrain_name, fatest_name = fa["train"]["file"], fa["test"]["file"]
        
        fatrain, fatest = dsa+"/"+fatrain_name, dsa+"/"+fatest_name
        
        if os.path.exists(fatrain) and os.path.exists(fatest):
            dfa = pd.concat([pd.read_csv(fatrain), pd.read_csv(fatest)], ignore_index=True)
            
            #dfc = pd.merge(dfa, dfb, how="left", on=keys)
            y = dfa["target"]

            dfc_train, dfc_test, _, _ = train_test_split(
                                    dfa, y, stratify=y, test_size=0.3, random_state=7)

            dfc_train.to_csv(dsc+"/"+fatrain_name, index=False)
            dfc_test.to_csv(dsc+"/"+fatest_name, index=False)
            
    os.popen('cp '+dsa+"/featureinfo.json"+' '+dsc+"/featureinfo.json")


def join_predicted(arg):
    keys = ["image","loc1","loc2","loc3","loc4","idseg","target"]
    #Util.makedir(dsc)
    fileclass = Util.read(arg["inputdir"]+"/"+arg["file"])
    classnames = Util.read(arg["inputdir"]+"/category_name.json")
    intclassname = {int(k):v for k,v in classnames.items()}
    print(intclassname)

    for fa in fileclass:
        #print(fa["testf"], fa["evals"])
        dfa = pd.read_csv(arg["inputdir"]+"/"+fa["testf"])
        #dfb = dfa[keys]
        print(fa["testf"])
        for k,v in fa["evals"].items():
            #print(k, v["ytrue"], v["ypred"])
            dfb = dfa[keys]
            dfb["target_predicted"] = v["ypred"]
            #dfb['target_pred'].replace(str(0),"nopleura")
            dfb['target_predicted'] = dfb['target_predicted'].map(intclassname)
            print(k, dfb)
            dfb.to_csv(arg["outputdir"]+"/predicted_"+k+"_"+fa["testf"], index=False)


            
if __name__ == "__main__":    
    with open(sys.argv[1], mode='r') as jsond:
        #print (jsdata)
        args = ujson.load(jsond)
        print(args)
        #outputdir = inputdira<-inputdirb
        for arg in args:
            if arg["type"] == "joinfeatures":
                join_features(arg)
            elif arg["type"] == "joindatasets":
                join_datasets(arg)
                #joinfeatures(inputdira, inputdirb, outputdir)
            elif arg["type"] == "joinresample":
                join_resample(arg)
            elif arg["type"] == "joinpredicted":
                join_predicted(arg)


