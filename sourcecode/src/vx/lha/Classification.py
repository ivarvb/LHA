#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


from xgboost import XGBClassifier
import xgboost as xgb
from thundersvm import SVC

from Util import *
from multiprocessing import Pool, Manager, Process, Lock



from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
#from sklearn.feature_selection import SelectKBest

from sklearn.svm import LinearSVC


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import mutual_info_classif

from sklearn.feature_selection import RFECV

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV


from sklearn import datasets, linear_model
from genetic_selection import GeneticSelectionCV


from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import SequentialFeatureSelector


from sklearn.svm import SVR

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import train_test_split


from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler


from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score

#from pickle import dump, load

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.model_selection import cross_validate


from sklearn.base import clone

from sklearn.linear_model import SGDClassifier

import time

class Classification:

    def __init__ (self, dat):
        self.dat = dat
        
        
        self.fetures = Util.read(self.dat["inputdir"]+"/featureinfo.json")

        #self.norms = ["None", "std","minmax"]
        #self.norms = ["minmax"]


        self.evals_wp = [{} for i in range(len(self.fetures)*len(self.dat["norms"]))]
        self.load_wp = 0

        manager = Manager()
        self.evals = manager.list([{} for i in range(len(self.fetures)*len(self.dat["norms"]))])
        self.load = manager.Value('i', 0)

    def get_features_selected(self, dframe, dfs):
        #features = dfs["features"]
        features = dframe.columns.tolist()
        selected = dfs["selected"]
        #cols = [features[index] for index in selected]
        return dframe.iloc[:, selected]

    def process(self, data):
       
        trainX, trainY, testX, testY = self.getDataSets(self.dat["inputdir"], data["trainf"], data["testf"])
        featuresnames = trainX.columns.tolist()
        
        """
        trainX["target"] = trainY
        testX["target"] = testY
        Xwwhole = pd.concat([trainX, testX])
        Ywwhole = Xwwhole["target"].to_numpy()
        Xwwhole = Xwwhole.drop(["target"], axis=1)

        #Xwwhole, Ywwhole = RandomOverSampler(random_state=7).fit_resample(Xwwhole, Ywwhole)
        #Xwwhole, Ywwhole = SMOTE().fit_resample(Xwwhole, Ywwhole)
        trainX, testX, trainY, testY = train_test_split(
                    Xwwhole, Ywwhole, stratify=Ywwhole, test_size=0.3)
        """



        clsr =  Classification.classifiers()

        CX, CY, c_skf = None, None, None
        if self.dat["type"]=="CROSS":
            trainX["target"] = trainY
            testX["target"] = testY
            CX = pd.concat([trainX, testX])
            CY = CX["target"].to_numpy()
            CX = CX.drop(["target"], axis=1)
            
            #print("sampling 11", len(CX), len(CY))
            #print("<trainX, testX>", CX)
            ## scaling       
            if data["norm"]!="None":
                scaler = Classification.getScale(data["norm"]) 
                CX = scaler.fit_transform(CX)
                ## save the scaler
                # dump(scaler, open('scaler.pkl', 'wb'))
                # scaler = load(open('scaler.pkl', 'rb'))


            ## balancing
            if self.dat["balancing"]=="RandomOverSampler":
                CX, CY = RandomOverSampler(random_state=7).fit_resample(CX, CY)
            elif self.dat["balancing"]=="SMOTE":
                CX, CY = SMOTE().fit_resample(CX, CY)

            ## sampling
            if data["sampling"]<1.0:
                CX, _, CY, _ = train_test_split(
                        CX, CY, stratify=CY, test_size=1.0-data["sampling"])
                #print("sampling", len(CX), len(CY))
                #print("samplingXX", CX, CY)

            #print("CY.value_counts()\n",CY.value_counts())

        ## scaling        
        trainX, testX = self.scaleData(trainX, testX, data["norm"])

        ## balancing
        if self.dat["balancing"]=="RandomOverSampler":
            trainX, trainY = RandomOverSampler(random_state=7).fit_resample(trainX, trainY)
        elif self.dat["balancing"]=="SMOTE":
            trainX, trainY = SMOTE().fit_resample(trainX, trainY)

        ## sampling
        if data["sampling"]<1.0:
            trainX, _, trainY, _ = train_test_split(
                    trainX, trainY, stratify=trainY, test_size=1.0-data["sampling"], random_state=7)

        #if self.dat["type"]!="CROSS":
        #    print("trainY.value_counts()\n",trainY.value_counts())


        #rus = RandomUnderSampler(random_state=7)
        #trainX, trainY = rus.fit_resample(trainX, trainY)
        #print("trainY.value_counts()\n",trainY.value_counts())


        
        ## execute feature selection        
        if self.dat["type"]=="FESE" and self.dat["featureselection"]["name"]!="None":

            estimator =  clsr[self.dat["featureselection"]["estimator"]]["model"]
            #print("estimator", estimator)
            dfs = self.process_feature_selection(estimator, featuresnames, trainX, trainY, data["trainf"], data["norm"])
            trainX = self.get_features_selected(trainX, dfs)
            testX = self.get_features_selected(testX, dfs)

        ## use feature selected
        if self.dat["usefeatureselected"]:
            normshift = {"None":0,"std":1,"minmax":2}
            file = self.dat["outputdir"]+"/featureselection/"+self.dat["featureselection"]["name"]+"_"+self.dat["featureselection"]["estimator"]+"_"+str(normshift[data["norm"]])+"_"+data["trainf"]
            if os.path.exists(file):
                dfs = Util.read(file)
                trainX = self.get_features_selected(trainX, dfs)
                testX = self.get_features_selected(testX, dfs)
                print("use best parameters", dfs)
    
        evals = {}
        

        for name, argms in self.dat["classifiers"].items():
            begin_t = time.time()
            if name in clsr:
                m = clsr[name]
                clf = m["model"]
                clf = clf.set_params(**argms["modelparameters"])

                fiparam = self.dat["outputdir"]+"/bestparameters/"+data["label"]+"_"+self.dat["balancing"]+"_"+name+"_"+data["norm"]+"_"+str(data["xval"])+".json"
                
                #read best parameters
                if self.dat["usebestparameters"] and os.path.exists(fiparam):
                    bestpar = Util.read(fiparam)
                    clf = clf.set_params(**bestpar)

                if self.dat["type"]=="GRID":
                    cvs = ShuffleSplit(10, test_size=0.2, train_size=0.2, random_state=7)
                    #cvs = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
                    clf = GridSearchCV(clf, argms["gridsearch"]["gridparameters"],
                                    scoring=argms["gridsearch"]["scoring"],
                                    n_jobs=argms["gridsearch"]["n_jobs"],
                                    
                                    cv=cvs)


                if self.dat["type"]=="CROSS":
                    #clf = svm.SVC(kernel='linear', C=1, random_state=7)
                    
                    scores = {"acc":0.0, "f1":0.0, "roc":0.0, "jac":0.0, "pre":0.0, "rec":0.0}
                    stds = {"acc":0.0, "f1":0.0, "roc":0.0, "jac":0.0, "pre":0.0, "rec":0.0}
                    scores_ar = {"acc":[], "f1":[], "roc":[], "jac":[], "pre":[], "rec":[]}
                    ytrue, ypred = [], []

                    c_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
                    scorer = ['accuracy','precision', 'recall', 'f1','jaccard','roc_auc']
                    scoress = cross_validate(clf, CX, CY, cv=c_skf, scoring=scorer, n_jobs=-1)
                    #print("scores XY", scoress)

                    scores["acc"], stds["acc"] = scoress["test_accuracy"].mean(), scoress["test_accuracy"].std() 
                    scores["f1"], stds["f1"] = scoress["test_f1"].mean(), scoress["test_f1"].std() 
                    scores["roc"], stds["roc"] = scoress["test_roc_auc"].mean(), scoress["test_roc_auc"].std() 
                    scores["jac"], stds["jac"] = scoress["test_jaccard"].mean(), scoress["test_jaccard"].std() 
                    scores["pre"], stds["pre"] = scoress["test_precision"].mean(), scoress["test_precision"].std()
                    scores["rec"], stds["rec"] = scoress["test_recall"].mean(), scoress["test_recall"].std() 

                    #cv_c = KFold(n_splits=10, random_state=7, shuffle=True)
                    #scores_cv = cross_val_score(clf, CX, CY, cv=cv_c, scoring='f1', n_jobs=-1)
                    #scores["f1"], stds["f1"] =  scores_cv.mean(), scores_cv.std()

                    evals[name]={ "metrics":scores, "std":stds, "ytrue":ytrue, "ypred":ypred }
                    print("evals[name]", evals[name]["metrics"])

                elif self.dat["type"]=="GRID" or self.dat["type"]=="SIMP" or self.dat["type"]=="FESE":

                    #training
                    clf.fit(trainX, trainY)
                    #save best parameters
                    if self.dat["type"]=="GRID":
                        Util.makedir(self.dat["outputdir"]+"/bestparameters/")
                        Util.write(fiparam, clf.best_params_)
                    
                    #testing
                    pre = clf.predict(testX)

                    evals[name]={"metrics":self.evaluation(testY, pre), "ytrue":testY.tolist(), "ypred":pre.tolist()}
                    del clf

                elif self.dat["type"]=="COMB":
                    mscores, preys = self.combine_class(clf, data["ferkg"], trainX, trainY, testX, testY)
                    evals[name]={"metrics":mscores, "ytrue":testY.tolist(), "ypred":preys}
                    del clf



                if self.dat["ismultiprocessing"]:
                    lod ="{:.2f}%".format( ( self.load.value/(len(self.evals)*len(clsr)) )*100.0 )
                    print("Classification:", lod, end="\r", flush=True)
                    self.load.value += 1
                else:
                    lod ="{:.2f}%".format( ( self.load_wp/(len(self.evals_wp)*len(clsr)) )*100.0 )
                    print("Classification:", lod, end="\r", flush=True)
                    self.load_wp += 1
            
            end_t = time.time()
            print("time", name, data["xval"], end_t-begin_t, trainX.shape, testX.shape)
        if self.dat["ismultiprocessing"]:
            self.evals[data["id"]] = evals
        else:
            self.evals_wp[data["id"]] = evals


    def get_data(self):
        filt = self.dat["filter"].keys()
        datacsv = []
        ferkg = ""
        it = 0
        for f in self.fetures:
            for norm in self.dat["norms"]:
                ftrain = self.dat["inputdir"]+"/"+f["train"]["file"]
                ftest  = self.dat["inputdir"]+"/"+f["test"]["file"]

                infilter, xval, sampling = True, 0, 1.0
                if len(filt)>0:
                    infilter = True if f["train"]["file"] in filt else False
                    if infilter:
                        row = self.dat["filter"][f["train"]["file"]]
                        xval = row["id"]
                        if "ferkg" in row:
                            ferkg = row["ferkg"]
                        sampling = row["sampling"]

                if  infilter and os.path.exists(ftrain) and os.path.exists(ftest):
                    #print("XXXXXXXXXXX",f["train"]["label"],"XXXXXXXXXXXXXX")
                    label = f["train"]["label"].replace("{", "").replace("}", "")
                    #print("parameters label", label)

                    datacsv.append({    
                        "id":it,
                        "name":f["name"],
                        "trainf":f["train"]["file"],
                        "testf":f["test"]["file"],
                        "norm":norm,
                        "parameters":f,
                        "label":label,
                        "xval":xval,
                        "sampling":sampling,
                        "type":self.dat["type"],
                        "ferkg":ferkg
                        })
                    it +=1
        return datacsv

    def execute(self):
        datacsv = self.get_data()
        #print("datacsv", datacsv)
        if self.dat["ismultiprocessing"]:
            pool = Pool(processes=6)
            rs = pool.map(self.process, datacsv)
            pool.close()
            for i in range(len(datacsv)):
                datacsv[i]["evals"] = self.evals[i]
        else:
            for dar in datacsv:
                self.process(dar)
            for i in range(len(datacsv)):
                datacsv[i]["evals"] = self.evals_wp[i]

        datacsv = list(datacsv);


        full = "_SIMP_"
        if self.dat["type"]=="GRID":
            full= "_GRID_"
        elif self.dat["type"]=="CROSS":
            full= "_CROSS_"
        elif self.dat["type"]=="FESE":
            full= "_FESE_"

        full+=self.dat["balancing"]+""

        idx = f'{self.dat["id"]}{full}'
        fileout = f'classinfo_{idx}.json'

        Util.write(self.dat["outputdir"]+"/"+fileout, datacsv)

        Util.makeheatmap(idx, self.dat["outputdir"], self.dat["outputdir"], fileout)
        Util.makebar(idx, self.dat["outputdir"], self.dat["outputdir"], fileout)
        print("Complete: Classification")


    @staticmethod
    def classifiers():        
        """ 
        classifiers = [
            {"name":"EnsembleSVM","model":Classification.get_voting()},
        ]
        """

        """
        classifiers = [
            {"name":"XGBC","model":XGBClassifier(learning_rate=0.1, n_estimators=150, max_depth=4,
                min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27)},

        ]
        """




        #model.best_params_
        param_1 = {
            'C': [1, 10, 100, 1000, 10000],
            'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 'scale', 'auto'], 
        }
        param_2 = {
            'max_depth': [2,3,4,5,6,7,8],
            'n_estimators': [60, 100, 140, 180, 220]
            #'learning_rate': [0.1, 0.01, 0.05]
            #range(60, 221, 40),
        }        

        model1 = svm.SVC(kernel="rbf")
        model2 = XGBClassifier(objective= 'binary:logistic', learning_rate=0.1, nthread=4, seed=27, n_jobs=-1)

        classifiers = {
            ## GridSearchCV
            ##"SVCRBF_GS":{"type":1, "model":model1, "grid":GridSearchCV(model1, param_1, scoring='accuracy', n_jobs=-1)},
            ##"XGBC_GS":{"type":1, "model":model2, "grid":GridSearchCV(model2, param_2, scoring='accuracy')},

            ## Others
            #"RFC":{"model":RandomForestClassifier(n_estimators=100, random_state=7, n_jobs=-1)},
            "RFC":{"model":RandomForestClassifier(n_estimators=50, random_state=7, n_jobs=-1)},
            "KNN":{"model":KNeighborsClassifier(n_neighbors = 3)},
            #{"name":"SVCLINEAR","model":svm.SVC(kernel="linear", C=2)},
            #{"name":"SVCLINEAR","model":svm.SVC(kernel="linear", C=0.025)},
            #{"name":"SVCLINEAR","model":SVC(kernel="linear", C=2, n_jobs=-1)},
            
            #{"name":"SVCLINEAR","model":LinearSVC(C=0.01, penalty="l1", dual=False)},
            
            "SVCRBF":{"model":svm.SVC(kernel="rbf")},
            "DTC":{"model":DecisionTreeClassifier(random_state=7)},
            "ADBC":{"model":AdaBoostClassifier(random_state=7)},
            "GNBC":{"model":GaussianNB()},
            "XGBC":{"model":XGBClassifier()},
            "ETC":{"model":ExtraTreesClassifier(n_estimators=150, n_jobs=-1)},


            "ENSE":{"model":VotingClassifier(estimators=[svm.SVC(kernel="rbf"), XGBClassifier()], voting='soft')},
            "SGDC":{"model":SGDClassifier(loss="hinge", penalty="l2", max_iter=5)}
            #{"name":"MLPC","model":MLPClassifier(activation='tanh',solver='adam',alpha=1e-5,learning_rate='constant',random_state=7)},
        }


        #"XGBC":{"type":0, "model":XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=4,
        #        min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
        #        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27)},
        




        """ 
            {"name":"XGBC","model":GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=4,
            min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
            objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
            param_grid = param_test2b, scoring='roc_auc',n_jobs=4, cv=5)}
        """

        """
        {"name":"KNN","model":KNeighborsClassifier(n_neighbors = 3)},
        {"name":"SVCGRID","model":GridSearchCV(skn.svm.SVC(kernel='rbf'), paramGrid, scoring='accuracy', n_jobs=1)},
        {"name":"SVCLINEAR","model":skn.svm.SVC(kernel="linear", C=0.025)},
        #{"name":"SVC","model":skn.svm.SVC(gamma=2, C=0.025)},
        {"name":"DTC","model":DecisionTreeClassifier(random_state=7)},
        {"name":"RFC","model":RandomForestClassifier(n_estimators=10, random_state=7, n_jobs=1)},
        #{"name":"MLPC","model":GridSearchCV(MLPClassifier(random_state=7), paramMLPC, n_jobs=-1)},
        #{"name":"MLPC","model":MLPClassifier(activation='tanh',solver='adam',alpha=1e-5,learning_rate='constant',random_state=7)},            
        {"name":"ADBC","model":AdaBoostClassifier(random_state=7)},
        {"name":"GNBC","model":GaussianNB()},
        """

        return classifiers

    @staticmethod
    def get_voting():
        models = list()
        models.append(('svm1', svm.SVC(probability=True, kernel='poly', degree=1)))
        models.append(('svm2', svm.SVC(probability=True, kernel='poly', degree=2)))
        models.append(('svm3', svm.SVC(probability=True, kernel='poly', degree=3)))
        models.append(('svm4', svm.SVC(probability=True, kernel='poly', degree=4)))
        models.append(('svm5', svm.SVC(probability=True, kernel='poly', degree=5)))
        
        #ensemble = VotingClassifier(estimators=models, voting='soft', n_jobs=-1)
        ensemble = VotingClassifier(estimators=models, voting='soft')
        return ensemble

    @staticmethod
    def evaluation(y_true, y_pred):
        y_true, y_pred = y_true.tolist(), y_pred.tolist()
        acc = metrics.accuracy_score(y_true, y_pred, normalize=True)
        f1 = metrics.f1_score(y_true, y_pred)
        roc = metrics.roc_auc_score(y_true, y_pred)
        jac = metrics.jaccard_score(y_true, y_pred)       
        pre = metrics.precision_score(y_true, y_pred)
        rec = metrics.recall_score(y_true, y_pred)
        
 
        return {"acc":acc, "f1":f1, "roc":roc, "jac":jac, "pre":pre, "rec":rec}


    @staticmethod
    def getDataSets(inputDir, trainFileName, testFileName):
        trainDataSet = pd.read_csv(inputDir+"/"+trainFileName)
        testDataSet = pd.read_csv(inputDir+"/"+testFileName)
        
        #trainDataSet.columns = trainDataSet.columns.str.strip()
        #testDataSet.columns = testDataSet.columns.str.strip()


        columns = trainDataSet.columns.tolist()
        #columns.pop() 
        columns.remove("image")
        columns.remove("loc1")
        columns.remove("loc2")
        columns.remove("loc3")
        columns.remove("loc4")
        columns.remove("idseg")
        columns.remove("target")
        
        #if "original_firstorder_10Percentile" in columns:
        #    columns.remove("original_firstorder_10Percentile")
        
        #if "original_firstorder_90Percentile" in columns:
        #    columns.remove("original_firstorder_90Percentile")




        #rcols = ["original_firstorder_10Percentile","original_firstorder_90Percentile","original_firstorder_Energy","original_firstorder_Entropy","original_firstorder_InterquartileRange","original_firstorder_Kurtosis","original_firstorder_Maximum","original_firstorder_MeanAbsoluteDeviation","original_firstorder_Mean","original_firstorder_Median","original_firstorder_Minimum","original_firstorder_Range","original_firstorder_RobustMeanAbsoluteDeviation","original_firstorder_RootMeanSquared","original_firstorder_Skewness","original_firstorder_TotalEnergy","original_firstorder_Uniformity","original_firstorder_Variance"]
        #for c in rcols:
        #    columns.remove(c);

        trainX = trainDataSet[columns]
        trainY = trainDataSet.target.astype('category').cat.codes
        
        datcat = dict(enumerate(trainDataSet.target.astype('category').cat.categories))
        Util.write(inputDir+"/category_name.json", datcat)

        #columns = testDataSet.columns.tolist()
        #columns.pop() 
        testX = testDataSet[columns]
        testY = testDataSet.target.astype('category').cat.codes

        """ 
        print("############")
        print("Train: Pleura ", sum(trainY == 0))
        print("Train: Non pleura ", sum(trainY == 1))
        print("Test: Pleura ", sum(testY == 0))
        print("Test: Non pleura ", sum(testY == 1))
        print("############")
        """
        #print(trainX, trainX)
        return trainX, trainY, testX, testY
    
    @staticmethod
    def scaleData(trainX, testX, norm):
        if norm != "None":
            if norm == "std":
                trainX = pd.DataFrame(StandardScaler().fit_transform(trainX))
                testX = pd.DataFrame(StandardScaler().fit_transform(testX))
            elif norm == "minmax":
                trainX = pd.DataFrame(MinMaxScaler().fit_transform(trainX))
                testX = pd.DataFrame(MinMaxScaler().fit_transform(testX))
        return trainX, testX

    @staticmethod
    def getScale(norm):
        sc = None
        if norm == "std":
            sc = StandardScaler()
        elif norm == "minmax":
            sc = MinMaxScaler()

        return sc

    def process_feature_selection(self, estimator, features, trainX, trainY, file, norm):
        model = None
        selection = None
        subsets = []
        ####SequentialFeatureSelector from sklearn

        if self.dat["featureselection"]["name"]=="fs_importance":

            # Train model
            model = estimator.fit(trainX, trainY)
            selection = estimator.feature_importances_.argsort().tolist()

        elif self.dat["featureselection"]["name"]=="fs_extratrees":

            #estimator = ExtraTreesClassifier(n_estimators=150, n_jobs=-1)
            model = SelectFromModel(
                        estimator,
                        prefit=False).fit(trainX, trainY)
            selection = model.get_support(indices=True).tolist()
        


        elif self.dat["featureselection"]["name"]=="fs_svc":
            #estimator = LinearSVC(C=0.01, penalty="l2", dual=False)
            
            model = SelectFromModel(
                        estimator,
                        prefit=False).fit(trainX, trainY)
            selection = model.get_support(indices=True).tolist()
        ## only for positive values
        #elif self.feature_selection=="fs_chi2":
        #    model = SelectKBest(score_func=chi2, k=trainX.shape[1]-5)
        #   trainX = model.fit_transform(trainX)
        #    testX = model.transform(testX)

        elif self.dat["featureselection"]["name"]=="fs_geuni":
            model = GenericUnivariateSelect(
                        score_func=mutual_info_classif,
                        mode='percentile',
                        param=70).fit(trainX, trainY)
            selection = model.get_support(indices=True).tolist()

        elif self.dat["featureselection"]["name"]=="fs_rfecv":
            #clf = DecisionTreeClassifier()
            #clf = LogisticRegression(C=1, penalty='l2')
            #estimator = SVR(kernel="linear")
            #model = RFECV(clf, trainX.shape[1]-15)
            
            #estimator = LinearSVC(C=0.01, penalty="l2", dual=False)
            model = RFECV(
                        estimator,
                        min_features_to_select=int(len(features)/3.),
                        n_jobs=-1).fit(trainX, trainY)
            selection = model.get_support(indices=True).tolist()

        elif self.dat["featureselection"]["name"]=="fs_lasso":
            #estimator = LassoCV(cv=5, normalize = True, n_jobs=1)
            model = SelectFromModel(
                        estimator,
                        threshold=0.25,
                        norm_order=1,
                        max_features=None,
                        prefit=False).fit(trainX, trainY)
            selection = model.get_support(indices=True).tolist()

        elif self.dat["featureselection"]["name"]=="fs_genetic":
            #estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")
            
            #estimator = ExtraTreesClassifier(n_estimators=150)
            
            #estimator = KNeighborsClassifier(n_neighbors=2, n_jobs=-1)
            model = GeneticSelectionCV(
                        estimator,
                        cv=5,
                        verbose=1,
                        scoring="f1",
                        max_features=int((len(features))-(len(features)/3)),
                        n_population=70,
                        crossover_proba=0.5,
                        mutation_proba=0.2,
                        n_generations=40,
                        crossover_independent_proba=0.5,
                        mutation_independent_proba=0.05,
                        tournament_size=3,
                        n_gen_no_change=10,
                        caching=True,
                        n_jobs=-1).fit(trainX, trainY)
            selection = model.get_support(indices=True).tolist()

        elif self.dat["featureselection"]["name"]=="fs_sequential_forward":
            #estimator = KNeighborsClassifier(n_neighbors=2)
            #estimator = LogisticRegression()
            
            #estimator = RandomForestClassifier(n_estimators=50, random_state=7)
            model = SequentialFeatureSelector(
                        estimator,
                        direction="forward",
                        n_features_to_select=self.dat["featureselection"]["n_features"],
                        n_jobs=-1).fit(trainX, trainY)
            selection = model.get_support(indices=True).tolist()

        elif self.dat["featureselection"]["name"]=="fs_sequential_backward":
            #estimator = KNeighborsClassifier(n_neighbors=2)
            
            #cls = LogisticRegression()
            #cls = RandomForestClassifier(n_estimators=100, random_state=7, n_jobs=1)
            model = SequentialFeatureSelector(
                        estimator,
                        direction="backward",
                        n_features_to_select=None,
                        n_jobs=-1).fit(trainX, trainY)
            selection = model.get_support(indices=True).tolist()






        elif self.dat["featureselection"]["name"]=="fs_mlxtend_sequential_forward":
            #estimator = KNeighborsClassifier(n_neighbors=2)

            
            #cls = LogisticRegression()
            #estimator = RandomForestClassifier(n_estimators=100, random_state=7, n_jobs=1)
            #estimator = svm.SVC(kernel="rbf")
            model = SFS(
                        estimator,
                        k_features=int(len(features)/2.), 
                        forward=True, 
                        floating=False, 
                        verbose=2,
                        scoring='f1',
                        cv=3,
                        n_jobs=-1).fit(trainX, trainY)
            selection = list(model.k_feature_idx_)
            #subsets = model.subsets_()

        elif self.dat["featureselection"]["name"]=="fs_mlxtend_sequential_backward":
            #estimator = KNeighborsClassifier(n_neighbors=2)
            
            
            #cls = LogisticRegression()
            #cls = RandomForestClassifier(n_estimators=100, random_state=7, n_jobs=1)
            model = SFS(
                        estimator,
                        k_features=50, 
                        #k_features=int(len(features)/2.), 
                        forward=False, 
                        floating=False, 
                        scoring='accuracy',
                        cv=4,
                        n_jobs=-1).fit(trainX, trainY)
            selection = list(model.k_feature_idx_)
            #subsets = model.subsets_()




        elif self.dat["featureselection"]["name"]=="fs_mlxtend_sequential_forward_floating":
            #estimator = KNeighborsClassifier(n_neighbors=2)
            #estimator = LogisticRegression()
            
            
            #estimator = RandomForestClassifier(n_estimators=50, random_state=7)
            
            
            #estimator = XGBClassifier(
            #                learning_rate=0.2, n_estimators=50, max_depth=4,
            #                min_child_weight=2, gamma=0.0, subsample=0.8, colsample_bytree=0.8,
            #                objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27,
            #                ##tree_method='gpu_hist'  # THE MAGICAL PARAMETER
            #            )
                    
            #estimator = svm.SVC(kernel="rbf")


            model = SFS(
                        estimator,
                        k_features=self.dat["featureselection"]["n_features"],
                        #k_features=int(len(features)/2.), 
                        forward=True, 
                        floating=True, 
                        verbose=2,
                        scoring='f1',
                        cv=3,
                        n_jobs=-1
                        ).fit(trainX, trainY)
            selection = list(model.k_feature_idx_)
            #subsets = model.subsets_

        elif self.dat["featureselection"]["name"]=="fs_mlxtend_sequential_backward_floating":
            #estimator = KNeighborsClassifier(n_neighbors=2)
            #estimator = LogisticRegression()
            
            
            #estimator = RandomForestClassifier(n_estimators=50, random_state=7)
            model = SFS(
                        estimator,
                        k_features=self.dat["featureselection"]["n_features"], 
                        forward=False, 
                        floating=True, 
                        verbose=2,
                        scoring='f1',
                        cv=4,
                        n_jobs=-1).fit(trainX, trainY)
            selection = list(model.k_feature_idx_)
            #subsets = model.subsets_


        #selection = model.get_support(indices=False).tolist()
        datafesel = {"features":features, "selected":selection, "subset":subsets}
        normshift = {"None":0, "std":1,"minmax":2}
        Util.makedir(self.dat["outputdir"]+"/featureselection/")
        Util.write(self.dat["outputdir"]+"/featureselection/"+self.dat["featureselection"]["name"]+"_"+self.dat["featureselection"]["estimator"]+"_"+str(normshift[norm])+"_"+file, datafesel)
        return datafesel



    def combine_class(self, model, ferkg, trainX, trainY, testX, testY): 
        objferkg = Util.read(self.dat["inputdir"]+"/featureselection/"+ferkg)

        Xtrain = self.get_features_selected(trainX, objferkg)
        Xtest = self.get_features_selected(testX, objferkg)

        scores, ypres = [], []
        #print("trainX.shape", trainX.shape)
        for i in range(1,(trainX.shape[1])):
            clf = clone(model)
            Xtra = Xtrain.iloc[:,0:i]
            print("X.shape", Xtra.shape)
            clf.fit(Xtra, trainY)

            Xtes = Xtest.iloc[:,0:i]
            ypre = clf.predict(Xtes)
            metrics = self.evaluation(testY, ypre)
            scores.append(metrics)
            print(metrics)
            del clf
        return  scores, ypres

if __name__ == "__main__":    
    with open(sys.argv[1], mode='r') as jsond:
        #print (jsdata)
        args = ujson.load(jsond)
        #print(args)

        for arg in args:
            if "fromUtil" in arg:
                arg = Util.makeConfigureFormUtil(arg)
                print(arg)                
                for aa in arg:
                    obje = Classification(aa)
                    obje.execute()
            else:
                obje = Classification(aa)
                obje.execute()



    """
    ####### DS1: LBP
    inputdir = "../../../../data/LHA/dataset_1/csv/exp/20210626035559"
    outputdir = inputdir
    #mfeature_selection(inputdir, outputdir)
    mclassification(inputdir, outputdir)
    
    
    ####### DS1: RAD
    inputdir = "../../../../data/LHA/dataset_1/csv/exp/20210626045026"
    outputdir = inputdir
    mfeature_selection(inputdir, outputdir)
    mclassification(inputdir, outputdir)
    
    ####### DS1: LBP+RAD
    inputdir = "../../../../data/LHA/dataset_1/csv/exp/20210626035559+20210626045026"
    outputdir = inputdir
    mfeature_selection(inputdir, outputdir)
    mclassification(inputdir, outputdir)
    """






    """
    ####### DS2: LBP
    inputdir = "../../../../data/LHA/dataset_2/csv/exp/20210624080011"
    outputdir = inputdir
    #mfeature_selection(inputdir, outputdir)
    mclassification(inputdir, outputdir)

    ####### DS2: RAD
    inputdir = "../../../../data/LHA/dataset_2/csv/exp/20210625215732"
    outputdir = inputdir
    #mfeature_selection(inputdir, outputdir)
    mclassification(inputdir, outputdir)
    

    ####### DS2: LBP+RAD
    inputdir = "../../../../data/LHA/dataset_2/csv/exp/20210624080011+20210625215732"
    outputdir = inputdir
    #mfeature_selection(inputdir, outputdir)
    mclassification(inputdir, outputdir)    
    
    
    ####### DS2+DS1: LBP+RAD
    inputdir = "../../../../data/LHA/dataset_2/csv/exp/20210624080011+20210625215732+20210626035559+20210626045026"
    outputdir = inputdir
    #mfeature_selection(inputdir, outputdir)
    mclassification(inputdir, outputdir)
    """


    """
    ####### DS2: LBP+RAD 30 test
    inputdir = "../../../../data/LHA/dataset_2/csv/exp/ds2wholetes30"
    outputdir = inputdir
    #mfeature_selection(inputdir, outputdir)
    mclassification(inputdir, outputdir)
    """

    #mcombine("dataset_1","20210626035559", "20210626045026")
    #mjoindatasets("dataset_2", "dataset_1", "20210624080011+20210625215732", "20210626035559+20210626045026")
    #mjoindatasetswhole("dataset_2", "dataset_2", "20210624080011+20210625215732", "ds2wholetes30")




#cmap='gray', vmin=0, vmax=255