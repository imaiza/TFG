import numpy as np
import matplotlib                                                                                                                   
matplotlib.use('Agg')   
import matplotlib.pyplot as plt
import pandas as pd

import os

from MSRF7_monotarget3_fun import RF_train
from MSnn7_monotarget_fun import NN_train
from MSXB7_fun2 import XGBoost_train

print(__doc__)
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from scipy.cluster import hierarchy

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

import sklearn.preprocessing as skpp

def find_all(name, path):
    result = []
    for file in os.listdir(path):
        if file.startswith(name):
            result.append(file)
    return result

name='New_Matched-Rockstar-G3X-snap_' #name to search by
#name='New_Matched-Rockstar-G3X-snap_all'
#path=os.getcwd() #get current file dir
path= "/home2/frobledo/datafiles/V7/" #get file dir

new_columns=['num_prog(4)', 'sam_mvir(9)', 'mvir(10)', 'rvir(11)', 'rs(12)',\
       'vrms(13)', 'scale_of_last_MM(15)', 'vmax(16)', 'Spin(26)',\
       'Rs_Klypin(37)', 'Mmvir_all(38)', 'M200b(39)', 'M200c(40)', 'M500c(41)',\
       'M2500c(42)', 'Xoff(43)', 'Voff(44)', 'Spin_Bullock(45)', 'b_to_a(46)',\
       'c_to_a(47)', 'b_to_a(500c)(51)', 'c_to_a(500c)(52)', 'T/|U|(56)',\
       'Macc(59)', 'Mpeak(60)', 'Vacc(61)', 'Vpeak(62)', 'Halfmass_Scale(63)',\
       'First_Acc_Mvir(72)', 'First_Acc_Vmax(73)', 'Vmax\@Mpeak(74)','a',\
       'G3XMgas(80)','G3XMstar(81)' ,'G3XTgas_mw(82)', 'G3XYx(84)', 'G3XYsz(85)']


targetname=['G3XMgas(80)','G3XMstar(81)' ,'G3XTgas_mw(82)', 'G3XYx(84)', 'G3XYsz(85)']
monotarget=['G3XMgas(80)']

train_file_names=sorted(find_all(name,path))
setnames=[]
dfp_list=[]
for i in range(len(train_file_names)):
    setnames.append(str(train_file_names[i][:-8]))
    dfp_temp=pd.read_csv(path + train_file_names[i] , sep=',')
    
    dfp_temp= dfp_temp[new_columns]
    
    dfp_list.append( dfp_temp )
    

dfp_final=pd.concat(dfp_list)
dfp_final2=dfp_final.drop(dfp_final.columns[[33,34,35,36]],axis=1)
dfp_dimred=dfp_final2.drop(dfp_final2.columns[[1,2,3,5,7,9,10,11,12,14,15,16,17,18,19,21,23,24,25,27,28,29,30]],axis=1)    

#Lo ponemos en orden de mayor importancia  a menor

new_kolumns=['M500c(41)','Vpeak(62)','num_prog(4)','scale_of_last_MM(15)',\
             'rs(12)','T/|U|(56)','a','Spin(26)','b_to_a(500c)(51)','G3XMgas(80)']

dfp_dimred=dfp_dimred.reindex(columns=new_kolumns)    

RF_train(dfp_dimred,monotarget, setname='New_Matched_Rockstar-G3X-snap_merge_data')
print('RF all data done')

