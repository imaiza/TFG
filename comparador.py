# -*- coding: utf-8 -*-
"""
Compara las prediciones realizadas para las resoluciones de 2048 y 4096.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import pickle

from graficador2 import all_graph, all_graph_color
from reg_lin import cov

#from bineador import bineador
#from bineadortemp import bineadortemp
from scipy import stats
#from outliers import outliers
path_model=os.path.join( os.getcwd() , 'saved_models')
path_datafiles=os.path.join( os.getcwd(), 'UNITSIMS')

def file_finder(path, search_name):
    result=[]
    for file in os.listdir(path):
        if file.startswith(search_name):
            result.append(file)
    return result


new_colum=['num_prog(4)','rs(12)','scale_of_last_MM(15)','Spin(26)',\
           'M500c(41)','b_to_a(500c)(51)','T/|U|(56)','Vpeak(62)','a']

#Carga los algoritmos con los que realizamos las predicciones
#NN data
ML_NN_list= file_finder(path_model, 'NNNew_Matched_Rockstar-G3X-snap_merge_data_reducido')
normalicer_list_NN= file_finder(path_model, 'normalicerNNNew_Matched_Rockstar-G3X-snap_merge_data_reducido')
stat_list_NN= file_finder(path_model, 'statdataNNNew_Matched_Rockstar-G3X-snap_merge_data_reducido')
#dfp_list_NN= file_finder(path_datafiles, 'NN_New_Matched-Rockstar-G3X-snap_')    

#RF data
ML_RF_list=file_finder(path_model, 'RFNew_Matched_Rockstar-G3X-snap_merge_data_ReducedDim')
normalicer_list_RF=file_finder(path_model, 'normalicerRFNew_Matched_Rockstar-G3X-snap_merge_data_ReducedDim')
stat_list_RF=file_finder(path_model, 'statdataRFNew_Matched_Rockstar-G3X-snap_merge_data_ReducedDim')
#dfp_list_RF=file_finder(path_datafiles, 'RF_New_Matched')    


#%%
    
### PARA EL CASO DE 4096

hlist_list=file_finder(path_datafiles, 'hlist_reducido')
print('Predicting 4096')
print('Predicting hlist with NN')

#NN Loop
hlist_all_NN=[]
for i in range(len(hlist_list)):
    
    print('predicting ' + hlist_list[i])
    
    dfp_hlist=pd.read_csv(os.path.join(path_datafiles, hlist_list[i]), sep =' ',header=None)
    dfp_hlist.columns=new_colum
    dfp_hlist=np.log10(dfp_hlist)
    
    #All alg
    print('Doing with all alg')
    ML_NN_all= pickle.load(open(os.path.join(path_model, 'NNNew_Matched_Rockstar-G3X-snap_merge_data_reducido_multitargetV7.pickle'), 'rb'))
    normalicer_all= pickle.load(open(os.path.join(path_model, 'normalicerNNNew_Matched_Rockstar-G3X-snap_merge_data_reducido_multitargetV7.pickle'), 'rb'))
    ymax_all, ymin_all= pickle.load(open(os.path.join(path_model, 'statdataNNNew_Matched_Rockstar-G3X-snap_merge_data_reducido_multitargetV7.pickle'), 'rb'))
    
    X_norm_all= normalicer_all.transform(dfp_hlist)
    Y_pred_all= ML_NN_all.predict(X_norm_all)
    
    Y_pred_all= Y_pred_all * (ymax_all.values - ymin_all.values) + ymin_all.values
    
    targetnames=['G3XMgas_NN(80)','G3XMstar_NN(81)','G3XTgas_mw_NN(82)','G3XYx_NN(84)','G3XYsz_NN(85)']
    Y_pred_all= pd.DataFrame(Y_pred_all, columns=targetnames)
    
    dfp_all=dfp_hlist.copy()
    dfp_all[targetnames]=Y_pred_all
    
    hlist_all_NN.append(dfp_all)
    
    #plot hlist
    z=hlist_list[i][10:-6]
    all_graph(dfp_filtered=dfp_all, z=z, setname='NNhlist4096', alg_name='NN')
    
    print('Done')
    print('----------------')

dfp1=pd.concat(hlist_all_NN)
dfp_1=dfp1.sample(frac=0.1).reset_index(drop=True)
dfp_1.to_csv(os.path.join(path_datafiles, 'NN_Hlist4096_all.csv')) #Un super csv con todos los hlist

z='All data'
all_graph_color(dfp_filtered=dfp_1, z=z, setname='NNhlist_4096', alg_name='NN')

print('STARTING WITH RANDOM FOREST')
#RF Loop
hlist_all_RF=[]
for i in range(len(hlist_list)):
    print('predicting ' + hlist_list[i])
    dfp_hlist=pd.read_csv(os.path.join(path_datafiles, hlist_list[i]), sep =' ',header=None)
    dfp_hlist.columns=new_colum
    dfp_hlist=np.log10(dfp_hlist)
    
    
    #All alg
    print('Doing with all alg')
    ML_RF_all= pickle.load(open(os.path.join(path_model, 'RFNew_Matched_Rockstar-G3X-snap_merge_data_ReducedDimMultitargetV7.pickle'), 'rb'))
    normalicer_all= pickle.load(open(os.path.join(path_model, 'normalicerRFNew_Matched_Rockstar-G3X-snap_merge_data_ReducedDimMultitargetV7.pickle'), 'rb'))
    ymax_all, ymin_all= pickle.load(open(os.path.join(path_model, 'statdataRFNew_Matched_Rockstar-G3X-snap_merge_data_ReducedDimMultitargetV7.pickle'), 'rb'))
    
    X_norm_all= normalicer_all.transform(dfp_hlist)
    Y_pred_all= ML_RF_all.predict(X_norm_all)
    
    targetnames=['G3XMgas_NN(80)','G3XMstar_NN(81)','G3XTgas_mw_NN(82)','G3XYx_NN(84)','G3XYsz_NN(85)']
    Y_pred_all= Y_pred_all * (ymax_all.values - ymin_all.values) + ymin_all.values
    
    Y_pred_all= pd.DataFrame(Y_pred_all, columns=targetnames)
    
    dfp_all=dfp_hlist.copy()
    dfp_all[targetnames]=Y_pred_all
    
    hlist_all_RF.append(dfp_all)
    
    #plot hlist
    z=hlist_list[i][10:-6]
    all_graph(dfp_filtered=dfp_all, z=z, setname='RFhlist4096', alg_name='RF')
    
    print('Done')
    print('----------------')

dfp=pd.concat(hlist_all_RF)
dfp_10=dfp.sample(frac=0.1).reset_index(drop=True)
dfp_10.to_csv(os.path.join(path_datafiles, 'RF_Hlist4096_all.csv'))

z='All data'
all_graph_color(dfp_filtered=dfp_1, z=z, setname='RFhlist_4096', alg_name='RF')
# REPETIMOS EL PROCESO PARA LA OTRA RESOLUCION
### PARA EL CASO DE 2048

hlist_list=file_finder(path_datafiles, 'hlist_reducido2')
print('Predicting 2048')
print('Predicting hlist with NN')

#NN Loop
hlist_all_NN=[]
for i in range(len(hlist_list)):
    
    print('predicting ' + hlist_list[i])
    
    dfp_hlist=pd.read_csv(os.path.join(path_datafiles, hlist_list[i]), sep =' ',header=None)
    dfp_hlist.columns=new_colum
    dfp_hlist=np.log10(dfp_hlist)
    
    #All alg
    print('Doing with all alg')
    ML_NN_all= pickle.load(open(os.path.join(path_model, 'NNNew_Matched_Rockstar-G3X-snap_merge_data_reducido_multitargetV7.pickle'), 'rb'))
    normalicer_all= pickle.load(open(os.path.join(path_model, 'normalicerNNNew_Matched_Rockstar-G3X-snap_merge_data_reducido_multitargetV7.pickle'), 'rb'))
    ymax_all, ymin_all= pickle.load(open(os.path.join(path_model, 'statdataNNNew_Matched_Rockstar-G3X-snap_merge_data_reducido_multitargetV7.pickle'), 'rb'))
    
    X_norm_all= normalicer_all.transform(dfp_hlist)
    Y_pred_all= ML_NN_all.predict(X_norm_all)
    
    Y_pred_all= Y_pred_all * (ymax_all.values - ymin_all.values) + ymin_all.values
    
    targetnames=['G3XMgas_NN(80)','G3XMstar_NN(81)','G3XTgas_mw_NN(82)','G3XYx_NN(84)','G3XYsz_NN(85)']
    Y_pred_all= pd.DataFrame(Y_pred_all, columns=targetnames)
    
    dfp_all=dfp_hlist.copy()
    dfp_all[targetnames]=Y_pred_all
    
    hlist_all_NN.append(dfp_all)
    
    #plot hlist
    z=hlist_list[i][10:-6]
    all_graph(dfp_filtered=dfp_all, z=z, setname='NNhlist2048', alg_name='NN')
    
    print('Done')
    print('----------------')

dfp1=pd.concat(hlist_all_NN)
dfp_1=dfp1.sample(frac=0.1).reset_index(drop=True)
dfp_1.to_csv(os.path.join(path_datafiles, 'NN_Hlist2048_all.csv')) #Un super csv con todos los hlist

z='All data'
all_graph_color(dfp_filtered=dfp_1, z=z, setname='NNhlist_2048', alg_name='NN')

print('STARTING WITH RANDOM FOREST')
#RF Loop
hlist_all_RF=[]
for i in range(len(hlist_list)):
    print('predicting ' + hlist_list[i])
    dfp_hlist=pd.read_csv(os.path.join(path_datafiles, hlist_list[i]), sep =' ',header=None)
    dfp_hlist.columns=new_colum
    dfp_hlist=np.log10(dfp_hlist)
    
    
    #All alg
    print('Doing with all alg')
    ML_RF_all= pickle.load(open(os.path.join(path_model, 'RFNew_Matched_Rockstar-G3X-snap_merge_data_ReducedDimMultitargetV7.pickle'), 'rb'))
    normalicer_all= pickle.load(open(os.path.join(path_model, 'normalicerRFNew_Matched_Rockstar-G3X-snap_merge_data_ReducedDimMultitargetV7.pickle'), 'rb'))
    ymax_all, ymin_all= pickle.load(open(os.path.join(path_model, 'statdataRFNew_Matched_Rockstar-G3X-snap_merge_data_ReducedDimMultitargetV7.pickle'), 'rb'))
    
    X_norm_all= normalicer_all.transform(dfp_hlist)
    Y_pred_all= ML_RF_all.predict(X_norm_all)
    
    targetnames=['G3XMgas_NN(80)','G3XMstar_NN(81)','G3XTgas_mw_NN(82)','G3XYx_NN(84)','G3XYsz_NN(85)']
    Y_pred_all= Y_pred_all * (ymax_all.values - ymin_all.values) + ymin_all.values
    
    Y_pred_all= pd.DataFrame(Y_pred_all, columns=targetnames)
    
    dfp_all=dfp_hlist.copy()
    dfp_all[targetnames]=Y_pred_all
    
    hlist_all_RF.append(dfp_all)
    
    #plot hlist
    z=hlist_list[i][10:-6]
    all_graph(dfp_filtered=dfp_all, z=z, setname='RFhlist2048', alg_name='RF')
    print('Done')
    print('----------------')

dfp=pd.concat(hlist_all_RF)
dfp_10=dfp.sample(frac=0.1).reset_index(drop=True)
dfp_10.to_csv(os.path.join(path_datafiles, 'RF_Hlist2048_all.csv'))

z='All data'
all_graph_color(dfp_filtered=dfp_1, z=z, setname='RFhlist_2048', alg_name='RF')






