"""


@author: Iñigo Maiza
"""
def XGBoost_train(filetrain, targetname, setname):
    #%% Import libraries
    
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    import time
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    from scipy import stats
    
    from eli5.sklearn import PermutationImportance #get feature importance per K-fold


    import pickle
#    from sklearn.externals import joblib
    #ML functions
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
    import sklearn.preprocessing as skpp
    import xgboost as xgb
    
    import re
    regex = re.compile(r"\[|\]|<", re.IGNORECASE) #fix naming in columnames
    
    
    #%% Import data



    dfp=filetrain
    
    print(setname + ' column list:')
    print(dfp.columns)
    print('-----------------------------------')

    print('Data imported')
    

    dfp=dfp[dfp['M500c(41)'] >13.5]

    dfp=dfp.dropna(axis=1)
    
    
    
#    targetname=['G3XMgas(80)','G3XMstar(81)' ,'G3XTgas_mw(82)', 'G3XYx(84)', 'G3XYsz(85)']
    
    dfp.reset_index(drop=True, inplace=True)

    Y_train=dfp[targetname]
    
    dfptrain=dfp.copy()
    
    
    dfptrain.drop(labels=targetname, inplace=True, axis=1)
    
    coldrop=[col for col in dfptrain.columns if 'G3X' in col]
    
    dfptrain.drop(labels=coldrop, inplace=True, axis=1)
    
    new_col_list=dfptrain.columns
    
    print('New column list:')
    print(new_col_list)
    
    
    print('------------------')
    print('OG dfp columns')
    print(dfp.columns)
    
    
    #plot hist
#    plt.figure('Mass hist')
#    plt.hist(dfp['M500c(41)'], bins=20, zorder=1, label=[setname + ' set'])
#    plt.legend()
    #%% Preprocessing of data
    
    
#    Y_train=np.log10(Y_train)
    
    if targetname== 'G3XYsz(85)':
        Ysz_error=Y_train.index[Y_train == -np.inf]
        print('This are the index of log10(Ysz)=-inf')
        print(Ysz_error)
        
        Y_train.drop(labels=Ysz_error, axis=0, inplace=True)
        dfptrain.drop(labels=Ysz_error, axis=0, inplace=True)    
    
    #statistical data from Y_train
#    mu=np.mean(Y_train) #median
#    sigma=np.std(Y_train) #standard deviation

    #%% Weight of data
#    nbins=15
#    mhist, mhist_edge= np.histogram(np.log10(dfptrain['M500c(41)']), bins=nbins)
#    dfptrain['Class']=1 #initialise column
#    
#    rho=0.0
#    
#    for i in range(nbins):
#        dfptrain.loc[(np.log10(dfptrain['M500c(41)']) >= mhist_edge[i]) & (np.log10(dfptrain['M500c(41)']) <= mhist_edge[i+1]), \
#                         'Class'] = rho * (len(dfptrain['M500c(41)']) - mhist[i] * nbins)/(len(dfptrain['M500c(41)']) * mhist[i] * nbins) + (len(dfptrain['M500c(41)']))**-1
#    
#    sampleweight= dfptrain['Class']
#    dfptrain.drop(labels='Class', inplace=True, axis=1)
#    columndrop=[col for col in dfptrain.columns if ('MUSIC' in col) or ('G3X' in col)]
#
#    dfptrain.drop(labels=columndrop, inplace=True, axis=1)
    
    
    
    
    #%% Analysis train data correlation
    
    #we add back target data for correlation analysis
    #dfptrain= dfptrain.copy()
    dfptrain[targetname]=Y_train
    
    corr= dfptrain.corr()
    
    plt.figure('Correlation matrix - training data', figsize=(9,9))
    nticks=len(dfptrain.columns)
    plt.xticks(range(nticks), [lab[0:7] for lab in dfptrain.columns], rotation='vertical')
    plt.yticks(range(nticks), dfptrain.columns)
    _ = plt.colorbar(plt.imshow(corr, interpolation='nearest', vmin=-1., vmax=1., cmap=plt.get_cmap('YlOrBr')))
    plt.title('Correlation matrix - Training data', fontsize=20)
    #plt.savefig('plots/correlation/correlation_plot_XGBoost_'+targetname+'.png')    
    #%% XB on training data - Creation of XB algorithm
    
    #We get the test/train index
    #indexFolds = KFold(dfptrain.values.shape[0], n_folds=5, shuffle=True, random_state=11)
    indexFolds = KFold(n_splits=5, shuffle=True, random_state=11)
    lVarsTarg=dfptrain.columns
    
    R2_XB = []
    MAE_XB = []
    MSE_XB = []
    tuned_parameters = [{'max_depth': [2,4,6],
                        'n_estimators': [51,101,301,601]}]
    
    
    
    # Recorremos las particiones
    ind = 0
    
    Ypred = np.zeros(len(dfptrain[targetname]),)
    Ytarg = np.zeros(len(dfptrain[targetname]),)

    Feature_mean=np.zeros([new_col_list.shape[0] ,])
    print('Feature mean shape ' + str(Feature_mean.shape))
    if len(targetname)==1:
        Ypred= np.ravel(Ypred)
        Ytarg= np.ravel(Ytarg)    
    
    for idxTr, idxTs in indexFolds.split(dfptrain):
    
        ind = ind+1
        print()
        print()
        print('K-fold:',ind)
        
        #Making Min-Max Scaler
        Scaler=skpp.MinMaxScaler()
        X= dfptrain.drop(labels=targetname, axis=1)
        print(X.columns)
        print(X.columns.shape)
        Scaler.fit(X) #Fit scaler to data, then transform
        
        y_min=dfptrain[targetname].min(axis=0)
        y_max=dfptrain[targetname].max(axis=0) #stat data for inv transform
        
        print('y_min:')
        print(y_min)
        print('y_max:')
        print(y_max)
        
        '''
            dfp_scaled= (dfp - dfp.min(axis=0)) / (dfp.max(axis=0) - dfp.min(axis=0))
            dfp_inv= dfp_scaled * (dfp.max(axis=0) - dfp.min(axis=0)) + dfp.min(axis=0)
        
            y_min=dfp.min(axis=0)
            y_max=dfp.max(axis=0)
        '''
        dfptrain_old=dfptrain.copy() #backup
#        dfptrain_scaled=(dfptrain - dfptrain.min(axis=0)) / (dfptrain.max(axis=0) - dfptrain.min(axis=0))
        dfptrain_X=Scaler.transform(X)
        dfptrain_X= pd.DataFrame(dfptrain_X, columns=X.columns)
        
        Y=dfptrain[targetname].squeeze() #change from DataFrame to Series
        print('Y shape : ' + str(Y.shape))
        dfptrain_Y=(Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))
        print('dfptrain_Y shape: ' + str(dfptrain_Y.shape))
#        dfptrain_scaled=pd.DataFrame(dfptrain_scaled, columns= dfptrain_old.columns)
        print('Scaling done')
        #Separamos la informacion entre entrenamiento y testeo
#        X_train = dfptrain_X.values[idxTr,:-len(targetname)]
#        Y_train = dfptrain_Y.values[idxTr,-len(targetname):]
#        X_test = dfptrain_X.values[idxTs,:-len(targetname)]
#        Y_test = dfptrain_Y.values[idxTs,-len(targetname):]
        X_train = dfptrain_X.values[idxTr,:]
        Y_train = dfptrain_Y.values[idxTr]
        X_test = dfptrain_X.values[idxTs,:]
        Y_test = dfptrain_Y.values[idxTs]
        
        print('Y_train shape ' + str(Y_train.shape))
        print('Y_test shape ' + str(Y_test.shape))
        
        
        #Transform back into dataframe
        X_train = pd.DataFrame(X_train,columns=X.columns)
        Y_train = pd.Series(Y_train)
        X_test = pd.DataFrame(X_test,columns=X.columns)
        Y_test = pd.Series(Y_test)
        print('Sets ready')
        
        
        
        #GRID SEARCH ON XGBoost
    #    clf_bp = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, n_jobs=4) #n_jobs=4 da error, probamos con 1
        #buscamos que parametros hacen mejor el trabajo con randomforest
        clf_bp = GridSearchCV(xgb.XGBRegressor(), tuned_parameters, cv=5, n_jobs=-1)
        clf_bp.fit(X_train, Y_train)
        n_trees = clf_bp.best_params_['n_estimators'] 
        maxd = clf_bp.best_params_['max_depth']
        
        
        clf = xgb.XGBRegressor(n_estimators=n_trees, max_depth=maxd, nthread=-1)#.fit(X_train, Y_train)
    #    clf.fit(X_train,Y_train, sample_weight=weight_train) #with weight
        clf.fit(X_train,Y_train) #without weight
        score = clf.score(X_train,Y_train) #obtenemos el score con los datos de training asociado con ese parametro     

    #    feature_importances.append(clf.feature_importances_) #guardamos la importancia de cara parametro en la simulacion
        perm= PermutationImportance(clf).fit(X_test, Y_test)
        perm_weight= perm.feature_importances_
        print(perm_weight.shape)
        feature_imp= {'Feature':X_train.columns, 'Importance': perm_weight}
        feature_imp= pd.DataFrame(feature_imp)
        feature_imp_name= 'feature_importance-kfold_' + str(ind) + '-' + str(targetname) + '.csv'
        
        print('feature shape ' + str(feature_imp.shape))
        
        Feature_mean += perm_weight / 5
        
        feature_imp.to_csv( 'datafiles/feature_importance_XB/' +feature_imp_name) #export feature importance of dataframe in kfold ind for setname
            
        print ("Score training = ", score)
    
            
        y_pred = clf.predict(X_test)#(dfP.values[:,:-1]) #obtenemos las predicciones de X_test
        y_target = Y_test #dfP.values[:,-1] #Y_tot #copiamos Y_test para poder compararlos 
        
        #change shape to fit mayor Y
        y_pred=y_pred.reshape(Ypred[idxTs].shape)
#        y_target=y_target.reshape(Ytarg[idxTs].shape)
        
        Ypred[idxTs] = y_pred
        Ytarg[idxTs] = y_target
        
        print ('MSE = ',(mean_squared_error(y_target, y_pred)))
        print ('MAE = ',(mean_absolute_error(y_target, y_pred)))
        print ('R^2 score =',(r2_score(y_target, y_pred)))
        
        #Guardamos estos resultados
        MSE_XB.append(mean_squared_error(y_target, y_pred))
        MAE_XB.append(mean_absolute_error(y_target, y_pred))
        R2_XB.append(r2_score(y_target, y_pred))
    
    
    print()
    print()
    MSE_XB = np.array(MSE_XB)
    MAE_XB = np.array(MAE_XB)
    R2_XB = np.array(R2_XB)
    print('MSE - 5 Folds : ',MSE_XB.mean())
    print('MAE - 5 Folds : ',MAE_XB.mean())
    print('R^2 - 5 Folds : ',R2_XB.mean())


#    Feature_mean= Feature_mean.mean(axis=0, skipna=True)
    print(Feature_mean)
    Feature_name='Feature importance for ' + str(targetname) + '.csv'
    Features={'Name': X_train.columns, 'Weight': Feature_mean}
    Features=pd.DataFrame(Features)
    
    
    Features.to_csv( 'datafiles/feature_importance_XB/' +Feature_name)
    print('Feature importance exported')
            
    Ypred_XB = Ypred
    Ytarg_XB = Ytarg
    #%%
    
    Ypred_XB=Ypred_XB*(y_max - y_min) + y_min
    Ytarg_XB=Ytarg_XB*(y_max - y_min) + y_min
    #name=str(targetname)+'data.pickle'
    #with open(name, 'wb') as f:
    #    pickle.dump([Ypred_XB, Ytarg_XB], f)
    
    #name=str(targetname)+'XBV1' + 'weight'+ str(rho) + '.pickle' #with weight
    name='XGBoost_' + setname + '_' + targetname + 'V7.pickle'
    pickle.dump(clf, open('saved_models/'+name,'wb'), protocol=2) #Export XGBoost algorithm
    pickle.dump(Scaler, open('saved_models/normalicer'+name, 'wb'), protocol=2) #normalicer for data
    
    pickle.dump([y_max, y_min], open('saved_models/statdata'+name, 'wb'), protocol=2)
    
    print('Files exported')
    
#    Y_XB=pd.DataFrame(Ypred_XB, columns=[targetname])
    New_targetname=targetname + 'XB'
    dfp[New_targetname] = Ypred_XB
    print('dfp final columns')
    print(dfp.columns)
    
    dfptrain_old_name='datafiles/XB_' + setname + '_' + targetname + 'V7.csv'
    dfp.to_csv(dfptrain_old_name)