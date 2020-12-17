"""
@author: Iñigo Maiza
"""

def RF_train2(filetrain, targetname,setname):
    
    #%% Import libraries

    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        
        
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    from scipy import stats
    
    from eli5.sklearn import PermutationImportance #get feature importance per K-fold

    
    import pickle
#    from sklearn.externals import joblib
    #ML functions
#    from sklearn.model_selection import KFold
    from sklearn.model_selection import KFold, GridSearchCV
#    from sklearn.grid_search import GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
    import sklearn.preprocessing as skpp

    
    
    
    
    #%% Import data
#    if __name__ == "__main__":

    
    dfp=filetrain
    
    print(setname + ' column list:')
    print(dfp.columns)
    print('-----------------------------------')

    print('Data imported')
    

    dfp=dfp[dfp['M500c(41)'] >13.5]

    dfp=dfp.dropna(axis=1)
    
    new_col_list=dfp.columns
    
    print('New column list:')
    print(new_col_list)
    
#    targetname=['G3XMgas(80)','G3XMstar(81)' ,'G3XTgas_mw(82)', 'G3XYx(84)', 'G3XYsz(85)']
    
    dfp.reset_index(drop=True, inplace=True)

    Y_train=dfp[targetname]
    

    dfptrain=dfp.copy()
    
    
    dfptrain.drop(labels=targetname, inplace=True, axis=1)
    
    coldrop=[col for col in dfptrain.columns if 'G3X' in col]
    dfptrain.drop(labels=coldrop, inplace=True, axis=1)
    
    new_col_list=dfptrain.columns
    print(new_col_list)       
    #plot hist
    plt.figure('Mass hist')
    plt.hist(dfp['M500c(41)'], bins=20, zorder=1, label=[setname + ' set'])
    plt.legend()
    
    
    #%% Preprocessing of data
    
    
#    Y_train=np.log10(Y_train)
    
#    Ysz_error=Y_train.index[Y_train['G3XYsz(85)'] == -np.inf]
#    print('This are the index of log10(Ysz)=-inf')
#    print(Ysz_error)
#    
#    Y_train.drop(labels=Ysz_error, axis=0, inplace=True)
#    dfptrain.drop(labels=Ysz_error, axis=0, inplace=True)
    
    #statistical data from Y_train
#    mu=np.mean(Y_train) #median
#    sigma=np.std(Y_train) #standard deviation
    
 
    #%% Analysis of train data
    
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
#   plt.savefig('plots/correlation/correlation_plot.png')

    #%% RF on training data - Creation of RF algorithm
    
    #We get the test/train index
    #indexFolds = KFold(dfptrain.values.shape[0], n_folds=5, shuffle=True, random_state=11)
    indexFolds = KFold(n_splits=5, shuffle=True, random_state=42)
    lVarsTarg=dfptrain.columns
    
    R2_RF = []
    MAE_RF = []
    MSE_RF = []
    tuned_parameters = [
                       {'max_features' : ['auto','sqrt','log2']}
                       ]
    
    
    
    # Recorremos las particiones
    ind = 0
    
    Ypred = np.zeros(np.shape(dfptrain[targetname]))
    Ytarg = np.zeros(np.shape(dfptrain[targetname]))
    
    Feature_mean=np.zeros([new_col_list.shape[0] ,])
    Feature_error_mean=np.zeros([new_col_list.shape[0] ,])
    r2score = np.zeros((new_col_list.shape[0],5)) #Para guardar los 5 K-folds
    r2media = np.zeros([new_col_list.shape[0] ,]) #Para guardar media de K-folds
    score_error=np.zeros([new_col_list.shape[0] ,]) #Para guardar los errores
    scoreFake_mean=np.zeros([new_col_list.shape[0] ,]) #Para guardar los scores default
    
    if len(targetname)== 1:
        print('resizing Y')
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
        
        Y=dfptrain[targetname]
        dfptrain_Y=(Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))
        
#        dfptrain_scaled=pd.DataFrame(dfptrain_scaled, columns= dfptrain_old.columns)
        print('Scaling done')
        #Separamos la informacion entre entrenamiento y testeo
#        X_train = dfptrain_X.values[idxTr,:-len(targetname)]
#        Y_train = dfptrain_Y.values[idxTr,-len(targetname):]
#        X_test = dfptrain_X.values[idxTs,:-len(targetname)]
#        Y_test = dfptrain_Y.values[idxTs,-len(targetname):]
        X_train = dfptrain_X.values[idxTr,:]
        Y_train = dfptrain_Y.values[idxTr,:]
        X_test = dfptrain_X.values[idxTs,:]
        Y_test = dfptrain_Y.values[idxTs,:]
        if len(targetname)==1:
            Y_train = dfptrain_Y.values[idxTr,-len(targetname)]
            Y_test = dfptrain_Y.values[idxTs,-len(targetname)]
            
        #Estandariza la informacion quitando la media y escalando a la unidad
#        norm_train = skpp.StandardScaler().fit(X_train) #Normal L2 transform
#        norm_train = skpp.PowerTransformer().fit(X_train) #Power transform to gaussian like
    
#        X_train = skpp.StandardScaler().fit_transform(X_train) #Normal L2 transform
#        X_train = skpp.PowerTransformer().fit_transform(X_train) #Power transform to gaussian like
        
        #Transform back into dataframe
        X_train = pd.DataFrame(X_train,columns=X.columns)
        Y_train = pd.DataFrame(Y_train,columns=targetname)
        X_test = pd.DataFrame(X_test,columns=X.columns)
        Y_test = pd.DataFrame(Y_test,columns=targetname)
        print('Sets ready')
        
        columnas=X_train.columns
        
        for i in columnas:
            
            X_train_new = X_train.copy()
            X_train_new[i]=np.random.permutation(X_train_new[i].values)
            print('Modificamos ' + str(i)+ 'para K-fold='+str(ind))
                    
            if len(targetname)==1:
                Y_train= np.ravel(Y_train)
                
            #GRID SEARCH ON RANDOM FORSEST
    #    clf_bp = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, n_jobs=4) #n_jobs=4 da error, probamos con 1
            #buscamos que parametros hacen mejor el trabajo con randomforest
            clf_bp = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, n_jobs=-1)
            clf_bp.fit(X_train_new, Y_train)
            max_f = clf_bp.best_params_['max_features'] #nos da el mejor parametro
            clf = RandomForestRegressor(n_estimators=100, max_features=max_f, n_jobs=-1) #creamos el randomforest regressor con ese parametro
            '''
            n_estimators=100 is default for version 0.22 onwards
            '''
          
               
            clf.fit(X_train_new,Y_train)
            score = clf.score(X_train_new,Y_train) #obtenemos el score con los datos de training asociado con ese parametro     
        
#    feature_importances.append(clf.feature_importances_) #guardamos la importancia de cara parametro en la simulacion
            #perm= PermutationImportance(clf).fit(X_test, Y_test)
    
            #perm_weight= (perm.feature_importances_)*100/score
            #perm_weight2= (perm.feature_importances_std_)*100/score
 
            #print(perm_weight.shape)
            #print(perm_weight2.shape)
            #feature_imp= {'Feature':X_train.columns, 'Importance': perm_weight,'Error':perm_weight2}
            #feature_imp= pd.DataFrame(feature_imp)
            #feature_imp_name= 'feature_importance-kfold_' + str(ind) + '-' + str(setname) + '.csv'
        
            #Feature_mean += perm_weight[i] / 5
            #Feature_error_mean += perm_weight2 / 5
        
            #feature_imp.to_csv(feature_imp_name) #export feature importance of dataframe in kfold ind for setname
                
            print ("Score training = ", score)
    
            score_t = clf.score(X_test,Y_test)# score del test
            print ("Score test = ",score_t)
            print ()
            print ()
            
            y_pred = clf.predict(X_test)#(dfP.values[:,:-1]) #obtenemos las predicciones de X_test
            y_target = Y_test #dfP.values[:,-1] #Y_tot #copiamos Y_test para poder compararlos (?)
        
            if len(targetname)==1:
                y_target= np.ravel(y_target)
        
        
        
            Ypred[idxTs,] = y_pred
            Ytarg[idxTs,] = y_target
            
            indice=np.where(columnas==i)
            r2score[indice,ind-1]=r2_score(y_target, y_pred)
            
            #print ('MSE = ',(mean_squared_error(y_target, y_pred)))
            #print ('MAE = ',(mean_absolute_error(y_target, y_pred)))
            #print ('R^2 score =',(r2_score(y_target, y_pred)))
            
            #Guardamos estos resultados
            MSE_RF.append(mean_squared_error(y_target, y_pred))
            MAE_RF.append(mean_absolute_error(y_target, y_pred))
            R2_RF.append(r2_score(y_target, y_pred))
    
        new_imp= {'Shuffled variable':X_train_new.columns, 'Score':r2score[:,ind-1]}
        new_imp_name= 'NEWfeature_importance-kfold_' + str(ind) + '-' + str(targetname) + '.csv'
        new_imp=pd.DataFrame(new_imp)
        new_imp.to_csv('datafiles/feature_importance_RF/' +new_imp_name)
        
        
        
        
        
    print()
    print()
                       
    #Guardamos la media y el error de los 5 K-folds
                       
    for i in np.arange(9):
        r2media[i]= np.mean(r2score[i,:])
        score_error[i]= np.std(r2score[i,:])
                          
                                           
    '''              
    MSE_RF = np.array(MSE_RF)
    MAE_RF = np.array(MAE_RF)
    R2_RF = np.array(R2_RF)
    print('MSE - 5 Folds : ',MSE_RF.mean())
    print('MAE - 5 Folds : ',MAE_RF.mean())
    print('R^2 - 5 Folds : ',R2_RF.mean())
    
    def_score = R2_RF.mean()
    '''
#    Feature_mean= Feature_mean.mean(axis=0, skipna=True)
    print(Feature_mean)
    Feature_name='NEWFeature importance for RF' + str(targetname) + '.csv'
    Features={'Name': X_train.columns,'Error':score_error,'Score':r2media}
    Features=pd.DataFrame(Features)
    
    
    Features.to_csv('datafiles/feature_importance_RF/' +Feature_name)
    print('Feature importance exported')
