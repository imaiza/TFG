"""
@author: Iñigo Maiza
"""

def RF_train(filetrain, targetname,setname):
    
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
    score1=[]
    score2=[]
    error1=[]
    error2=[]
        
    dfp_old=filetrain
    new_df=pd.DataFrame()
    target=dfp_old.columns[-1]
    
    for i in dfp_old.columns:
    
        new_df[i]=dfp_old[i] #Vamos añadiendo una nueva
    
        new_df[target]=dfp_old[target]
        
        #Reordenamos para que el target vaya el último
        cols= new_df.columns.tolist()
        cols.remove(target)
        index=len(cols)
        cols.insert(index,target)
        new_df=new_df[cols]
        
        dfp=new_df #Volvemos a llamarlo dfp para mantener el código
        
        #Guardamos las columnas
        porsica=dfp.copy()
        kolumnas=porsica.columns
   
    
    
    
        print(setname + ' column list:')
        print(dfp.columns)
        print('-----------------------------------')

        print('Data imported')
    

        #dfp=dfp[dfp['M500c(41)'] >13.5]

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
        
        
        dfptrain[targetname]=Y_train
    
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
       
#   plt.savefig('plots/correlation/correlation_plot.png')

    #%% RF on training data - Creation of RF algorithm
    
    #We get the test/train index
    #indexFolds = KFold(dfptrain.values.shape[0], n_folds=5, shuffle=True, random_state=11)
        indexFolds = KFold(n_splits=5, shuffle=True, random_state=11)
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
        scoreFake_mean=np.zeros([new_col_list.shape[0] ,])
        
    
        if len(targetname)==1:
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
        
            dfptrain_old=dfptrain.copy() #backup
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
         
            #Transform back into dataframe
            X_train = pd.DataFrame(X_train,columns=X.columns)
            Y_train = pd.DataFrame(Y_train,columns=targetname)
            X_test = pd.DataFrame(X_test,columns=X.columns)
            Y_test = pd.DataFrame(Y_test,columns=targetname)
            print('Sets ready')
            
            
        
            #GRID SEARCH ON RANDOM FORSEST
    
            #buscamos que parametros hacen mejor el trabajo con randomforest
            clf_bp = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, n_jobs=-1)
            clf_bp.fit(X_train, Y_train)
            max_f = clf_bp.best_params_['max_features'] #nos da el mejor parametro
            clf = RandomForestRegressor(n_estimators=100, max_features=max_f, n_jobs=-1) #creamos el randomforest regressor con ese parametro
            
            clf.fit(X_train,Y_train)
            score = clf.score(X_train,Y_train) #obtenemos el score con los datos de training asociado con ese parametro
            #scoreFake=clf.feature_importances_
        

            print ("Score training = ", score)
    
            score_t = clf.score(X_test,Y_test)# score del test
            print ("Score test = ",score_t)
            
            y_pred = clf.predict(X_test)#(dfP.values[:,:-1]) #obtenemos las predicciones de X_test
            y_target = Y_test #dfP.values[:,-1] #Y_tot #copiamos Y_test para poder compararlos (?)
        
            if len(targetname)==1:
                y_target= np.ravel(y_target)
            
        
        
            Ypred[idxTs,] = y_pred
            Ytarg[idxTs,] = y_target
        
            print ('MSE = ',(mean_squared_error(y_target, y_pred)))
            print ('MAE = ',(mean_absolute_error(y_target, y_pred)))
            print ('R^2 score =',(r2_score(y_target, y_pred)))
        
            #Guardamos estos resultados
            MSE_RF.append(mean_squared_error(y_target, y_pred))
            MAE_RF.append(mean_absolute_error(y_target, y_pred))
            R2_RF.append(r2_score(y_target, y_pred))
    
    
        print()
        print()
        MSE_RF = np.array(MSE_RF)
        MAE_RF = np.array(MAE_RF)
        R2_RF = np.array(R2_RF)
        print('MSE - 5 Folds : ',MSE_RF.mean())
        print('MAE - 5 Folds : ',MAE_RF.mean())
        print('R^2 - 5 Folds : ',R2_RF.mean())
    
        def_score1 = R2_RF.mean()
        def_score2 = MSE_RF.mean()
        def_error1 = np.std(R2_RF)
        def_error2 = np.std(MSE_RF)
        
        score1.append(def_score1)
        score2.append(def_score2)
        error1.append(def_error1)
        error2.append(def_error2)
        print('Score1:'+ str(score1))
        print('Su errorres:'+ str(error1))
       
    score1=np.array(score1) 
    score2=np.array(score2)
    error1=np.array(error1)
    error2=np.array(error2)
    
    print('Se acabo')
    print(score1.shape)
    print(score2.shape)
    print(error1.shape)
    print(error2.shape)
    
    kolumnas=np.array(kolumnas)
    print('Columnas:')
    print(kolumnas.shape)
    name='Score evolution for RF' + str(targetname) + '.csv'
    Features={'Name': kolumnas, 'R2': score1, 'MSE': score2 ,'ErrorR2':error1,'ErrorMSE':error2}
    Features=pd.DataFrame(Features)
    
                        
    Features.to_csv(name)
    print('All data exported')
        
        
    '''  
        Ypred_RF = pd.DataFrame(Ypred,columns=targetname)
        Ytarg_RF = pd.DataFrame(Ytarg,columns=targetname)
    #%%
    
        for target in targetname:
            Ypred_RF[target]=Ypred_RF[target]*(y_max[target] - y_min[target]) + y_min[target]
            Ytarg_RF[target]=Ytarg_RF[target]*(y_max[target] - y_min[target]) + y_min[target]
    #name=str(targetname)+'data.pickle'
    #with open(name, 'wb') as f:
    #    pickle.dump([Ypred_RF, Ytarg_RF], f)
    
    name='RF'+ str(targetname) + setname +'MonotargetV7.pickle'
    pickle.dump(clf, open('saved_models/' +name,'wb'), protocol=2) #Export RF algorithm
    #joblib.dump(clf,name)
    pickle.dump(Scaler, open('saved_models/normalicer'+name, 'wb'), protocol=2) #normalicer for data

    
    pickle.dump([y_max, y_min], open('saved_models/statdata'+name, 'wb'), protocol=2)


    #target_NN=['G3XMgas_NN(80)','G3XMstar_NN(81)' ,'G3XTgas_mw_NN(82)', 'G3XYx_NN(84)', 'G3XYsz_NN(85)']
    
    Y_NN= pd.DataFrame(data=Ypred_RF.values, columns= targetname)
    
    dfptrain_old=pd.concat([dfptrain_old, Y_NN], axis=1)
    dfptrain_old_name='RF_' + str(targetname)+ setname + 'V7'
    dfptrain_old.to_csv(dfptrain_old_name)
    '''
    print('Data exported')
    #%%

#    f1=plt.figure('Mgas RF'+ setname)
#    plt.scatter(Ytarg_RF['G3XMgas(80)'].values, Ypred_RF['G3XMgas(80)'].values,marker='o', s=(72./f1.dpi)**2,lw=0)    
#    plt.plot(np.linspace(min(Ytarg_RF['G3XMgas(80)'].values), max(Ytarg_RF['G3XMgas(80)'].values)), \
#             np.linspace(min(Ytarg_RF['G3XMgas(80)'].values), max(Ytarg_RF['G3XMgas(80)'].values)), '-r' )
#    plt.title('Mgas - RF vs real '+ setname)
#    plt.xlabel('Mgas real - log scale')
#    plt.ylabel('Mgas RF - log scale')
#    f1.savefig('Mgas RF'+ setname + ".pdf", bbox_inches='tight')
#    plt.close()
#    
#    f2=plt.figure('Mstar RF'+ setname)
#    plt.scatter(Ytarg_RF['G3XMstar(81)'].values, Ypred_RF['G3XMstar(81)'].values,marker='o', s=(72./f2.dpi)**2,lw=0)
#    plt.plot(np.linspace(min(Ytarg_RF['G3XMstar(81)'].values), max(Ytarg_RF['G3XMstar(81)'].values)), \
#             np.linspace(min(Ytarg_RF['G3XMstar(81)'].values), max(Ytarg_RF['G3XMstar(81)'].values)), '-r' )
#    plt.title('Mstar - RF vs real ' + setname)
#    plt.xlabel('Mstar real - log scale')
#    plt.ylabel('Mstar RF - log scale')
#    f2.savefig('Mstar RF'+ setname+ ".pdf", bbox_inches='tight')
#    plt.close()
#    
#    f3=plt.figure('Tgas RF'+ setname)
#    plt.scatter(Ytarg_RF['G3XTgas_mw(82)'].values, Ypred_RF['G3XTgas_mw(82)'].values,marker='o', s=(72./f3.dpi)**2,lw=0)
#    plt.plot(np.linspace(min(Ytarg_RF['G3XTgas_mw(82)'].values), max(Ytarg_RF['G3XTgas_mw(82)'].values)), \
#             np.linspace(min(Ytarg_RF['G3XTgas_mw(82)'].values), max(Ytarg_RF['G3XTgas_mw(82)'].values)), '-r' )
#    plt.title('Tgas - RF vs real ' + setname)
#    plt.xlabel('Tgas real - log scale')
#    plt.ylabel('Tgas RF - log scale')
#    f3.savefig('Tgas RF'+ setname+ ".pdf", bbox_inches='tight')
#    plt.close()
#
#    f4=plt.figure('G3XYx RF'+ setname)
#    plt.scatter(Ytarg_RF['G3XYx(84)'].values, Ypred_RF['G3XYx(84)'].values,marker='o', s=(72./f4.dpi)**2,lw=0)
#    plt.plot(np.linspace(min(Ytarg_RF['G3XYx(84)'].values), max(Ytarg_RF['G3XYx(84)'].values)), \
#             np.linspace(min(Ytarg_RF['G3XYx(84)'].values), max(Ytarg_RF['G3XYx(84)'].values)), '-r' )
#    plt.title('Yx - RF vs real ' + setname)
#    plt.xlabel('G3XYx real - log scale')
#    plt.ylabel('G3XYx RF - log scale')
#    f4.savefig('G3XYx RF'+ setname+ ".pdf", bbox_inches='tight')
#    plt.close()
#    
#    f5=plt.figure('G3XYsz RF'+setname)
#    plt.scatter(Ytarg_RF['G3XYsz(85)'].values, Ypred_RF['G3XYsz(85)'].values,marker='o', s=(72./f5.dpi)**2,lw=0)
#    plt.plot(np.linspace(min(Ytarg_RF['G3XYsz(85)'].values), max(Ytarg_RF['G3XYsz(85)'].values)), \
#             np.linspace(min(Ytarg_RF['G3XYsz(85)'].values), max(Ytarg_RF['G3XYsz(85)'].values)), '-r' )
#    plt.title('Yx - RF vs real '+setname)
#    plt.xlabel('G3XYsz real - log scale')
#    plt.ylabel('G3XYsz RF - log scale')
#    f5.savefig('G3XYsz RF'+setname+ ".pdf", bbox_inches='tight')
#    plt.close()
    
  