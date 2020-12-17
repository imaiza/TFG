# -*- coding: utf-8 -*-
"""

Collection of functions used in  plotter.py and predict.py to make graphs.
it includes
    -all_graph: 
            1. Single graph of Mgas, Mstar, Tgas, Yx and Ysz compared to M500c
            2. Single graph of data density of Mgas, Mstar, Tgas, Yx and Ysz compared to M500c
            3. Fraction mass 
    -all_graph_color: single graph of Mgas, Mstar, Tgas, Yx and Ysz compared to M500c using
                      different colours for each expansion parameter
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import stats
from bineador import bineador
from bineadortemp import bineadortemp



def all_graph(dfp_filtered, z, setname, alg_name):
    f=plt.figure('G3X (filtered) RF expansion par='+ z , figsize=(24,13))
    ax1=plt.subplot(2,3,1)
    ax2=plt.subplot(2,3,2)
    ax3=plt.subplot(2,3,3)
    ax4=plt.subplot(2,3,4)
    ax5=plt.subplot(2,3,5)
    dfp_filtered= dfp_filtered[10**dfp_filtered['G3XTgas_mw_NN(82)'] != 0]
    dfp_filtered= dfp_filtered[10**dfp_filtered['G3XYx_NN(84)'] != 0]
    #dfp_filtered= dfp_filtered[dfp_filtered['G3XMstar(86)'] < 10**5] # ~1900/64000 elementos daban error
    dfp_filtered= dfp_filtered.replace([np.inf, -np.inf], np.nan)
    dfp_filtered= dfp_filtered[ dfp_filtered['M500c(41)']>13.5]
    
    M500c=dfp_filtered['M500c(41)']
    
    plt.suptitle('G3X data (filtered) expansion parameter='+z ,fontsize=20)
    Y_temp, X_temp= 10**dfp_filtered['G3XMgas_NN(80)'] / (10**M500c) , M500c
    ax1.scatter(X_temp, Y_temp, marker='o', s=(72./f.dpi)**2,lw=0, label='RF data')
    ax1bincen,ax1bmean,ax1yerr = bineador(dfp_filtered, 'G3XMgas_NN(80)', 20) #binning
    ax1.errorbar(ax1bincen, ax1bmean, yerr=ax1yerr, fmt='sr',markerfacecolor="None",zorder=11)
    ax1.set_title('Mgas/M500c - M500c')
    ax1.set_ylim(0,0.5)
    ax1.set_xlabel('log(M500c) ($h^{-1} M_{\odot}$)')
    ax1.set_ylabel('Mgas/M500c')
    ax1.set_ylim(0.05,0.3)
    ax1.set_xlim(13.5,15.25)
    ax1.legend()
    
    Y_temp, X_temp= 10**dfp_filtered['G3XMstar_NN(81)'] / (10**M500c) , M500c
    ax2.scatter(X_temp, Y_temp, marker='o', s=(72./f.dpi)**2,lw=0, label='RF data ')
    ax2bincen,ax2bmean,ax2yerr = bineador(dfp_filtered,'G3XMstar_NN(81)', 20) #binning
    ax2.errorbar(ax2bincen, ax2bmean, yerr=ax2yerr, fmt='sr',markerfacecolor="None",zorder=11)
    ax2.set_title('Mstar/M500c - M500c')
    ax2.set_xlabel('log(M500c) ($h^{-1} M_{\odot}$)')
    ax2.set_ylim(0.0,0.03)
    ax2.set_xlim(13.5,15.25)
    ax2.set_ylabel('Mstar/M500c')
    ax2.legend()
    
    ax3.scatter(M500c, 10**dfp_filtered['G3XTgas_mw_NN(82)'], marker='o', s=(72./f.dpi)**2,lw=0, label='RF data ')
    ax3bincen,ax3bmean,ax3yerr = bineadortemp(dfp_filtered,'G3XTgas_mw_NN(82)', 20) #binning
    ax3.errorbar(ax3bincen, ax3bmean, yerr=ax3yerr, fmt='sr',markerfacecolor="None",zorder=11)
    ax3.set_title('Tgas - M500c')
    ax3.set_xlabel('log(M500c) ($h^{-1} M_{\odot}$)')
    ax3.set_ylabel('Tgas')
    ax3.set_yscale('log')
    ax3.set_ylim(0.4,13)
    ax3.set_xlim(13.5,15.25)
    ax3.legend()
    
    #fix Units
    dfp_filtered['G3XYsz_NN(85)'] = dfp_filtered['G3XYsz_NN(85)'] - 6
    dfp_filtered['G3XYx_NN(84)'] = dfp_filtered['G3XYx_NN(84)'] - 6
    
    ax4.scatter(M500c, 10**dfp_filtered['G3XYsz_NN(85)'], marker='o', s=(72./f.dpi)**2,lw=0, label='RF data ')
    ax4bincen,ax4bmean,ax4yerr = bineadortemp(dfp_filtered,'G3XYsz_NN(85)', 20) #binning
    ax4.errorbar(ax4bincen, ax4bmean, yerr=ax4yerr,fmt='sr',markerfacecolor="None",zorder=11)
    ax4.set_title('Ysz - M500c')
    ax4.set_xlabel('log(M500c) ($h^{-1} M_{\odot}$)')
    ax4.set_ylabel('Ysz')
    ax4.set_yscale('log')
    ax4.set_ylim(1e-7,1e-3)
    ax4.set_xlim(13.5,15.25)
    ax4.legend()
    
    
    ax5.scatter(M500c, 10**dfp_filtered['G3XYx_NN(84)'], marker='o', s=(72./f.dpi)**2,lw=0, label='RF data ')
    ax5bincen,ax5bmean,ax5yerr = bineadortemp(dfp_filtered,'G3XYx_NN(84)', 20) #binning
    ax5.errorbar(ax5bincen, ax5bmean, yerr=ax5yerr, fmt='sr',markerfacecolor="None",zorder=11)
    ax5.set_title('Yx - M500c')
    ax5.set_xlabel('log(M500c) ($h^{-1} M_{\odot}$)')
    ax5.set_ylabel('Yx')
    ax5.set_yscale('log')
    ax5.set_ylim(1e6,1e10)
    ax5.set_xlim(13.5,15.25)
    ax5.legend()
    
    #manager = plt.get_current_fig_manager()
    #manager.resize(*manager.window.maxsize()) #for TkAgg: https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen/32428266
    
    
#    f.savefig('G3X (filtered) expansion parameter='+z+ ' NN'  + setname + '.pdf',bbox_inches='tight')
#    f.savefig('plots/hlist/'+ alg_name+ '/Mass_Tgas_Yx_Ysz_expansion_parameter='+z+ '_' + alg_name + '_'  + setname + '.pdf',bbox_inches='tight')
    f.savefig('plots/hlist3/'+ alg_name+ '/Mass_Tgas_Yx_Ysz_expansion_parameter='+z+ '_' + alg_name + '_'  + setname + '.png',bbox_inches='tight')
    plt.close()
    
    print('First graph done')
    #%%
    
    
    f_dens=plt.figure('G3X (filtered) - data density expansion parameter='+ z , figsize=(24,13))
    ax1=plt.subplot(2,3,1)
    ax2=plt.subplot(2,3,2)
    ax3=plt.subplot(2,3,3)
    ax4=plt.subplot(2,3,4)
    ax5=plt.subplot(2,3,5)
    #dfp_filtered=dfp[dfp['M500c(41)'] > 10**13.5]
    #dfp_filtered= dfp_filtered.replace([np.inf, -np.inf], np.nan)
    #dfp_filtered= dfp_filtered[10**dfp_filtered['G3XTgas_mw_NN(87)'] != 0]
    #dfp_filtered= dfp_filtered[10**dfp_filtered['G3XYx_NN(89)'] != 0]
    
    import seaborn as sns
    
    plt.suptitle('G3X data (filtered) RF - data density expansion parameter='+z+ ' (log scale)',fontsize=20)
    
    Y_temp, X_temp= 10**dfp_filtered['G3XMgas_NN(80)'] / (10**M500c) , M500c
    hb1=ax1.hexbin(X_temp, Y_temp,bins='log', cmap='inferno', mincnt= 2 )
#    sns.kdeplot(X_temp, Y_temp, cbar=True, ax=ax1, cmap='inferno', shade=True, shade_lowest=False)
    ax1bincen,ax1bmean,ax1yerr = bineador(dfp_filtered, 'G3XMgas_NN(80)', 15) #binning
    ax1.errorbar(ax1bincen, ax1bmean, yerr=ax1yerr, fmt='sg',markerfacecolor="None",zorder=11)
    ax1.set_title('Mgas/M500c - M500c')
    ax1.set_xlabel('log(M500c) ($h^{-1} M_{\odot}$)')
    ax1.set_ylabel('Mgas/M500c')
    ax1.set_ylim(0.05,0.3)
    ax1.set_xlim(13.5,15.25)
    cb1 = plt.colorbar(hb1, ax=ax1)
    cb1.set_label('log10(N)')
    cb1.set_label('N')
    
    Y_temp, X_temp= 10**dfp_filtered['G3XMstar_NN(81)'] / (10**M500c) , M500c
    hb2=ax2.hexbin(X_temp, Y_temp,bins='log', cmap='inferno',mincnt= 2 )
#    sns.kdeplot(X_temp, Y_temp, cbar=True, ax=ax2, cmap='inferno', shade=True, shade_lowest=False)
    ax2bincen,ax2bmean,ax2yerr = bineador(dfp_filtered,'G3XMstar_NN(81)', 20) #binning
    ax2.errorbar(ax2bincen, ax2bmean, yerr=ax2yerr, fmt='sg',markerfacecolor="None",zorder=11)
    ax2.set_title('Mstar/M500c - M500c')
    ax2.set_xlabel('log(M500c) ($h^{-1} M_{\odot}$)')
    ax2.set_ylim(0,0.03)
    ax2.set_xlim(13.5,15.25)
    ax2.set_ylabel('Mstar/M500c')
    cb2 = plt.colorbar(hb2, ax=ax2)
    cb2.set_label('log10(N)')
    cb2.set_label('N')
    
    hb3=ax3.hexbin(M500c[10**dfp_filtered['G3XTgas_mw_NN(82)'] !=0], 10**dfp_filtered['G3XTgas_mw_NN(82)'][10**dfp_filtered['G3XTgas_mw_NN(82)'] !=0], \
               yscale='log',bins='log',cmap='inferno', mincnt= 2 )
#    sns.kdeplot(M500c[10**dfp_filtered['G3XTgas_mw_NN(82)'] !=0], 10**dfp_filtered['G3XTgas_mw_NN(82)'][10**dfp_filtered['G3XTgas_mw_NN(82)'] !=0], cbar=True, ax=ax3, cmap='inferno', shade=True, shade_lowest=False)
    ax3bincen,ax3bmean,ax3yerr = bineadortemp(dfp_filtered,'G3XTgas_mw_NN(82)', 20) #binning
    ax3.errorbar(ax3bincen, (ax3bmean), yerr=(ax3yerr), fmt='sg',markerfacecolor="None",zorder=11)
    ax3.set_title('Tgas - M500c')
    ax3.set_xlabel('log(M500c) ($h^{-1} M_{\odot}$)')
    ax3.set_ylabel('Tgas')
    ax3.set_yscale('log')
    ax3.set_ylim(0.4,13)
    ax3.set_xlim(13.5,15.25)
    cb3 = plt.colorbar(hb3, ax=ax3)
    cb3.set_label('log10(N)')
    cb3.set_label('N')
#    z3=np.polyfit(M500c[10**dfp_filtered['G3XTgas_mw_NN(82)'] !=0], dfp_filtered['G3XTgas_mw_NN(82)'][10**dfp_filtered['G3XTgas_mw_NN(82)'] !=0],1)
#    poly3=np.poly1d(z3)
#    x=np.linspace(13.5,15.25, 100)
#    ax3.plot(x,10**(poly3(x)),'w-')
#    print('log(T)='+ str(round(z3[0],4)) + 'log(M500c) + '+str(round(z3[1],4)))
    
    
    hb4=ax4.hexbin(M500c[10**dfp_filtered['G3XYsz_NN(85)'] !=0], 10**dfp_filtered['G3XYsz_NN(85)'][10**dfp_filtered['G3XYsz_NN(85)'] !=0],\
                   yscale='log',bins='log', cmap='inferno', mincnt= 2 )
#    sns.kdeplot(M500c[10**dfp_filtered['G3XYsz_NN(85)'] !=0], 10**dfp_filtered['G3XYsz_NN(85)'][10**dfp_filtered['G3XYsz_NN(85)'] !=0], cbar=True, ax=ax4, cmap='inferno', shade=True, shade_lowest=False)    
    ax4bincen,ax4bmean,ax4yerr = bineadortemp(dfp_filtered,'G3XYsz_NN(85)', 20) #binning
    ax4.errorbar(ax4bincen, ax4bmean, yerr=ax4yerr, fmt='sg',markerfacecolor="None",zorder=11)
    ax4.set_title('Ysz - M500c')
    ax4.set_xlabel('log(M500c) ($h^{-1} M_{\odot}$)')
    ax4.set_ylabel('Ysz')
    ax4.set_yscale('log')
    ax4.set_ylim(1e-7,1e-3)
    ax4.set_xlim(13.5,15.25)
    cb4 = plt.colorbar(hb4, ax=ax4)
    cb4.set_label('log10(N)')
    cb4.set_label('N')
#    z4=np.polyfit(M500c[10**dfp_filtered['G3XYsz_NN(85)'] !=0], dfp_filtered['G3XYsz_NN(85)'][10**dfp_filtered['G3XYsz_NN(85)'] !=0],1)
#    poly4=np.poly1d(z4)
#    x=np.linspace(13.5,15.25, 100)
#    ax4.plot(x,10**(poly4(x)),'w-')
#    print('log(Ysz)='+ str(round(z4[0],4)) + 'log(M500c) + '+str(round(z4[1],4)))
    
    
    hb5=ax5.hexbin(M500c[10**dfp_filtered['G3XYx_NN(84)'] !=0], 10**dfp_filtered['G3XYx_NN(84)'][10**dfp_filtered['G3XYx_NN(84)'] !=0],yscale='log',bins='log', cmap='inferno', mincnt= 2 )
#    sns.kdeplot(M500c[10**dfp_filtered['G3XYx_NN(84)'] !=0], 10**dfp_filtered['G3XYx_NN(84)'][10**dfp_filtered['G3XYx_NN(84)'] !=0], cbar=True, ax=ax5, cmap='inferno', shade=True, shade_lowest=False)    
    ax5bincen,ax5bmean,ax5yerr = bineadortemp(dfp_filtered,'G3XYx_NN(84)', 20) #binning
    ax5.errorbar(ax5bincen, ax5bmean, yerr=ax5yerr, fmt='sg',markerfacecolor="None",zorder=11)
    ax5.set_title('Yx - M500c')
    ax5.set_xlabel('log(M500c) ($h^{-1} M_{\odot}$)')
    ax5.set_ylabel('Yx')
    ax5.set_yscale('log')
    ax5.set_ylim(1e6,1e10)
    ax5.set_xlim(13.5,15.25)
    cb5 = plt.colorbar(hb5, ax=ax5)
    cb5.set_label('log10(N)')
    cb5.set_label('N')
#    z5=np.polyfit(M500c[10**dfp_filtered['G3XYx_NN(84)'] !=0], dfp_filtered['G3XYx_NN(84)'][10**dfp_filtered['G3XYx_NN(84)'] !=0],1)
#    poly5=np.poly1d(z5)
#    x=np.linspace(13.5,15.25, 100)
#    ax5.plot(x,10**(poly5(x)),'w-')
#    print('log(Yx)='+ str(round(z5[0],4)) + 'log(M500c) + '+str(round(z5[1],4)))
    
#    f_dens.savefig('G3X data (filtered) NN - data density expansion parameter='+z+' (log scale).pdf',bbox_inches='tight' )
    f_dens.savefig('plots/hlist3/'+ alg_name+ '/data_density_expansion_parameter='+z+ '_' + alg_name + '_'  + setname + '.pdf',bbox_inches='tight')
    f_dens.savefig('plots/hlist3/'+ alg_name+ '/data_density_expansion_parameter='+z+ '_' + alg_name + '_'  + setname + '.png',bbox_inches='tight')
    
    plt.close()
    #manager = plt.get_current_fig_manager()
    #manager.resize(*manager.window.maxsize()) #for TkAgg: https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen/32428266
    
    
#    output= open('plots/hlist/'+alg_name+ '/lin_regr_expansion_parameter='+z+alg_name+ '_'  + setname + '.txt', 'w')
#    output.write('log(T)='+ str(round(z3[0],4)) + 'log(M500c) + '+str(round(z3[1],4)))
#    output.write('\nlog(Ysz)='+ str(round(z4[0],4)) + 'log(M500c) + '+str(round(z4[1],4)))
#    output.write('\nlog(Yx)='+ str(round(z5[0],4)) + 'log(M500c) + '+str(round(z5[1],4)))
#    output.close()
    
    print('second graph done')
    #%%
    '''
    Fgas=10**dfp_filtered['G3XMgas_NN(80)'] / 10**M500c
    Fstar=10**dfp_filtered['G3XMstar_NN(81)'] / 10**M500c
    Fmass=(10**dfp_filtered['G3XMgas_NN(80)'] + 10**dfp_filtered['G3XMstar_NN(81)']) / 10**M500c
    
    gas_inf= pd.read_csv('/home2/frobledo/FINAL/datafiles/BF-gas-G3X-figure.txt')
    gas_inf= gas_inf.values
    
    star_inf= pd.read_csv('/home2/frobledo/FINAL/datafiles/BF-star-G3X-figure.txt')
    star_inf= star_inf.values
    
    f1=plt.figure('Fgas NN expansion parameter='+z+ '')
    plt.scatter(M500c, Fgas, marker='o', s=(72./f1.dpi)**2,lw=0)
    plt.plot(gas_inf[:,0], gas_inf[:,1], '*g', zorder=15, label='GADGET-X')
    bmean, bedge, bnumber= stats.binned_statistic(M500c, Fgas, statistic='mean',bins=20)
    yerr, bedge2, bnumber2= stats.binned_statistic(M500c, Fgas, statistic='std',bins=20)
    bincenters=0.5*(bedge[1:]+bedge[:-1])
    plt.errorbar(bincenters, bmean, yerr=yerr, fmt='sr',markerfacecolor="None", zorder=10)
    plt.title('Fgas (Mgas/M500c) expansion parameter='+z+ '')
    plt.xlabel('M500c')
    plt.ylabel('Fgas')
    plt.ylim([0.01,0.2])
    plt.xlim([13.5,15.25])
    plt.yscale('log')
    plt.plot([13.5,15.25], [0.15163,0.15163], '-r', label='Baryon Fraction', zorder=20)
    f1.savefig('plots/hlist/'+alg_name+'/Fgas_'+alg_name+'_expansion_parameter='+z+'_'  + setname + '.pdf')
    f1.savefig('plots/hlist/'+alg_name+'/Fgas_'+alg_name+'_expansion_parameter='+z+'_'  + setname + '.png')
    plt.legend()
    plt.close()
    
    
    f2=plt.figure('Fstar NN expansion parameter='+z+ '')
    plt.scatter(M500c, Fstar, marker='o', s=(72./f2.dpi)**2,lw=0)
    plt.plot(star_inf[:,0], star_inf[:,1], '*g', zorder=15, label='GADGET-X')
    bmean, bedge, bnumber= stats.binned_statistic(M500c, Fstar, statistic='mean',bins=20)
    yerr, bedge2, bnumber2= stats.binned_statistic(M500c, Fstar, statistic='std',bins=20)
    bincenters=0.5*(bedge[1:]+bedge[:-1])
    plt.errorbar(bincenters, bmean, yerr=yerr, fmt='sr',markerfacecolor="None", zorder=10)
    plt.title('Fstar (Mstar/M500c) expansion parameter='+z+ '')
    plt.xlabel('M500c')
    plt.ylabel('Fstar')
    plt.ylim([0.01,0.03])
    plt.xlim([13.5,15.25])
    plt.yscale('log')
    f2.savefig('plots/hlist/'+alg_name+'/Fstar_'+alg_name+'_expansion_parameter='+z+'_'  + setname + '.pdf')
    f2.savefig('plots/hlist/'+alg_name+'/Fstar_'+alg_name+'_expansion_parameter='+z+'_'  + setname + '.png')
    plt.close()
    
#    f3=plt.figure('Fstar NN - noscale expansion parameter='+z+ '')
#    plt.scatter(M500c, Fstar,  marker='o', s=(72./f3.dpi)**2,lw=0)
#    bmean, bedge, bnumber= stats.binned_statistic(M500c, Fstar, statistic='mean',bins=20)
#    yerr, bedge2, bnumber2= stats.binned_statistic(M500c, Fstar, statistic='std',bins=20)
#    bincenters=0.5*(bedge[1:]+bedge[:-1])
#    plt.errorbar(bincenters, bmean, yerr=yerr, fmt='sr',markerfacecolor="None", zorder=10)
#    plt.title('Fstar (Mstar/M500c) expansion parameter='+z+ '')
#    plt.xlabel('M500c')
#    plt.ylabel('Fstar')
#    plt.yscale('log')
#    plt.xscale('log')
#    f3.savefig('plots/hlist/'+alg_name+'/Fstar_'+alg_name+'-_noscale_expansion_parameter='+z+'_'  + setname + '.pdf')
#    f3.savefig('plots/hlist/'+alg_name+'/Fstar_'+alg_name+'-_noscale_expansion_parameter='+z+'_'  + setname + '.png')
#    plt.close()
    
    
    f4=plt.figure('Fmass NN expansion parameter='+z+ '')
    plt.scatter(M500c, Fmass,  marker='o', s=(72./f4.dpi)**2,lw=0)
    bmean, bedge, bnumber= stats.binned_statistic(M500c, Fmass, statistic='mean',bins=20)
    yerr, bedge2, bnumber2= stats.binned_statistic(M500c, Fmass, statistic='std',bins=20)
    bincenters=0.5*(bedge[1:]+bedge[:-1])
    plt.errorbar(bincenters, bmean, yerr=yerr, fmt='sr',markerfacecolor="None", zorder=10)
    plt.title('Fmass ((Mgas + Mstar)/M500c) expansion parameter='+z+ '')
    plt.xlabel('M500c')
    plt.ylabel('Fmass')
    plt.yscale('log')
    plt.ylim([0,0.3])
    plt.xlim([13.5,15.25])
    f4.savefig('plots/hlist/'+alg_name+'/Fmass_'+alg_name+'_expansion_parameter='+z+'_'  + setname + '.pdf')
    f4.savefig('plots/hlist/'+alg_name+'/Fmass_'+alg_name+'_expansion_parameter='+z+'_'  + setname + '.png')
    plt.close()
    
    print('F graphs done')
'''

def all_graph_color(dfp_filtered, z, setname, alg_name):
    

    
    f=plt.figure('G3X (filtered) RF expansion par='+ z, figsize=(24,13))
    ax1=plt.subplot(2,3,1)
    ax2=plt.subplot(2,3,2)
    ax3=plt.subplot(2,3,3)
    ax4=plt.subplot(2,3,4)
    ax5=plt.subplot(2,3,5)
    dfp_filtered= dfp_filtered[10**dfp_filtered['G3XTgas_mw_NN(82)'] != 0]
    dfp_filtered= dfp_filtered[10**dfp_filtered['G3XYx_NN(84)'] != 0]
    #dfp_filtered= dfp_filtered[dfp_filtered['G3XMstar(86)'] < 10**5] # ~1900/64000 elementos daban error
    dfp_filtered= dfp_filtered.replace([np.inf, -np.inf], np.nan)
    dfp_filtered= dfp_filtered[ dfp_filtered['M500c(41)']>13.5]
    M500c=dfp_filtered['M500c(41)']
    
    
    #fix Units
    dfp_filtered['G3XYsz_NN(85)'] = dfp_filtered['G3XYsz_NN(85)'] - 6
    dfp_filtered['G3XYx_NN(84)'] = dfp_filtered['G3XYx_NN(84)'] - 6
    
    
    a=dfp_filtered['a']
    a=10**a
    
    a=a.values
    
    plt.suptitle('G3X data (filtered)  expansion parameter='+z+ '',fontsize=20)
    
    
    Y_temp, X_temp= 10**dfp_filtered['G3XMgas_NN(80)'] / (10**M500c) , M500c
    sc1=ax1.scatter(X_temp, Y_temp, c=a, marker='o', s=(72./f.dpi)**2,lw=0, label='RF data ')
    ax1bincen,ax1bmean,ax1yerr = bineador(dfp_filtered, 'G3XMgas_NN(80)', 20) #binning
    ax1.errorbar(ax1bincen, ax1bmean, yerr=ax1yerr, fmt='sr',markerfacecolor="None",zorder=11)
    cb1 = plt.colorbar(sc1, ax=ax1)    
    ax1.set_title('Mgas/M500c - M500c')
    ax1.set_ylim(0,0.5)
    ax1.set_xlabel('log(M500c) ($h^{-1} M_{\odot}$)')
    ax1.set_ylabel('Mgas/M500c')
    ax1.set_ylim(0.05,0.3)
    ax1.set_xlim(13.5,15.25)
    ax1.legend()
    
    Y_temp, X_temp= 10**dfp_filtered['G3XMstar_NN(81)'] / (10**M500c) , M500c
    sc2= ax2.scatter(X_temp, Y_temp, c=a, marker='o', s=(72./f.dpi)**2,lw=0, label='RF data ')
    ax2bincen,ax2bmean,ax2yerr = bineador(dfp_filtered,'G3XMstar_NN(81)', 20) #binning
    ax2.errorbar(ax2bincen, ax2bmean, yerr=ax2yerr, fmt='sr',markerfacecolor="None",zorder=11)
    cb2 = plt.colorbar(sc2, ax=ax2)    
    ax2.set_title('Mstar/M500c - M500c')
    ax2.set_xlabel('log(M500c) ($h^{-1} M_{\odot}$)')
    ax2.set_ylim(0.0,0.03)
    ax2.set_xlim(13.5,15.25)
    ax2.set_ylabel('Mstar/M500c')
    ax2.legend()
    
    sc3=ax3.scatter(M500c, 10**dfp_filtered['G3XTgas_mw_NN(82)'],c=a, marker='o', s=(72./f.dpi)**2,lw=0, label='RF data ')
    ax3bincen,ax3bmean,ax3yerr = bineadortemp(dfp_filtered,'G3XTgas_mw_NN(82)', 20) #binning
    cb3 = plt.colorbar(sc3, ax=ax3)    
    ax3.errorbar(ax3bincen, ax3bmean, yerr=ax3yerr, fmt='sr',markerfacecolor="None",zorder=11)
    ax3.set_title('Tgas - M500c')
    ax3.set_xlabel('log(M500c) ($h^{-1} M_{\odot}$)')
    ax3.set_ylabel('Tgas')
    ax3.set_yscale('log')
    ax3.set_ylim(0.4,13)
    ax3.set_xlim(13.5,15.25)
    ax3.legend()
    
    sc4=ax4.scatter(M500c, 10**dfp_filtered['G3XYsz_NN(85)'],c=a, marker='o', s=(72./f.dpi)**2,lw=0, label='RF data ')
    ax4bincen,ax4bmean,ax4yerr = bineadortemp(dfp_filtered,'G3XYsz_NN(85)', 20) #binning
    ax4.errorbar(ax4bincen, ax4bmean, yerr=ax4yerr,fmt='sr',markerfacecolor="None",zorder=11)
    cb4 = plt.colorbar(sc4, ax=ax4)    
    ax4.set_title('Ysz - M500c')
    ax4.set_xlabel('log(M500c) ($h^{-1} M_{\odot}$)')
    ax4.set_ylabel('Ysz')
    ax4.set_yscale('log')
    ax4.set_ylim(1e-7,1e-3)
    ax4.set_xlim(13.5,15.25)
    ax4.legend()
    
    sc5= ax5.scatter(M500c, 10**dfp_filtered['G3XYx_NN(84)'],c=a, marker='o', s=(72./f.dpi)**2,lw=0, label='RF data ')
    ax5bincen,ax5bmean,ax5yerr = bineadortemp(dfp_filtered,'G3XYx_NN(84)', 20) #binning
    ax5.errorbar(ax5bincen, ax5bmean, yerr=ax5yerr, fmt='sr',markerfacecolor="None",zorder=11)
    cb5 = plt.colorbar(sc5, ax=ax5)    
    ax5.set_title('Yx - M500c')
    ax5.set_xlabel('log(M500c) ($h^{-1} M_{\odot}$)')
    ax5.set_ylabel('Yx')
    ax5.set_yscale('log')
    ax5.set_ylim(1e6,1e10)
    ax5.set_xlim(13.5,15.25)
    ax5.legend()
    
    #manager = plt.get_current_fig_manager()
    #manager.resize(*manager.window.maxsize()) #for TkAgg: https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen/32428266
    
    
#    f.savefig('G3X (filtered) expansion parameter='+z+ ' NN'  + setname + '.pdf',bbox_inches='tight')
    f.savefig('plots/hlist3/'+alg_name+'/Mass_Tgas_Yx_Ysz_expansion_parameter='+z+ '_'+alg_name  + setname + '.pdf',bbox_inches='tight')
    f.savefig('plots/hlist3/'+alg_name+'/Mass_Tgas_Yx_Ysz_expansion_parameter='+z+ '_'+alg_name  + setname + '.png',bbox_inches='tight')
    plt.close()



