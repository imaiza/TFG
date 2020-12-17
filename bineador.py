import numpy as np
import pandas as pd
from scipy import stats

def bineador(dfp, targetname, nbins):
    
    dfp2=dfp.sort_values(by='M500c(41)')
    
    X=(dfp2['M500c(41)']) .values
    Y=10**dfp2[targetname] .values
    Y=Y/ (10**dfp2['M500c(41)'])
    
    bmean, bedge, bnumber= stats.binned_statistic(X, Y, statistic='mean',bins=nbins) #bar values
    yerr, bedge2, bnumber2= stats.binned_statistic(X, Y, statistic='std',bins=nbins) #standard deviation values
    
    bincenters=0.5*(bedge[1:]+bedge[:-1])
    return bincenters, bmean, yerr

def bineador3(dfp1,dfp2, targetname, nbins):
    
    #dfp1=dfp1.sort_values(by='M500c(41)')
    
    X=(dfp1['M500c(41)']) .values
    
    Y=(10**dfp1[targetname]/10**dfp1['M500c(41)']) .values
    Z=(10**dfp2[targetname]/10**dfp2['M500c(41)']) .values
    
    W=(Y-Z)/Y
    
    bmean, bedge, bnumber= stats.binned_statistic(X, W, statistic='mean',bins=nbins) #bar values
    yerr, bedge2, bnumber2= stats.binned_statistic(X, W, statistic='std',bins=nbins) #standard deviation values
    
    bincenters=0.5*(bedge[1:]+bedge[:-1])
    return bincenters, bmean, yerr

def binead0r4(dfp1,dfp2, targetname, nbins):
    
    #dfp1=dfp1.sort_values(by='M500c(41)')
    
    X=(dfp1['M500c(41)']) .values
    
    Y=(10**dfp1[targetname]) .values
    Z=(10**dfp2[targetname]) .values
    
    W=(Y-Z)/Y
    
    bmean, bedge, bnumber= stats.binned_statistic(X, W, statistic='mean',bins=nbins) #bar values
    yerr, bedge2, bnumber2= stats.binned_statistic(X, W, statistic='std',bins=nbins) #standard deviation values
    
    bincenters=0.5*(bedge[1:]+bedge[:-1])
    return bincenters, bmean, yerr