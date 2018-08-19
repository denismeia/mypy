# CALCULATE ANOMALIES FOR CHIRPS precipitation DATA
# Author: Denis Araujo Mariano
# Last edit: March, 9, 2016
# email: mariano@huskers.unl.edu OR denis.mariano@usp.br
#----------------
# This conchasumadre calculates standard deviation and averages
# 	to finally calculate z-scores (anomalies) for
#	chirps pentad data.
#----------------
# HOWEVER, pentad anomalies don't make any sense
# next step is to convert to monthly anomalies.

import xuleta2, os, shutil
import numpy as np

#input folder with CHIRP files
folder = '/media/denis/seagate/prec_chirps/USA2/'

#list of files
files = xuleta2.listfiles(folder,full=False)
len(files),files[9],files[3][17:-6]

#creating output folders
folderout = folder+'anomalies/'
foldermean = folderout+'mean/'
folderstd = folderout+'std/'
if not os.path.exists(folderout): os.makedirs(folderout)
if not os.path.exists(foldermean): os.makedirs(foldermean)
if not os.path.exists(folderstd): os.makedirs(folderstd)

#create list of pentads (or days)
pentads = []
for i in files:
    pentads.append(i[17:-6])
    pentads.sort()
pentads = [ii for n,ii in enumerate(pentads) if ii not in pentads[:n]]

#Calculates means and standard deviations for each day in a series
for p in pentads:
    toOpen = []
    for i in files:
        if i[17:-6]==p:
            toOpen.append(i)
    arrays = []
    for j in toOpen:
        image, meta = xuleta2.TifToArray(folder+j)
        #something to avoid no-data
        image = np.where(image<-0,np.nan,image)
        arrays.append(image)
        arrays2 = np.dstack(arrays)

    print 'calculating mean and standard deviation for the pentad ' +p
    xuleta2.ArrayToTif(np.nanmean(arrays2,axis=2),p,foldermean, meta, Type=3)
    xuleta2.ArrayToTif(np.nanstd(arrays2,axis=2),p,folderstd, meta, Type=3)
    toOpen, arrays, arrays2, image = None,None,None,None

#Creating list of files for mean and std
lmean = xuleta2.listfiles(foldermean)
lstd = xuleta2.listfiles(folderstd)

#Calculating z-scores
for i in files:
    for mean_,std_ in zip(lmean,lstd):
        if i[17:-6] == mean_[:4]:
            image, meta = xuleta2.TifToArray(folder+i)
            mean, meta = xuleta2.TifToArray(foldermean+mean_)
            std, meta = xuleta2.TifToArray(folderstd+std_)
            print 'Calculating z-score for '+i
            xuleta2.ArrayToTif((image - mean)/std, i[:-4], folderout, meta, Type=3)
        image, mean, std, meta = None, None, None, None
lmean,lstd = None, None

print 'Done!'

#if os.path.exists(foldermean): shutil.rmtree(foldermean)
#if os.path.exists(folderstd): shutil.rmtree(folderstd)