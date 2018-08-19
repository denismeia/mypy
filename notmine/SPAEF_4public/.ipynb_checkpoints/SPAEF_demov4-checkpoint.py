# -*- coding: utf-8 -*-


from __future__ import print_function
print(__doc__)

import numpy as np
import sys,csv


sys.path.append('.\py_files')


from SPAEF_metric import SPAEF
from figures import plot_SPAEFstats, plot_maps


mask_1km = np.loadtxt('./map_files/mask_1km.asc', delimiter=',')   # X is an array

dpi = 60 # Arbitrary. The number of pixels in the image will always be identical
height, width = [12,18]#np.array(aET_sim.shape, dtype=float) / dpi
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 16,
        }

obs=[]
sim=[]
############################################################ EXAMPLE: descent monthly match
obs=np.loadtxt('./map_files/obs.asc')   
sim=np.loadtxt('./map_files/sim_1.asc') 

obs[mask_1km==0]=np.nan
sim[mask_1km==0]=np.nan

notnan=np.argwhere(np.isnan(obs)==False)

obss=obs[notnan[:,0],notnan[:,1]]  
simm=sim[notnan[:,0],notnan[:,1]]

obss=obss[np.isnan(obss)==False]
simm=simm[np.isnan(obss)==False]
   
SPAef1, cc, alpha, hh = SPAEF(simm,obss,100)
print ('SPAEF_case1',SPAef1)
########################################################################        PLOT   MAPS and STATS

plot_SPAEFstats(simm,obss,SPAef1, cc, alpha, hh,'Case-1') # first two inputs should be 1D vector

plot_maps(sim,obs,'Map1map') # first two inputs should be 2D matrix (not vector)

############################################################ EXAMPLE: better monthly match

obs=[]
sim=[]
obss=[]
simm=[]

obs=np.loadtxt('./map_files/obs.asc')   
sim=np.loadtxt('./map_files/sim_2.asc') 

obs[mask_1km==0]=np.nan
sim[mask_1km==0]=np.nan

notnan=np.argwhere(np.isnan(obs)==False)

obss=obs[notnan[:,0],notnan[:,1]]  
simm=sim[notnan[:,0],notnan[:,1]]

obss=obss[np.isnan(obss)==False]
simm=simm[np.isnan(obss)==False]

   
SPAef2, cc, alpha, hh = SPAEF(simm,obss,100)
print ('SPAEF_case2',SPAef2)
########################################################################        PLOT   MAPS and STATS

plot_SPAEFstats(simm,obss,SPAef2, cc, alpha, hh,'Case-2') # first two inputs should be 1D vector

plot_maps(sim,obs,'Map2map') # first two inputs should be 2D matrix (not vector)

############################################################ EXAMPLE: poor match
obs=[]
sim=[]
obss=[]
simm=[]

obs=np.loadtxt('./map_files/obs.asc')   
sim=np.loadtxt('./map_files/sim_3.asc') 

obs[mask_1km==0]=np.nan
sim[mask_1km==0]=np.nan

notnan=np.argwhere(np.isnan(obs)==False)

obss=obs[notnan[:,0],notnan[:,1]]  
simm=sim[notnan[:,0],notnan[:,1]]

obss=obss[np.isnan(obss)==False]
simm=simm[np.isnan(obss)==False]

   
SPAef3, cc, alpha, hh = SPAEF(simm,obss,100)
print ('SPAEF_case3',SPAef3)
########################################################################        PLOT   MAPS and STATS

plot_SPAEFstats(simm,obss,SPAef3, cc, alpha, hh,'Case-3') # first two inputs should be 1D vector

plot_maps(sim,obs,'Map3map') # first two inputs should be 2D matrix (not vector)



############################################################ EXAMPLE: poor match
obs=[]
sim=[]
obss=[]
simm=[]

obs=np.loadtxt('./map_files/obs.asc')   
sim=np.loadtxt('./map_files/sim_4.asc') 

obs[mask_1km==0]=np.nan
sim[mask_1km==0]=np.nan

notnan=np.argwhere(np.isnan(obs)==False)

obss=obs[notnan[:,0],notnan[:,1]]  
simm=sim[notnan[:,0],notnan[:,1]]

obss=obss[np.isnan(obss)==False]
simm=simm[np.isnan(obss)==False]

   
SPAef4, cc, alpha, hh = SPAEF(simm,obss,100)
print ('SPAEF_case4',SPAef4)
########################################################################        PLOT   MAPS and STATS

plot_SPAEFstats(simm,obss,SPAef4, cc, alpha, hh,'Case-4') # first two inputs should be 1D vector

plot_maps(sim,obs,'Map4map') # first two inputs should be 2D matrix (not vector)



############################################################ EXAMPLE: poor match
obs=[]
sim=[]
obss=[]
simm=[]

obs=np.loadtxt('./map_files/obs.asc')   
sim=np.loadtxt('./map_files/sim_5.asc') 

obs[mask_1km==0]=np.nan
sim[mask_1km==0]=np.nan

notnan=np.argwhere(np.isnan(obs)==False)

obss=obs[notnan[:,0],notnan[:,1]]  
simm=sim[notnan[:,0],notnan[:,1]]

obss=obss[np.isnan(obss)==False]
simm=simm[np.isnan(obss)==False]

   
SPAef5, cc, alpha, hh = SPAEF(simm,obss,100)
print ('SPAEF_case5',SPAef5)
########################################################################        PLOT   MAPS and STATS

plot_SPAEFstats(simm,obss,SPAef5, cc, alpha, hh,'Case-5-shifted') # first two inputs should be 1D vector

plot_maps(sim,obs,'Map5-shifted') # first two inputs should be 2D matrix (not vector)


############################################################ EXAMPLE: shifted cells
obs=[]
sim=[]
obss=[]
simm=[]

obs=np.loadtxt('./map_files/obs.asc')   
sim=np.loadtxt('./map_files/sim_6.asc') 

obs[mask_1km==0]=np.nan
sim[mask_1km==0]=np.nan

notnan=np.argwhere(np.isnan(obs)==False)

obss=obs[notnan[:,0],notnan[:,1]]  
simm=sim[notnan[:,0],notnan[:,1]]

obss=obss[np.isnan(obss)==False]
simm=simm[np.isnan(obss)==False]

   
SPAef6, cc, alpha, hh = SPAEF(simm,obss,100)
print ('SPAEF_case6',SPAef6)
########################################################################        PLOT   MAPS and STATS

plot_SPAEFstats(simm,obss,SPAef6, cc, alpha, hh,'Case-6') # first two inputs should be 1D vector

plot_maps(sim,obs,'Map6map') # first two inputs should be 2D matrix (not vector)


############################################################ EXAMPLE: biased sim=obsx30 times greater values, locations unchanged
obs=[]
sim=[]
obss=[]
simm=[]

obs=np.loadtxt('./map_files/obs.asc')   
sim=np.loadtxt('./map_files/sim_7.asc') 

obs[mask_1km==0]=np.nan
sim[mask_1km==0]=np.nan

notnan=np.argwhere(np.isnan(obs)==False)

obss=obs[notnan[:,0],notnan[:,1]]  
simm=sim[notnan[:,0],notnan[:,1]]

obss=obss[np.isnan(obss)==False]
simm=simm[np.isnan(obss)==False]

   
SPAef7, cc, alpha, hh = SPAEF(simm,obss,100)
print ('SPAEF_case7',SPAef7)
########################################################################        PLOT   MAPS and STATS

plot_SPAEFstats(simm,obss,SPAef7, cc, alpha, hh,'Case-7-biased') # first two inputs should be 1D vector

plot_maps(sim,obs,'Map7-biased') # first two inputs should be 2D matrix (not vector)




############################################################ EXAMPLE: biased sim=obsx30 times greater values, locations unchanged
obs=[]
sim=[]
obss=[]
simm=[]

obs=np.loadtxt('./map_files/obs.asc')   
sim=np.loadtxt('./map_files/sim_8.asc') 

obs[mask_1km==0]=np.nan
sim[mask_1km==0]=np.nan

notnan=np.argwhere(np.isnan(obs)==False)

obss=obs[notnan[:,0],notnan[:,1]]  
simm=sim[notnan[:,0],notnan[:,1]]

obss=obss[np.isnan(obss)==False]
simm=simm[np.isnan(obss)==False]

   
SPAef8, cc, alpha, hh = SPAEF(simm,obss,100)
print ('SPAEF_case8',SPAef8)
########################################################################        PLOT   MAPS and STATS

plot_SPAEFstats(simm,obss,SPAef8, cc, alpha, hh,'Case-8-shuffled') # first two inputs should be 1D vector

plot_maps(sim,obs,'Map8-shuffled') # first two inputs should be 2D matrix (not vector)


val=np.array([['ID','SPAEF'],['Case1',np.around(SPAef1,7)],['Case2',np.around(SPAef2,7)],['Case3',np.around(SPAef3,7)],['Case4',np.around(SPAef4,7)],['Case5',np.around(SPAef5,7)],['Case6',np.around(SPAef6,7)],['Case7',np.around(SPAef7,7)],['Case8',np.around(SPAef8,7)]])


csvfile = "SPAEF_results.out"

#Assuming res is a flat list
with open('./results/'+csvfile, "w") as output:
    writer = csv.writer(output,delimiter =' ', lineterminator='\n')
    for val in val:
        writer.writerow(val) 
        
### 
output.close()








print ('Welldone')

#sys.modules[__name__].__dict__.clear()



