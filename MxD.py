"""
Created on Sat Apr 19 12:08:54 2014
MOD & MYD combination script
This script combines MODIS TERRA and AQUA images. The methodology is described as follow:
    The script compares the blue band of AQUA and TERRA images and apply to all the bands the data
    from the winner satellite (winner is the one with lowest blue value for that pixel)
    The script also eliminates data for where the blue band value is larger than the threshold,
    for instance, 0.15 (reflectance (1500))


@authors: 
Denis Araujo Mariano
denis.mariano@usp.br

William Foschiera
wfoschiera@gmail.com


"""

import gdal, glob, time, os
import numpy as np

startTime = time.time()

#Edit these paths for the INPUT folders for MOD and MYD bands
#Set your output directory
#Set up your cloudh threshold
pathMOD = '/media/denis/seagate/MT/09A1/mod/prm/'
pathMYD = '/media/denis/seagate/MT/09A1/myd/prm/'
pathOUT = '/media/denis/seagate/MT/MODIS/mxd/'
threshold = 1500

##############################################################
#Don't mess after this point
if not os.path.exists(pathOUT): os.makedirs(pathOUT)
listMOD = glob.glob(pathMOD + '*.tif')
listMYD = glob.glob(pathMYD + '*.tif')

datum = gdal.Open(listMOD[0])
driver = gdal.GetDriverByName('GTiff')
geot = datum.GetGeoTransform()
proj = datum.GetProjection()
cols = datum.RasterYSize
rows = datum.RasterXSize

def listcheck():
    status = []
    datasMOD = []
    listaMOD = []
    for i in listMOD:
        i.split('.')
        datasMOD.append(i[-25:-17])
        for j in datasMOD:
            if j not in listaMOD:
                listaMOD.append(j) 

    datasMYD = []
    listaMYD = []
    for i in listMYD:
        i.split('.')
        datasMYD.append(i[-25:-17])
        for j in datasMYD:
            if j not in listaMYD:
                listaMYD.append(j)

    for i in range(len(listaMYD)):
        if len(listaMYD) != len(listaMOD):
            print 'Please check if your input folders contain exactly the same dates and bands'
            print 'for both MYD and MOD'
            break
        listaMYD.sort()
        listaMOD.sort()
        print 'Date %s is %s ' %(listaMOD[i],listaMOD[i] == listaMYD[i])
        status.append(listaMYD[i]==listaMOD[i])
        status2 = status.count(False)
    return listaMYD, listaMOD, status2
    
def applymask(listcheck):
    listaMYD, listaMOD, status = listcheck
    if status != 0:
        print 'Lists are not equal'
        return        
    for i in listaMYD:
        myd3 = gdal.Open(pathMYD + 'MYD09A1.'+ i + '.sur_refl_b03.tif').ReadAsArray()
        mod3 = gdal.Open(pathMOD + 'MOD09A1.'+ i + '.sur_refl_b03.tif').ReadAsArray()
        
        mydmask = (np.where(myd3 <= mod3, 1, 0)) * np.where(myd3 < threshold, 1, 0)
        modmask = (np.where(mod3 < myd3, 1, 0)) * np.where(mod3 < threshold, 1, 0)
        for j in range(7):
            j = j+1
            print 'Processing band %s of date %s ' %(j, i)
            mydj = gdal.Open(pathMYD + 'MYD09A1.'+ i + '.sur_refl_b0' + str(j) + '.tif').ReadAsArray()
            modj = gdal.Open(pathMOD + 'MOD09A1.'+ i + '.sur_refl_b0' + str(j) + '.tif').ReadAsArray()
            mydj = mydj * mydmask
            modj = modj * modmask
            mxdj = modj + mydj
            mxd_saida = driver.Create(pathOUT + 'MxD09A1.'+ i + '.sur_refl_b0' + str(j) + '.tif', rows, cols, 1, gdal.GDT_Int16)
            mxd_saida.SetGeoTransform(geot) # set the datum
            mxd_saida.SetProjection(proj)
            mxd_saida.GetRasterBand(1).WriteArray(mxdj) 
            mydj, modj, mxdj, mxd_saida = None, None, None, None
    print 'The whole processing took ', time.time() - startTime, 'seconds'
    print 'A total of %d bands for %d dates were processed' %(len(listMOD)/len(listaMOD),len(listaMOD))
    myd3, mod3, mydmask, modmask = None, None, None, None  
    return
    
applymask(listcheck())
#END