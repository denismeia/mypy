'''
    This script has to run INSIDE QGIS PYTHON CONSOLE
'''
def statsaga(start,stop,shape,folder,folderout,nameout,COUNT,MIN,MAX,RANGE,SUM,MEAN,VAR,STDDEV,QUANTILE):
    '''
    This will calculate spatial stats from a poligon over a bunch of
    rasters (tif) in a folder.
    You have to enther all the following parameters. Put 1 for what you want:
        COUNT=0;MIN=0;MAX=0;RANGE=0;SUM=0;MEAN=1;VAR=0;STDDEV=0;QUANTILE=0
    The period of analysis:
        start=2011;stop=2012
    And the data:
        shape = '/path/to/shapefile.shp'
        folder = '/your/tif_images/folder/'
        folderout = '/where/you/will/save/yourStats/'
        nameout = 'name_of_your_stats_file'
    '''
    import processing, os, xuleta2, shutil, time
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
    startTime = time.time()

    tempfolder = '/tmp/processing'
    os.chdir(folder)
    lista = xuleta2.listfiles(folder,extension='*.tif',full=False)
    lista2=[]
    years = [str(x) for x in range(start,stop+1,1)]
    for i in lista:
        if i[:-7] in years:
            lista2.append(i)
        lista2.sort()
    print lista2

    x=processing.runalg('saga:gridstatisticsforpolygons',lista2,shape,COUNT!=0,MIN!=0,MAX!=0,RANGE!=0,SUM!=0,MEAN!=0,VAR!=0,STDDEV!=0,QUANTILE,folderout + nameout + '.shp')
    print 'finished, cleaning up...'
    os.chdir(folder)
    try:
        xuleta2.dbf2csv(folderout + nameout + ".dbf")
        os.remove(folderout + nameout + ".dbf")
        os.remove(folderout + nameout + ".mshp");os.remove(folderout + nameout + ".prj")
        os.remove(folderout + nameout + ".shx");os.remove(folderout + nameout + ".shp")
        if os.path.exists(tempfolder): shutil.rmtree(tempfolder)
        print 'you choose one statistic'
        print 'which gives you a .csv file!'
        print 'please, tidy your spreadsheet \n'
    except:
        os.remove(folderout + nameout + ".mshp");os.remove(folderout + nameout + ".prj")
        os.remove(folderout + nameout + ".shx");os.remove(folderout + nameout + ".shp")
        if os.path.exists(tempfolder): shutil.rmtree(tempfolder)
        print 'You choose more than one statistic'
        print 'you will not have a .csv file, but a .dbf'
        print 'please, tidy your spreadsheet \n'

    print 'A total of ', len(lista2),' images were processed \n' 
    end = (time.time() - startTime)/60
    print 'The whole process took %.2f minutes' %end   

#import xuleta2
start = 2006; stop = 2009
shape = '/media/denis/seagate/QDRI_Project/Shapes/stats_base_counties.shp'
folderout = '/media/denis/seagate/QDRI_Project/Stats/'
#folder = '/media/denis/seagate/QDRI_Project/MODIS/VIs/ndvi/filtrada_1/anomalies/'
folder = '/media/denis/seagate/QDRI_Project/MODIS/VIs/lswi/filtrada_1/anomalies/'
nameout = 'lswi_anom_mean_2006_2009'


COUNT=0;MIN=0;MAX=0;RANGE=0;SUM=0;MEAN=1;VAR=0;STDDEV=0;QUANTILE=0
statsaga(start,stop,shape,folder,folderout,nameout,COUNT,MIN,MAX,RANGE,SUM,MEAN,VAR,STDDEV,QUANTILE)
