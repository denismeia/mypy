'''
    Developed by Denis Mariano
    @mail = mariano@huskers.unl.edu
    
    
    THIS WORKS ONLY INSIDE QGIS PYTHON CONSOLE!!!
        Don't ask me why!
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
import xuleta2, time
startall = time.time()

#Constant parameters
COUNT=0;MIN=0;MAX=0;RANGE=0;SUM=0;MEAN=1;VAR=0;STDDEV=0;QUANTILE=0
start=2006;stop=2009
shape = '/media/denis/seagate/QDRI_Project/Shapes/stats_base_counties.shp'
folderout = '/media/denis/seagate/QDRI_Project/Stats/'

#Use single folder and nameout OR use a for loop
#'''
folder = '/media/denis/seagate/QDRI_Project/MODIS/VIs/lswi/filtrada_1/anomalies/'
nameout = 'lswi_anom_mean_2006_2009'

### RUNS THE DAMN FUNCTION!
xuleta2.statsaga(start,stop,shape,folder,folderout,nameout,COUNT,MIN,MAX,RANGE,SUM,MEAN,VAR,STDDEV,QUANTILE)
#'''

'''
folders = [#'/media/denis/seagate/QDRI_Project/MODIS/VIs/ndvi/filtrada_1/anomalies/',
            '/media/denis/seagate/QDRI_Project/MODIS/VIs/ndvi/filtrada_1/',
            '/media/denis/seagate/QDRI_Project/MODIS/VIs/lswi/filtrada_1/anomalies/',
            '/media/denis/seagate/QDRI_Project/MODIS/VIs/lswi/filtrada_1/']
namesout = [#'NDVI_anom_mean_2006_2009',
            'NDVI_filt1_mean_2006_2009',
            'LSWI_anom_mean_2006_2009','LSWI_filt1_mean_2006_2009']

for folder,nameout in zip(folders,namesout):
    xuleta2.statsaga(start,stop,shape,folder,folderout,nameout,COUNT,MIN,MAX,RANGE,SUM,MEAN,VAR,STDDEV,QUANTILE)
    
'''
endall = (time.time() - startall)/60
print 'The whole (ALL) the processes took %.2f minutes' %endall 