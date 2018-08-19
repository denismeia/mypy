### paths
path = "/media/denis/seagate/ESI_2017/denis_conus_esi_12wk/"
pathout = "/media/denis/seagate/ESI_2017/esi_12wk/"

# image parameters
cols = 1456
rows = 625 
x_min = -125.0 #Westernmost point
pixel_size = 0.04 #in degrees, because we are using WGS84
y_min = 24.8 #southernmost point
y_max = y_min + rows*pixel_size # this will calculate northernmost point


import numpy, glob, os
from osgeo import osr, gdal
import xuleta


#we will create GeoTiffs
driver = gdal.GetDriverByName('GTiff') 

#the loop runs for a list of subdirectories in a root directory
# you can easily adapt this loop to match your files
for i in os.listdir(path):

	#now, looping through the files in each subdirectory
	#files have extension .dat
    for j in glob.glob(path+i+"/*.dat"): 

    	#setting the output name based on the input name (I want just the dates)
        nameout = j.split("/")[-1][-11:-4]

        #read the file and reshape it as a matrix
        a = numpy.fromfile(j,dtype='float32').reshape([rows,cols])
        
        #create an empty tif 
        dst_ds = driver.Create(pathout+nameout+".tif", cols, rows, 1, gdal.GDT_Float32)
        
        #create the mesh to shape the tif
        dst_ds.SetGeoTransform([x_min, pixel_size,0,y_max,0,-pixel_size])  
        
		# setting Spatial Rererence
        srs = osr.SpatialReference()                
        srs.ImportFromEPSG(4326)                     
        dst_ds.SetProjection(srs.ExportToWkt()) 

        #writing the matrix to the tif and seting the no-data value
        dst_ds.GetRasterBand(1).WriteArray(a)
        dst_ds.GetRasterBand(1).SetNoDataValue(-9999)
        dst_ds.FlushCache()
        dst_ds = None
        
#converting date format
xuleta.renamedate(pathout,oldf="%Y%j",newf="%Y-%m-%d")