# -*- coding: utf-8 -*-
"""
    THIS IS THE STABLE (lol) VERSION
    PYTHON 3, please!

    

    f(x)s  :
    		rastastats
    		ws_anomalies
            fenologia
    		partial_corr
            dwlchirps
            weightaux
            weightaverage
            piv_table
            mk_test
            shemale
            renamedate
            reindexing
            wavg
            zscore
            statsaga (requires SAGA)
            anomalies
            cumspan
            dnorm
            filtra
            dbf2csv
            listfiles
            findreplace
            ArrayToTif
            TifToArray
            hdftotif
            netcdftotif
            rotate


    @author: Denis Araujo Mariano
    @email: denismeia@icloud.com

"""
def rastastats(files,shp,column,stat,dateformat="%Y-%j"):
    '''
        Created by Denis Mariano - denismeia@icloud.com
        
        This function calculates zonal statistics for polygons and 
            creates a dataframe indexed as a time-series.
        
        folder (str): images folder
        shp (str): shapefile address
        column (str): column with the polygons names in the shapefile
        stat (str): 'mean','median'...
            or with nodata specified : 'nmean', 'nmedian' or 'ncount'
                and specify the nodata value.
        dateformat (str): the images filename date format

        ... I still have to find a smart way to pass nodata value to the f(x)
    '''    

    import pandas as pd
    import time
    from glob import glob
    from rasterstats import zonal_stats


    tic = time.time()
    
    #Because geopandas is a pain in the ass to install, I found a workaround using fiona
    # therefore, you gotta have one or another
    try:
        import geopandas as gpd
        print("Using geopandas " + gpd.__version__)
        #get the polygons names
        shape = gpd.read_file(shp)
        names=list(shape[column])
            
    except ImportError:
        import fiona
        print("Using Fiona " + fiona.__version__)

        #Complicated workaround to get column records without geopandas
        def records(shp, column):
           reader = fiona.open(shp)
           for feat in reader:
              new  = {}
              new['properties'] = {}
              new['properties'][column] = feat['properties'][column]
              yield new
        names = []
        for feat in records(shp,column):
            names.append(feat['properties'][column])
    
    df = pd.DataFrame() #create empty dataframe
    
    # folder or list?
    if type(files)==str:
        files = sorted(glob(fimages+"*.tif"))
        print('There are %d files in the folder \n Hopefully all the names are cool' %len(files))       
    else:
        print('using your specified list of files - total %d files' %len(files))
    
    for i,n in enumerate(files):  
        stats = zonal_stats(shp,files[i],stats=stat)
        
        date = n.split('/')[-1][:-4]
        print('calculating stats for date %s' %date)

        df[date] = [stats[x][stat] for x in range(5)]
    
    #organizing the Dataframe
    df.index = names
    df = df.T
    try:
        df['date'] = pd.to_datetime(df.index,format=dateformat)
    except:
        df['date'] = pd.to_datetime(df.index,format="%Y-%m-%d")
        print('Date format converted. \n')
        
    df.index = df['date']
    df.drop(labels='date',axis=1,inplace=True)
    toc = time.time()
    print("total execution time was %.0f seconds" %(toc-tic))
    return df


def ws_anomalies(folder,nodata,ws=5):
    '''
        !!!For image time-series!!!
        ws_anomalies is a function to calculate anomalies based
            on a accumulation window (like 5 periods) moving
            ~weekly. In short, MODIS data anomalies based on
            1 month averages.

        folder: string. Input files folder with "/" in the end
        nodata: float,int. the nodata value to be considered
        ws: integer. window summation. Is the accumulation period, it is
            actually ws-1.

        The output will be created in the "anomalies" folder in the
            input folder
    '''
    if ws%2 == 0:
        print('must be an odd number, we are setting 5 as default')
        ws = 5

    try:
        xu.renamedate(folder,oldf='%Y-%m-%d',newf='%Y-%j',wts=0)
        print('Filenames converted to DOY format')
    except:
        print('Maybe the files are already in DOY format')

    if ws%2 == 0:
        print('must be an odd number')

    #rename files to DOY format
    #xu.renamedate(folder,oldf='%Y-%m-%d',newf='%Y-%j',wts=0)

    # create the output folder for means, std and final result
    fout = folder + 'anomalies/'
    fmeans = fout + 'means'
    fstds = fout + 'stds'
    fs = [fout, fmeans, fstds]
    for i in fs:
        if not os.path.exists(i): os.makedirs(i)

    #LIST OF FILES and list of names
    lf = sorted(glob(folder+"*.tif"))
    ln = xu.listfiles(folder,full=False)
    print('Total of %d files, ready to start?' %len(lf))

    #List of unique dates
    days = sorted(list(set([ii[-7:-4] for n,ii in enumerate(ln) if ii not in ln[:n]])))

    # create a list of extended dates (like a cycle) considering the chosen window
    daysext = days[-int(ws/2):] + days + days[:int(ws/2)]

    for i in range(int(ws/2),len(daysext)-int(ws/2)):
        cumdates = daysext[i-int(ws/2):i+int(ws/2)]
        print('processing '+ daysext[i])
        print(cumdates)
        toOpen = [x for x in lf if x[-7:-4] in cumdates]
        arrays = []
        for j in toOpen:
            image,m = xu.TifToArray(j)
            arrays.append(image)
        arrays = np.asarray(arrays)
        arrays2 = np.reshape(arrays,arrays.shape)
        arrays2 = np.where(arrays2>nodata,np.nan,arrays2)

        mean = np.nanmean(arrays2, axis=0)
        std = np.nanstd(arrays2, axis=0)

        xu.ArrayToTif(mean,daysext[i]+".tif",Folder=fmeans,Metadata=m,Type=3)
        xu.ArrayToTif(std,daysext[i]+".tif",Folder=fstds,Metadata=m,Type=3)

        lprocess = [x for x in lf if x[-7:-4]==daysext[i]]
        for k in lprocess:
            proc, m = xu.TifToArray(k)
            proc = np.where(proc==nodata,np.nan,proc)
            anom = (proc-mean)/std
            xu.ArrayToTif(anom,k.split("/")[-1],fout,m,Type=3)
            proc,anom = None,None

        arrays, arrays2, mean, std, m, lprocess, = None,None,None,None,None,None


def fenologia(var,b=12+2,a=3+1):
    '''
        Dirty phenology:
            The function detects the peaks for each season based on
        GPP or NDVI (or other ok VI). You'd better to know beforehand
        what you're analyzing so set the n of weeks before and after
        the peak to constrain your analysis.

        var: dataframe with the data, preferred GPP or NDVI
        parameter b, a, lb and la are given in weeks.
        b = weeks before the peak, usually 12 for soybean
            +2 for margin
        a = weeks after the peak, usually 3 for soybean
            +1 for margin

        -------
        Example:
        xu.fenologia(xu.weightaux(gpp,aux,criterion,43011))
    '''
    import pandas as pd
    import numpy as np

    peak = [];peakd = []
    for y in np.unique(var.index.year):
        peakd.append(np.argmax(var.loc[var.index.year==y]))
        peak.append(np.max(var.loc[var.index.year==y]))

    df2 = pd.DataFrame(data={'date':peakd,'value': peak},index=pd.DatetimeIndex(peakd))
    df2['week'] = df2.index.weekofyear

    p = df2.week.median()
    sos = p - b
    eos = p + a
    if sos < 0:
        sos = 52 + sos
    if eos > 52:
        eos = abs(52 - eos)
    print(sos,p,eos)
    return sos,p,eos




# PARTIAL CORRELATION ANALYSIS
import numpy as np
from scipy import stats, linalg

def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """

    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr


def dwlchirps():
    import os
    '''
        All based on raw inputs! kinda dumb!
    '''


    dwlf = str(input('type download folder:  '))
    if dwlf[-1] != '/':
        dwlf = dwlf + '/'
    frequency = str(input('type m for monthly or d for decad:  '))
    if frequency == 'm':
        freq = 'global_2-monthly_EWX'
    else:
        freq = 'global_dekad_EWX'

    tipo = str(input('type: z, a or p for z-scores, anomalies or precipitation data:  '))
    if tipo == 'z':
        base = 'ftp://ftp.chg.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/'+freq+'/zscore/zscore.'
    elif tipo == 'a':
        base = 'ftp://ftp.chg.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/'+freq+'/anomaly/anom.'
    else:
        base = 'ftp://ftp.chg.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/'+freq+'/data/data.'

    yeari = int(input("First year:  "))
    yearf = int(input("Last year:  "))
    years = range(yeari,yearf+1)
    os.chdir(dwlf)
    for i in years:
        os.system('wget '+base+str(i)+'*.*')
        print('downloading '+ str(i))


#Weighted average from region in aux table
def weightaux(var,aux,criterion,group,verbose=False):
    '''
        var = dataframe
        aux = dataframe with auxiliar data!
        criterion = string for the column in the aux table to be used
        group = string with the value in the column OR a list of strings

    '''
    import os
    import pandas as pd
    pd.options.mode.chained_assignment = None #error omit

    #convert whatever comes into a list
    if type(group)!=list:
        group = [group]

    lista = sorted(list(aux.geocodig_m[aux[criterion].isin(group)]))

    if lista == []:
        group =  list(map(float, group))
        lista = sorted(list(aux.geocodig_m[aux[criterion].isin(group)]))

    #cross-check desired municipalities with those available in the dataset
    #l2 = sorted(list(map(float, aw.columns))) #to make sure they are all float
    l3 = sorted(list(set(lista).intersection(var.columns))) #the effective list

    #get the weights
    a = aux[aux['geocodig_m'].isin(l3)]
    a['w'] = a.area/sum(a.area)
    a = a[['geocodig_m','w']]
    a.T.to_csv('Xaux.csv',header=False,index=None)
    aw = pd.read_csv('Xaux.csv')
    os.remove('Xaux.csv')
    aw = aw.sort_index(axis=1) #df with weight per municipalitie

    #weight values and get the weighted average
    var_aw = pd.DataFrame(var[l3].values*aw.values, columns=aw.columns,index=var[l3].index)
    var_final = var_aw.sum(axis=1,skipna=True)
    var_final.index = pd.DatetimeIndex(var_final.index)
    #var_final.columns = ['values']
    if verbose == True:
        print('Total of %d municipalities averaged' %len(l3))
        print(aw)
    return var_final



def weightaverage(var,aux,lista):
    import pandas as pd
    import os
    pd.options.mode.chained_assignment = None #error omit

    '''
        var: dataframe
        aux = dataframe with auxiliar data!
        lista: group of columns
        This is very specific for the SQL output tables that I get
            when I extract time-series from raster data by polygons (municipalities).
        So, we need a "aux" dataframe containing the characteristics of the polygons and areas.
    '''
    if type(lista[0]) != str:
        lista =  list(map(str, lista))

    a = aux[aux['geocodig_m'].isin(lista)]

    if a.shape[0] == 0:
        lista =  list(map(float, lista))
        a = aux[aux['geocodig_m'].isin(lista)]

    a['w'] = a.area/sum(a.area)
    a = a[['geocodig_m','w']]
    a.T.to_csv('Xaux.csv',header=False,index=None)
    aw = pd.read_csv('Xaux.csv')
    os.remove('Xaux.csv')
    aw = aw.sort_index(axis=1)
    l2 =  list(map(float, aw.columns))
    return var_final


def piv_table(var,aux,lista,criterion=None):

    '''
        Create a pivot table based on year and month. The output
        has months as columns and years as rows. You can easily transpose
        to have it the other way around.

        This works with the weightaverage OR weightaux function.
        Criterion is set to None by defaut, so, you have to formally
            enter a list of municipalities, weightaverage function will be executed.
        If e.g criterion='microrregi', your lista has to match criterion requirements,
            then, weightaux will be executed


        var: dataframe with a DatetimeIndex
        aux: dataframe with auxiliar data
        if criterion is set to None (default),
            lista: municipalities list
        if criterion is applied (string with the column name)
            lista: elements in the criterion column
    '''
    import pandas
    import xuleta as xu
    month_names = pandas.date_range(start='2016-01-01', periods=12, freq='MS').strftime('%b')

    if type(criterion) == str:
        group = lista
        region_var = xu.weightaux(var,aux,criterion,group)
    else:
        region_var = xu.weightaverage(var,aux,lista)

    region_var = pandas.DataFrame(region_var)
    region_var.columns = ['values']
    region_var = region_var.resample('M').mean()

    region_var['year'] = region_var.index.year
    region_var['month'] = region_var.index.month
    region_var_piv = region_var.pivot(index='year', columns='month', values='values')
    region_var_piv.columns = month_names
    return region_var_piv


#Mann-Kendall test
def mk_test(x, alpha = 0.05):
    import numpy as np
    from scipy.stats import norm, mstats
    import statsmodels.api as sm
    import seaborn as sns
    """
    Input:
        x:   a vector of data
        alpha: significance level (0.05 default)

    Output:
        trend: tells the trend (increasing, decreasing or no trend)
        h: True (if trend is present) or False (if trend is absence)
        p: p value of the significance test
        z: normalized test statistics

    Examples
    --------
      >>> x = np.random.rand(100)
      >>> trend,h,p,z = mk_test(x,0.05)
    """
    n = len(x)

    # calculate S
    s = 0
    for k in range(n-1):
        for j in range(k+1,n):
            s += np.sign(x[j] - x[k])

    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)

    # calculate the var(s)
    if n == g: # there is no tie
        var_s = (n*(n-1)*(2*n+5))/18
    else: # there are some ties in data
        tp = np.zeros(unique_x.shape)
        for i in range(len(unique_x)):
            tp[i] = sum(unique_x[i] == x)
        var_s = (n*(n-1)*(2*n+5) + np.sum(tp*(tp-1)*(2*tp+5)))/18

    if s>0:
        z = (s - 1)/np.sqrt(var_s)
    elif s == 0:
            z = 0
    elif s<0:
        z = (s + 1)/np.sqrt(var_s)

    # calculate the p_value
    p = 2*(1-norm.cdf(abs(z))) # two tail test
    h = abs(z) > norm.ppf(1-alpha/2)

    if (z<0) and h:
        trend = 'decreasing'
    elif (z>0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'
    l = [trend,h,round(p, 3),round(z,3)]

    return l#trend, h, p, z

def mailme(taime=0):
    '''
        This function just sends me and email when I run it.
        It is good to run after a long process finishes, so I can
            come back and check the result once I'm notified by email.
    '''
    import smtplib
    to = 'denismeia@icloud.com'
    gmail_user = 'denismeia@gmail.com'
    conhao="/home/denis/pcloud/DeniStuff/conho.txt"
    with open(conhao, 'r') as myfile:
        senha=myfile.read().replace('\n', '')
    gmail_pwd = senha
    smtpserver = smtplib.SMTP("smtp.gmail.com",587)
    smtpserver.ehlo()
    smtpserver.starttls()
    smtpserver.ehlo
    smtpserver.login(gmail_user, gmail_pwd)
    header = 'To:' + to + '\n' + 'From: ' + gmail_user + '\n' + 'Subject:PROCESS DONE \n'
    print(header)

    if taime!=0:
        msg = header + '\n Volte ao trabalho, Denis.  \n\n' + 'O Processo levou %.3f minutos.' %(taime)
    else:
        msg = header + '\n Volte ao trabalho, Denis.  \n\n' + 'O processamento acabou.'

    smtpserver.sendmail(gmail_user, to, msg)
    print('done!')
    smtpserver.close()


def renamedate(folder,oldf,newf="%Y-%m-%d", wts=0):
    '''
        folder: files directory
        oldf: it has to be writen in datetime convention, for example:
                "%Y-%j", "%Y%j", or "%Y-%m-%d"
        newf: same thing as oldf, as default the final date format is
                "%Y-%m-%d"
        wts: stands for "where to start?" If the date information is not in
            the end of the filename, you have to tell me where to look at.
            Example: MYD09Q1.A2002297.h10v04.006.2015151154845.ndvi.tif,
            the date infor starts on character 10, with the format already
                        specified with oldf. The 10th character is the number 2. If no
            value is assigned to wts, we will assume the data in the end of the
            file name.
    '''
    if folder[-1] != "/":
        folder = folder + "/"
    import os, xuleta, datetime
    if oldf == "%Y_%j":
        n = 8
    elif oldf == "%Y-%j":
        n = 8
    elif oldf == "%Y%j":
        n = 7
    else:
        n = 10
    lista = xuleta.listfiles(folder,full=False)
    if wts == 0:
        for i in lista:
            os.rename(folder + i,folder + str(datetime.datetime.strptime(i[-abs(4+n):-4],oldf).strftime(newf)) + ".tif")
    else:
        wts=wts-1
        for i in lista:
            os.rename(folder + i,folder + str(datetime.datetime.strptime(i[wts:wts+n],oldf).strftime(newf)) + ".tif")
    print ("Done!")


def reindexing(var,fmt = "%Y%j"):
    '''
        var = dataframe containing the column 'date' with YYYYDOY
        stdate: (str) 'M/D/YYYY'
        years = (int) number of years
    '''
    import pandas as pd
    var.index = pd.to_datetime(var.index,format=fmt)
    date_index = pd.date_range(var.index[0], var.index[-1] , freq='D')
    var = var.reindex(date_index)
    var = var.interpolate(method='linear')
    if 'date' in var.columns:
        var = var.drop('date',axis=1)
    return var


def wavg(meta,var,mode,criterion,pfp=True):
    '''

        developer: Denis Mariano
        email: denismeia@gmail.com;mariano@huskers.unl.edu
        last modified: June, 8, 2016

        To run this function is necessary to have loaded the metadata as meta
        and the required variables. If you didn't do it, copy and paste something
        like the example below:

            base = 'data/QDRI_v3.h5'
            variables = ['gpp','lswi','ndwi','lstn','nrvi','lstd','ndvi','ctvi',
            'rvi','ttvi','lai','fpar','evi','prec','esi']
            meta =  pd.read_hdf(base,"counties/metadata")
            for var in variables[:-1]: #I'm excluding esi from the raw values
                globals()['z{}'.format(var)] = pd.read_hdf(base,'counties/zscore/'+var)
                print 'z{}'.format(var)
            for var in variables:
                globals()['r{}'.format(var)] = pd.read_hdf(base,'counties/values/'+var)
                print 'r{}'.format(var)

        INPUT:
        meta: dataframe containing metadata
        var: dataframe (e.g.: zndvi, rlai)
        mode: 'climate','climate_adm','state','lista'
        criterion: string_for 'climate', the climate region,
                   string_for 'climate_adm', the climate_adm region,
                   string_for 'state', the state acronym
                   list_ of counties for 'lista' mode
        pfp: prepare for plot, if True, convert the index to datetime
             if False, no datetime, which sucks so badly!

        RETURN:
            a tiny dataframe containing index (YYYYDOY) and the weighted average,
            the column name is called 'average'
    '''
    import pandas,os
    meta=meta #maybe change the first meta to another name and replace over the function
              #because this shit is confusing
    #omit the zuado errors of pandas (the error is mine, actually)
    pandas.options.mode.chained_assignment = None
    var2 = var

    if mode == 'climate':
        a = meta.loc[meta['climate'] == criterion]

    elif mode=='climate_adm':
        a = meta.loc[meta['climate_adm'] == criterion]
        a['w'] = a.prec/a.prec.sum()

    elif mode=='state':
        criterion = '_'+criterion+'_'
        lista = sorted([county for county in meta.county if criterion in county])
        a = meta[meta['county'].isin(lista)]

    else:
        criterion = sorted(criterion)
        a = meta[meta['county'].isin(criterion)]

    #and now, the calculation
    a['w'] = a.prec/a.prec.sum()
    a = a.drop(['climate','climate_adm','prec','esi','modis'],axis=1)



    a.T.to_csv('Xaux.csv',header=False,index=None)
    aw = pandas.read_csv('Xaux.csv')
    os.remove('Xaux.csv')
    aw = aw.sort_index(axis=1)
    var2 = var2[aw.columns]
    var2 = var2.sort_index(axis=1)
    var2 = pandas.DataFrame(var2.values*aw.values, columns=aw.columns,
        index=var.index)
    var2['average']=var2.sum(axis=1,skipna=True)
    if pfp == False:
        return var2['average']
    else:
        var2.index = pandas.to_datetime(var.index,format="%Y%j")
        return var2['average']



def statsaga(folder,shapein,allowance=0.8):
    '''
        developer: Denis Mariano
            email: denismeia@icloud.com
        last modified: May, 25, 2016
		
		## THIS IS REALLY DEPRECATED!!! not using it, 
			I'm just keeping it for future reference"

        This script uses saga_cmd to compute raster stats for polygons.
        Currently, I'm calculating only the MEAN, however, I might make it more
            flexible.
        The output CSV files and the final CSV file will be created in the same
        folder as the shape input.

        Parameters:
        ----------
        folder: string with / in the end
            path to the raster files
        shapein: string
            path to the shapefile
        allowance: float
            how much (from > 0 to <1 ) of the free memory will be used in the process. Low allowance will generate more output shp files, so, you'll have more work to concatenate them.
    '''

    import xuleta, os, psutil
    import pandas as pd

    l = xuleta.listfiles(folder,full=True)
    size = 0
    for i in l:
        size = size + os.path.getsize(l[1])/1e6 #em GB

    allowed = (psutil.virtual_memory()[1]/1e6)/size
    interval = int(len(l)*allowed*.2*allowance)

    n=(len(l)//interval)+1 # // because o python3

    if n > 1:
        for i in range(n):
            s = ''
            shapeout = shapein[:-4]+str(i)+'.shp'
            for j in reversed(l[i*interval:(i+1)*interval]):
                s=j+';'+s
            s = s[:-1]
            os.system('saga_cmd shapes_grid 2 -GRIDS=\'%s\' -POLYGONS=%s -METHOD=0 -NAMING=1 -RESULT=%s -COUNT=0 -MIN=0 -MAX=0 -RANGE=0 -SUM=0 -MEAN=1 -VAR=0 -STDDEV=0 -QUANTILE=0' % (s,shapein,shapeout))
            print('saga_cmd shapes_grid 2 -GRIDS=\'%s\' -POLYGONS=%s -METHOD=0 -NAMING=1 -RESULT=%s -COUNT=0 -MIN=0 -MAX=0 -RANGE=0 -SUM=0 -MEAN=1 -VAR=0 -STDDEV=0 -QUANTILE=0' % (s,shapein,shapeout))
            print('\n')

            try:
                xuleta.dbf2csv(shapeout[:-4] + ".dbf")
                os.remove(shapeout[:-4] + ".dbf")
                os.remove(shapeout[:-4] + ".shx")
                os.remove(shapeout[:-4] + ".shp")
                os.remove(shapeout[:-4] + ".mshp")
                #os.remove(shapeout[:-4] + ".prj")

            except:
                print('could not find the files')

        # preparing dataframe final as a CSV file
        l2 = xuleta.listfiles(shapein[:-6],extension='*.csv',full=True)
        e = pd.read_csv(l2[0])
        for i,j in zip(list(range(n-1)),l2[1:]):
            globals()['e{}'.format(i)] = pd.read_csv(j)
            e = pd.merge(e,globals()['e{}'.format(i)])
        e=e.T
        e.to_csv(shapein[:-4]+'_final.csv',header=False)
    else:
        s = ''
        shapeout = shapein[:-4]+str(0)+'.shp'
        for j in reversed(l):
            s=j+';'+s
        s = s[:-1]
        os.system('saga_cmd shapes_grid 2 -GRIDS=\'%s\' -POLYGONS=%s -METHOD=0 -NAMING=1 -RESULT=%s -COUNT=0 -MIN=0 -MAX=0 -RANGE=0 -SUM=0 -MEAN=1 -VAR=0 -STDDEV=0 -QUANTILE=0' % (s,shapein,shapeout))
        print('saga_cmd shapes_grid 2 -GRIDS=\'%s\' -POLYGONS=%s -METHOD=0 -NAMING=1 -RESULT=%s -COUNT=0 -MIN=0 -MAX=0 -RANGE=0 -SUM=0 -MEAN=1 -VAR=0 -STDDEV=0 -QUANTILE=0' % (s,shapein,shapeout))
        print('\n')

        xuleta.dbf2csv(shapeout[:-4] + ".dbf")
        e = pd.read_csv(shapein[:-4] + "0.csv")
        e = e.T
        e.to_csv(shapein[:-4]+'_final.csv',header=False)
        try:
            os.remove(shapein[:-4] + "0.csv")
            os.remove(shapeout[:-4] + ".dbf")
            os.remove(shapeout[:-4] + ".shx")
            os.remove(shapeout[:-4] + ".shp")
            os.remove(shapeout[:-4] + ".mshp")
            #os.remove(shapeout[:-4] + ".prj")

        except:
            print('could not find the files')


def zscore(df,index=False,datecolumn='acquisition'):

    import pandas as pd
    import numpy as np

    if index == False:
        df.index = pd.DatetimeIndex(df[datecolumn])
        df = df.drop(datecolumn,axis=1)
    else:
        df.index = pd.DatetimeIndex(df.index)

    # CORE da function
    mean=pd.groupby(df,by=[df.index.dayofyear]).aggregate(np.nanmean)
    std= pd.groupby(df,by=[df.index.dayofyear]).aggregate(np.nanstd)

    df2 = df.copy()
    for y in np.unique(df.index.year):
        for d in np.unique(df.index.dayofyear):
            df2[(df.index.year==y) & (df.index.dayofyear==d)] = (df[(df.index.year==y) & (df.index.dayofyear==d)]- mean.ix[d])/std.ix[d]
            df2.index.name = 'date'

    return df2


def anomalies(folder,di=4,df=7,deltemp=False):
    '''
        Specify the input folder as a string ended with a backslash "/"
        The deltemp=False will keep the mean and std folders for later use,
            otherwise, if True, the data will be deleted

    '''
    import os, shutil, xuleta
    import numpy as np

    #input and output folder
    folderout = folder+'anomalies/'
    foldermean = folderout+'mean/'
    folderstd = folderout+'std/'
    if not os.path.exists(folderout): os.makedirs(folderout)
    if not os.path.exists(foldermean): os.makedirs(foldermean)
    if not os.path.exists(folderstd): os.makedirs(folderstd)


    files = xuleta.listfiles(folder)
    days = []
    for i in files:
        days.append(i[di:df])
        days.sort()
    days = [ii for n,ii in enumerate(days) if ii not in days[:n]]

    #Calculates means and standard deviations for each day in a series
    for d in days:
        toOpen = []
        for i in files:
            if i[di:df]==d:
                toOpen.append(i)
        arrays = []
        for j in toOpen:
            image, meta = xuleta.TifToArray(folder+j)
            #something to avoid no-data

            #image = np.where(image>thu,np.nan,image)
            #image = np.where(image<thl,np.nan,image)
            arrays.append(image)
            arrays2 = np.dstack(arrays)

        print('calculating mean and standard deviation for the day ' +d)
        xuleta.ArrayToTif(np.nanmean(arrays2,axis=2),d,foldermean, meta, Type=3)
        xuleta.ArrayToTif(np.nanstd(arrays2,axis=2),d,folderstd, meta, Type=3)
        toOpen, arrays, arrays2, image = None,None,None,None

    # calculates the anomalies based on the mean and stdev
    lmean = xuleta.listfiles(foldermean)
    lstd = xuleta.listfiles(folderstd)
    for i in files:
        for mean_,std_ in zip(lmean,lstd):
            if i[di:df] == mean_[:3]:
                image, meta = xuleta.TifToArray(folder+i)
                mean, meta = xuleta.TifToArray(foldermean+mean_)
                std, meta = xuleta.TifToArray(folderstd+std_)
                #image[image > 3000] = 1
                anomaly = (image - mean)/std
                anomaly = np.where(anomaly==np.nan,0.00001,anomaly)
                anomaly = np.where(anomaly>3,3,anomaly)
                anomaly = np.where(anomaly<-3,-3,anomaly)
                anomaly = np.where(anomaly==0,0.00001,anomaly)
                #anomaly = np.where(anomaly == 0, anomaly+0.00001, anomaly)
                #anomaly = np.where(anomaly == np.nan, 0.00001, anomaly)
                print('Calculating z-score for '+i)
                xuleta.ArrayToTif(anomaly, i[:-3], folderout, meta, Type=3)
            image, mean, std, meta = None, None, None, None
    lmean,lstd = None, None

    #getting rid of the temporary data
    if deltemp==True:
        print('Say TCHAU! to your temporary data!')
        if os.path.exists(foldermean): shutil.rmtree(foldermean)
        if os.path.exists(folderstd): shutil.rmtree(folderstd)


def cumspan(quantidade,anoinicial,anofinal,diainicial,diafinal,folderin,folderout):
    '''
		The f(x) needs a better name
		and proper docstring, although this might be deprecated.
    '''

    for ano in range(anoinicial,anofinal+1):
        import os
        import xuleta
        if not os.path.exists(folderout): os.makedirs(folderout)
        dia = diainicial
        #print ano
        while dia <= diafinal:
            soma = 0
            print(dia)
            for j in range(quantidade):
                #print j
                try:
                    a,m = xuleta.TifToArray('{0}{1}{2:0>3d}.tif'.format(folderin,ano,dia+j))
                    #print '/media/denis/seagate/PRECIPITATION/PERSIANN/daily_r/{0}{1:0>3d}.tif'.format(ano,dia+j)
                    soma += a
                except:
                    print('Erro: {0}{1:0>3d}.tif'.format(ano,dia+j))
                    pass
            if type(soma) != int:
                xuleta.ArrayToTif(soma, '{0}{1:0>3d}'.format(ano,dia), folderout, m)
                print('Salvo: {0}{1:0>3d}.tif'.format(ano,dia))
            dia += quantidade




def dnorm(v):
    '''
        Returns the normalized data of an array 'v'.
        The normalization is scaled from 0 to 1
        v = array
    '''
    import numpy as np
    v = np.asarray(v, dtype=float)
    nor = (v-np.nanmin(v))/(np.nanmax(v)-np.nanmin(v))
    return nor


def dbf2csv(filename):
    '''
        tem que ter a package dbf (sudo pip install dbf)
        entre com o caminho+nome da tabela, a saida sera escrita
        na mesma pasta com o .csv
    '''
    import dbf, os
    a = dbf.Table(filename).open()
    output = filename[:-4] + '.csv'
    dbf.export(a,filename=output, encoding='utf-8')
    f1 = open(output, 'r')
    f2 = open(output[:-4] + '_.csv', 'w')
    for line in f1:
        f2.write(line.replace(' ', ''))
    f1.close()
    f2.close()
    os.remove(output)
    os.rename(output[:-4]+'_.csv',output)
    del a


def findreplace(folder,find,replace):
    '''
        Renomeia arquivos na pasta atual, entre com as strings
        em find e replace.

        Cuidado, isso pode foder com seus arquivos se utilizado de forma errada.
    '''
    import os
    d = os.getcwd()
    os.chdir(folder)
    [os.rename(f, f.replace(find, replace)) for f in os.listdir('.') if not f.startswith('.')]
    os.chdir(d)

def listfiles(folder,extension='*.tif',full=False):
    '''
        DAH PRA MELHORAR. muda esse *.tif para tif
        FAZ ALGUMA COISA pra n ter que botar a / no final do folder
        ALGUM IF, SEI LAH

        Lista os arquivos de extensÃ£o escolhida para a pasta escolhida.
        A lista Ã© ordenada e os os valores repetidos sÃ£o eliminados.
        ExtensÃ£o default = '*.tif'
        full=False, se for True, dÃ¡ o endereÃ§o completo.

    '''

    listname = []
    import glob
    for j in glob.glob(folder + extension):
        if full==True:
            listname.append(j)
        else:
            b = j.split('/')
            c = b[-1]
            listname.append(c)
    listname.sort()
    listname = [ii for n,ii in enumerate(listname) if ii not in listname[:n]]
    return listname

def ArrayToTif(Array, Filename, Folder, Metadata, Type = 1):
    '''
    Converts a array like to .tif file.
    Array: Single or Multiband array
    Folder: Destination folder of files. Ex. /tmp/
    Filename: Name of output tif. Ex. MyTifFile
    Metadata: Dictionary with keys: GeoTrasform, projection, rows, cols of original
    raster.
    Type: 1 = gdal.GDT_Int16
          2 = gdal.GDT_Int32
          3 = gdal.GDT_Float32
          4 = gdal.GDT_Float64
    '''
    import gdal, os
    driver = gdal.GetDriverByName("GTiff")
    if not os.path.exists(Folder): os.makedirs(Folder)
    geot = Metadata['Geo']
    proj = Metadata['proj']
    rows = Metadata['rows']
    cols = Metadata['cols']
    if Type == 0:
        TypeIF = gdal.GDT_Byte
    if Type == 1:
        TypeIF = gdal.GDT_Int16
    if Type == 2:
        TypeIF = gdal.GDT_Int32
    if Type == 3:
        TypeIF = gdal.GDT_Float32
    if Type == 4:
        TypeIF = gdal.GDT_Float64
    if Array.ndim == 2:
        Raster = driver.Create(Folder + '/' + Filename, cols, rows, 1, TypeIF)
        Raster.GetRasterBand(1).WriteArray(Array)
    else:
        Raster = driver.Create(Folder + '/' + Filename, cols, rows, Array.shape[2], TypeIF, [ '-co COMPRESS=DEFLATE -co PREDICTOR=1 -co ZLEVEL=9 -multi' ])
        for band in range(Array.shape[2]):
            print('Writing band' +str(band+1)+'of '+Filename)
            Raster.GetRasterBand(band+1).WriteArray(Array[:,:,band])
    Raster.SetGeoTransform(geot)
    Raster.SetProjection(proj)
    Raster.FlushCache()
    print('%s saved.\n '%(Filename))
    del geot, proj, rows, cols, Raster
    return

def TifToArray(Raster):
    '''
    This function opens a Raster data as Array to use numpy manipulation
    Return: Array, Metadata
    Array: A array like variable
    Metadata: Metadata of Raster file. Could be needed for a future\n
    ArrayToRaster transformation.
    '''
    import gdal
    openGdal = gdal.Open(Raster)
    geot = openGdal.GetGeoTransform()
    proj = openGdal.GetProjection()
    rows = openGdal.RasterYSize
    cols = openGdal.RasterXSize
    Metadata = {'Geo':geot, 'proj':proj, 'rows':rows, 'cols':cols}
    Array = openGdal.ReadAsArray()
    return Array, Metadata

#HDF to tif
def hdftotif(folder, bands):
    import os, glob
    lista = []
    if not os.path.exists(folder+'Tif'): os.makedirs(folder+'Tif')
    for files in sorted(glob.glob(folder+'*.HDF')):
        print(files)
        filename = files.split('/')[-1]
        filename = filename[:-4]
        print(filename)
        for band in bands:
            print(band)
            cmd = 'gdal_translate -of "GTiff" HDF4_SDS:UNKNOWN:'+folder+filename+'.HDF:'+band+' '+folder+'Tif/'+filename+'_'+band+'.tif'
            print(cmd)
            os.system(cmd)
            lista.append(folder+'Tif/'+filename+'_'+band+'.tif')
    return lista


#NetCDF to tif
def netcdftotif(folder, bands):
    '''
    bands: names of the bands
        e.g. bands = 'soil_moisture_x','soil_moisture_c'
    '''
    import os, glob
    lista = []
    if not os.path.exists(folder+'Tif'): os.makedirs(folder+'Tif')
    for files in sorted(glob.glob(folder+'*.nc')):
        print(files)
        filename = files.split('/')[-1]
        filename = filename[:-3]
        print(filename)
        for band in bands:
            print(band)
            cmd = 'gdal_translate -of "GTiff" netCDF:'+folder+filename+'.nc:'+band+' '+folder+'Tif/'+filename+'_'+band+'.tif'
            print(cmd)
            os.system(cmd)
            lista.append(folder+'Tif/'+filename+'_'+band+'.tif')
    return lista


#sometimes the trmm images come with rotated pole, so, you
#have to rotate them
def rotate(filenameList):
    '''
    rotate LPRM files converted with netcdftotif function
    '''
    import numpy as np
    import shutil
    for tif in filenameList:
        array, metadata = TifToArray(tif)
        metadata['cols'], metadata['rows'] = metadata['rows'],metadata['cols']
        metadata['Geo'] = (-180, 0.25, 0.0, 90.0, 0.0, -0.25)
        array = array.T
        array = np.fliplr(array)
        filename = tif.split('/')[-1]
        filename = filename[:-3]
        path = tif.split('/')[:-1]
        temp = path[:]
        path[-1] = 'TifLPRM'
        path = '/'.join(path)
        ArrayToTif(array, filename+'_transposed', path +'/', metadata, Type = 1)
    temp = '/'.join(temp) +'/'
    shutil.rmtree(temp+'/')


def clipraster(folderin, shapefile, folderout, format_end=''):
    '''
        folderin = pasta de arquivos (sem a barra no final)
        shapefile = caminho+arquivo.shp
        folderout = pasta de destino
        format_end= ' ' tipo de arquivo 'tif'
    '''


    from osgeo import gdal, ogr
    import sys
    import subprocess as sp
    import os

    files = [os.path.join(root, name)
               for root, dirs, files in os.walk(folderin)
                 for name in files
                 if name.endswith(format_end)]

    daShapefile = shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(daShapefile, 0)
    layer = dataSource.GetLayer()
    ex1,ex2,ex3,ex4 = layer.GetExtent()

    for j in files:
        out = j
        path = folderout+out
        paramsnorm = ["gdal_translate", "-projwin", str(ex1), str(ex4), str(ex2), str(ex3), j, path]
        print((sp.list2cmdline(paramsnorm)))


def filtra(folder,windowsize,upper,lower,Type=3):
    '''
        FALTA O DOCSTRING PORRA!
    '''
    import glob
    import numpy as np
    files = sorted(glob.glob(folder+'*.tif'))
    for i in range(len(files[windowsize:-windowsize])):
        toOpen = []
        filtrar = files[windowsize:-windowsize][i]
        toOpen.append(filtrar)
        for j in range(windowsize):
            print('Processing image ' + str(files[i]).split('/')[-1])
            print('Image ' + str(i+1) + ' of ' + str(len(files[windowsize:-windowsize])))
            toOpen.append(files[j+i])
            toOpen.append(files[windowsize+j+1+i])
        toOpen.sort()
        arrays = []
        for i in toOpen:
            nfiltrado, metadata = TifToArray(i)
            arrays.append(nfiltrado)
            arrays2 = np.dstack(arrays)
        np.seterr(all = 'ignore')

        #Here you can customize your filter

        arrays2[arrays2 > upper] = np.nan
        arrays2[arrays2 < lower] = np.nan

        media = np.nanmean(arrays2, axis = 2)
        median = np.nanmedian(arrays2, axis = 2)
        filtrada = np.where(media < 0.55*median, median, media)


        #Here you write the output
        ArrayToTif(filtrada, (filtrar.split('/')[-1]) ,folder + 'filt_' + str(windowsize) + '/', metadata,Type)#3)
        media, median, filtrada, arrays, arrays2 = None, None, None, None, None
    return



