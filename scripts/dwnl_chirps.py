import os

#global Z-score monthly
#ftp://ftp.chg.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/global_monthly_EWX/zscore/


dwlf = str(raw_input('type download folder:  '))
if dwlf[-1] != '/':
	dwlf = dwlf + '/'
frequency = str(raw_input('type m for monthly or d for decad:  '))
if frequency == 'm':
	freq = 'global_2-monthly_EWX'
else:
	freq = 'global_dekad_EWX'

tipo = str(raw_input('type: z, a or p for z-scores, anomalies or precipitation data:  '))
if tipo == 'z':
	base = 'ftp://ftp.chg.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/'+freq+'/zscore/zscore.'
elif tipo == 'a':
	base = 'ftp://ftp.chg.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/'+freq+'/anomaly/anom.'
else:
	#base = 'ftp://ftp.chg.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/'+freq+'/data/data.'
    base = 'ftp://ftp.chg.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/global_monthly/tifs/chirps-v2.0.'

yeari = int(raw_input("First year:  "))
yearf = int(raw_input("Last year:  "))
years = range(yeari,yearf+1)
os.chdir(dwlf)
for i in years:
	os.system('wget '+base+str(i)+'*.*')
	print 'downloading '+ str(i)

