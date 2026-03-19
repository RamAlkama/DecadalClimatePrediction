"""
# Calculer les indice climatiques
example: NAO_index = press_Icelandic - press_Azores

1) NAO: Smith et al. (2020)

2) WEPA: L'indice WEPA est calculé comme la différence de pression au niveau de la mer mesurée entre la station de Valentia (Irlande, -10.34°W ; 51.93°N) et Santa Cruz de Tenerife (16.15°W ; 28.28°N), sur l'île des Canaries (Espagne).

(3) EAP et autre indices voir:
https://journals.ametsoc.org/view/journals/mwre/115/6/1520-0493_1987_115_1083_csapol_2_0_co_2.xml?tab_body=pdf

contact: ram.alkama@hotmail.fr
last update: 27/01/2024

Example:
    python NAO_from_ERA5.py -file ERA5_psl_1961_2023.nc -var msl -time DJFM -refyrs 1981-2010 -o ERA5Box_NAO_DJFM.txt


NB: refyrs is needed for SOI index 
"""

import xarray as xr, os, sys
import numpy as np, glob
from pathlib import Path
import time
import pandas as pd

#season='DJFM' # 'MAM'  'JJA'  'SON'  'DJF'  'DJFA'  'monthly'

# ---------WEPAbox-----------
#Valentia_Irlande
lon_Valentia_IrlandeBox,lat_Valentia_IrlandeBox = (-21, 21),(47, 61)
#Santa Cruz de Tenerife
lon_Santa_CruzBox,lat_Santa_CruzBox = (-27, 0.),(22, 36)
# ---------MedSCAND-----------
lat_MedSCANDnord,lon_MedSCANDnord = (55,70),(3,22)
lat_MedSCANDsud,lon_MedSCANDsud = (30,46),(6,35)
# ---------PNA-----------Pacific North American Pattern
lat_PNAnord,lon_PNAnord=(40,50),(-160,-150)
lat_PNAsud,lon_PNAsud=(47,53),(-125,-105)
# ---------WPO----------- West Pacific Oscillation
lat_WPOnord,lon_WPOnord=(40,50),(-160,-150)
lat_WPOsud,lon_WPOsud=(47,53),(-125,-105)
# ---------TNH----------- Tropical Northern Hemispher Pattern
lat_TNHnord,lon_TNHnord=(40,50),(-140,-125)
lat_TNHsud,lon_TNHsud=(45,55),(-90,-80)
# ----------NA--------------Northern Asian Pattern
lat_NAnord, lon_NAnord=(60,70),(25,50)
lat_NAsud, lon_NAsud=(30,45),(80,100)
# --------EP-------------- East Pacific Pattern
lat_EPnord, lon_EPnord=(60,65),(-150,-135)
lat_EPsud , lon_EPsud =(40,45),(-140,-125)


# ---------WEPA-----------
#Valentia_Irlande
lon_Valentia_Irlande,lat_Valentia_Irlande = -10.34 ,51.93
#Santa Cruz de Tenerife
lon_Santa_Cruz,lat_Santa_Cruz = -16.15, 28.28
#  ----------SOI----------------Southern Oscillation Index (SOI)
# SOI=(sSLPTahiti - sSLPDarwin)/STDmonthly
lat_Tahiti, lon_Tahiti=-17.67, -149.47
lat_Darwin, lon_Darwin=-12.44, 130.84



def normalize_longitudes(ds):
    """
    Convertit les longitudes 0 a 360 en -180 a 180 si nécessaire.
    """
    if ds.lon.max() > 200:      # Détecte format 0 a 360
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
        ds = ds.sortby('lon')
    return ds

def select_box(ds, lat_range, lon_range):
    latmin, latmax = lat_range
    lonmin, lonmax = lon_range

    # Cas normal
    if lonmin < lonmax:
        sub = ds.sel(
            lat=slice(latmin, latmax),
            lon=slice(lonmin, lonmax)
        )
    else:
        # Cas où la boîte traverse le méridien 180° (ex : 170 a -160)
        sub1 = ds.sel(lat=slice(latmin, latmax), lon=slice(lonmin, 180))
        sub2 = ds.sel(lat=slice(latmin, latmax), lon=slice(-180, lonmax))
        sub = xr.concat([sub1, sub2], dim="lon")

    return sub

def extract_NAObox(fichier):

    # Lecture du NetCDF
    if len(fichier) == 1:
        ds = xr.open_dataset(fichier[0])
    else:
        ds = xr.open_mfdataset( fichier[0][:-3] + '*.nc',  combine='by_coords'  )

    # Normalisation des longitudes
    ds = normalize_longitudes(ds)

    # ---- Traitement saisonnier ----
    if season in ['DJF', 'DJFM', 'ONDJFM', 'AMJJAS']:
        window = {'DJF': 3, 'DJFM': 4, 'ONDJFM': 6, 'AMJJAS': 6}[season]
        target_month = {'DJF': 1, 'DJFM': 2, 'ONDJFM': 1, 'AMJJAS': 7}[season]

        ds_season = ds.resample(time='ME').mean()
        ds_season = ds_season.rolling(time=window, center=True, min_periods=window).mean()
        ds_season = ds_season.sel(time=ds_season['time.month'] == target_month)

    elif season in ['MAM','JJA','SON']:
        ds_season = ds.where(ds['time.season'] == season).groupby("time.year").mean()

    else:
        ds_season = ds

    # ---- Extraction des boîtes ----
    VAL = {}
    regions = [ 'Iceland','Azores','EAPnord','EAPsud','PNAnord','PNAsud','WPOnord','WPOsud',
        'TNHnord','TNHsud','NAnord','NAsud','EPnord','EPsud','SCANDnord','SCANDsud',
        'Valentia_IrlandeBox','Santa_CruzBox','MedSCANDnord','MedSCANDsud',
        'IcelandInd','AzoresInd'   ]

    for cible in regions:
        lonmin, lonmax = eval('lon_'+cible)
        latmin, latmax = eval('lat_'+cible)

        box_data = select_box(ds_season[var], (latmin, latmax), (lonmin, lonmax))
        VAL[cible] = box_data.mean(dim=('lon','lat'))

    # ---- Calcul des indices ----
    WEPAbox_index = VAL['Valentia_IrlandeBox'] - VAL['Santa_CruzBox']
    MedSCAND_index = VAL['MedSCANDnord'] - VAL['MedSCANDsud']
    SCAND_index = VAL['SCANDnord'] - VAL['SCANDsud']
    PNA_index = VAL['PNAnord'] - VAL['PNAsud']
    WPO_index = VAL['WPOnord'] - VAL['WPOsud']
    TNH_index = VAL['TNHnord'] - VAL['TNHsud']
    NA_index  = VAL['NAnord']  - VAL['NAsud']
    EP_index  = VAL['EPsud']   - VAL['EPnord']
    EAP_index = VAL['EAPnord'] - VAL['EAPsud']
    NAOi_index = VAL['AzoresInd'] - VAL['IcelandInd']
    NAO_index  = VAL['Azores'] - VAL['Iceland']

    return  NAO_index, EAP_index, PNA_index, WPO_index, TNH_index, NA_index, EP_index, SCAND_index, WEPAbox_index, MedSCAND_index, NAOi_index

def  extract_NAOpt(fichier):
    VAL={}
        # Lecture du NetCDF
    if len(fichier) == 1:
        ds = xr.open_dataset(fichier[0])
    else:
        ds = xr.open_mfdataset( fichier[0][:-33] + '*.nc',  combine='by_coords'  )

    # Normalisation des longitudes
    ds = normalize_longitudes(ds)

    # ---- Traitement saisonnier ----
    if season in ['DJF', 'DJFM', 'ONDJFM', 'AMJJAS']:
        window = {'DJF': 3, 'DJFM': 4, 'ONDJFM': 6, 'AMJJAS': 6}[season]
        target_month = {'DJF': 1, 'DJFM': 2, 'ONDJFM': 1, 'AMJJAS': 7}[season]

        ds_season = ds.resample(time='ME').mean()
        ds_season = ds_season.rolling(time=window, center=True, min_periods=window).mean()
        ds_season = ds_season.sel(time=ds_season['time.month'] == target_month)

    elif season in ['MAM','JJA','SON']:
        ds_season = ds.where(ds['time.season'] == season).groupby("time.year").mean()

    else:
        ds_season = ds

    for cible in ['Valentia_Irlande','Santa_Cruz', 'Tahiti','Darwin']:
        #
        index_lat = abs(ds['lat'].values - eval('lat_'+cible)).argmin()
        index_lon = abs(ds['lon'].values - eval('lon_'+cible)).argmin()
        VAL[cible]= ds_season[var][:, index_lat, index_lon]
    ds.close()
    WEPA_index= (VAL['Valentia_Irlande'] - VAL['Santa_Cruz'])
    return WEPA_index,VAL['Tahiti'],VAL['Darwin']




# Record the start time
start_time = time.time()

# ----------------------
#  main
# ----------------------
if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.exit("""
  Syntax 
    NAO_from_ERA5.py [options]
        -file   input NetCDF file(s)
        -var  variable id (from the NetCDF input file)
        -time DJF, DJFM, MAM, JJA, SON or monthly

        optional parameters: 
        -o     name of the output file, 
               by default "NAO.txt"
""")
    else:
        type = sys.argv[1::2]
        value = sys.argv[2::2]
        filout = 'NAO.txt'
        refyrs = 'all'
        for typ, val in zip(type, value):
            if typ == "-file": fil = val
            if typ == "-var": var = val
            if typ == "-time": season = val
            if typ == "-o": filout = val
            if typ == "-refyrs" : refyrs= val

        try: fil
        except: sys.exit("""
 You must specify an input NetCDF file(s) by: -file FileName (FileNames can be *.nc if many)
 """)
        try: var
        except: sys.exit("""
 You must specify the variable ID to read from the inputfile by: -var VariableID
 """)
        try: season
        except: sys.exit("""
 You must specify a time step by: -time DJF, DJFM, MAM, JJA, SON or monthly 
 """)

        if (not '*'  in fil) and filout=='NAO.txt':
            filout = fil[:-3] + "_NAO_%s.txt"%season

        if season in ['DJFM','DJF','ONDJFM']:
            # ---------NAOi-----------North Atlantic Oscillation
            # lisbone
            lat_AzoresInd,lon_AzoresInd = (34,36),(-80,30)
            #Reykjavik
            lat_IcelandInd,lon_IcelandInd = (64,66),(-80,30)
            # ---------NAO-----------North Atlantic Oscillation
            # lisbone
            lat_Azores,lon_Azores = (36,40),(-28,-20)
            #Reykjavik
            lat_Iceland,lon_Iceland = (63,70),(-25,-16)
            # ---------SCAND-----------
            lat_SCANDnord,lon_SCANDnord = (57,61),(3,7)
            lat_SCANDsud,lon_SCANDsud = (61,64),(-45,-40)
            # ---------EAP-----------East Atlantic Pattern
            lat_EAPnord,lon_EAPnord=(50,55),(-28,-20)
            lat_EAPsud,lon_EAPsud=(56,60),(26,30.5)
        elif season in ['JJA','AMJJAS']:
            # ---------NAOi-----------North Atlantic Oscillation
            # lisbone
            lat_AzoresInd,lon_AzoresInd = (45,55),(-25,5)
            #Reykjavik
            lat_IcelandInd,lon_IcelandInd = (55,69),(-60,-45)
            # ---------NAO-----------North Atlantic Oscillation
            # lisbone
            lat_Azores,lon_Azores = (51,59),(-5,5)
            #Reykjavik
            lat_Iceland,lon_Iceland = (58,65),(-46.5,-36.5)
            # ---------SCAND-----------
            lat_SCANDnord,lon_SCANDnord = (54.75,58.75),(17,21)
            lat_SCANDsud,lon_SCANDsud = (52,56),(-27.75,-23.75)
            # ---------EAP-----------East Atlantic Pattern
            lat_EAPnord,lon_EAPnord=(35.25,39.25),(-33,-29)
            lat_EAPsud,lon_EAPsud=(56,60),(-13.25,-9.25)
        elif season =='SON':
            # ---------NAOi-----------North Atlantic Oscillation
            # lisbone
            lat_AzoresInd,lon_AzoresInd = (34,36),(-80,30)
            #Reykjavik
            lat_IcelandInd,lon_IcelandInd = (64,66),(-80,30)         
            # ---------NAO-----------North Atlantic Oscillation
            # lisbone
            lat_Azores,lon_Azores = (44.5,48.5),(-18,-10)
            #Reykjavik
            lat_Iceland,lon_Iceland = (66,73),(-15,-6)
            # ---------SCAND-----------
            lat_SCANDnord,lon_SCANDnord = (62.5,66.5),(-2.75,2.75)
            lat_SCANDsud,lon_SCANDsud = (56.75,60.75),(-59.5,-55.5)
            # ---------EAP-----------East Atlantic Pattern
            lat_EAPnord,lon_EAPnord=(52.75,56.75),(-27,-21)
            lat_EAPsud,lon_EAPsud=(58.5,62.5),(29.75,33.75)
        elif season =='MAM':
            # ---------NAOi-----------North Atlantic Oscillation
            # lisbone
            lat_AzoresInd,lon_AzoresInd = (45,55),(-25,5)
            #Reykjavik
            lat_IcelandInd,lon_IcelandInd = (55,69),(-60,-45)
            # ---------NAO-----------North Atlantic Oscillation
            # lisbone
            lat_Azores,lon_Azores = (39.75,43.75),(-34.75,-26.75)
            #Reykjavik
            lat_Iceland,lon_Iceland = (60.5,67.5),(-42.25,-34.25)
            # ---------SCAND-----------
            lat_SCANDnord,lon_SCANDnord = (54.25,58.25),(-4.25,-0.25)
            lat_SCANDsud,lon_SCANDsud = (37.25,41.25),(-35.75,-31.75)
            # ---------EAP-----------East Atlantic Pattern
            lat_EAPnord,lon_EAPnord=(48.5,52.5),(-24,-20)
            lat_EAPsud,lon_EAPsud=(62,64),(20.75,24.75)
        if os.path.exists(filout):
            sys.exit("ouput file %s exist, please delet it or rename it"%filout)
        else:
            fichier=glob.glob(fil) 
            NAO,EAP,PNA,WPO,TNH,NA,EP,SCAND,WEPAbox,MedSCAND,NAOi=extract_NAObox(fichier)
            WEPA,Tahiti,Darwin=extract_NAOpt(fichier)
            if 'time' in NAO.coords: 
                df = NAO['time.year'].to_dataframe(name='Year')
            else:    
                df = NAO['year'].to_dataframe(name='Year')
            if season == 'monthly':
                df['month'] = NAO['time.month'].to_dataframe()['month']
            df['NAO']  = NAO.values  
            df['NAOi'] = NAOi.values 
            df['MedSCAND'] = MedSCAND.values
            df['WEPAbox']  = WEPAbox.values
            df['WEPA']     = WEPA.values
            df['EAP']      = EAP.values
            df['PNA']      = PNA.values
            df['WPO']      = WPO.values
            df['TNH']      = TNH.values
            df['NA']      = NA.values
            df['EP']      = EP.values
            df['SCAND']   = SCAND.values
            if refyrs == 'all':
                sTahiti=(Tahiti.values-np.nanmean(Tahiti.values))/np.nanstd(Tahiti.values)
                sDarwin=(Darwin.values-np.nanmean(Darwin.values))/np.nanstd(Darwin.values)
                SOI=(sTahiti-sDarwin)/np.nanstd(sTahiti-sDarwin)
            else:
                yr1=refyrs[0:-5]
                yr2=refyrs[5:]
                if 'time' in Tahiti.coords:
                    sTahiti=(Tahiti-np.nanmean(Tahiti.sel(time=slice(yr1,yr2)).values))/np.nanstd(Tahiti.sel(time=slice(yr1,yr2)).values)
                    sDarwin=(Darwin-np.nanmean(Darwin.sel(time=slice(yr1,yr2)).values))/np.nanstd(Darwin.sel(time=slice(yr1,yr2)).values)
                    SOI=(sTahiti.values-sDarwin.values)/np.nanstd(sTahiti.sel(time=slice(yr1,yr2)).values-sDarwin.sel(time=slice(yr1,yr2)).values)
                else:
                    sTahiti=(Tahiti-np.nanmean(Tahiti.sel(year=slice(yr1,yr2)).values))/np.nanstd(Tahiti.sel(year=slice(yr1,yr2)).values)
                    sDarwin=(Darwin-np.nanmean(Darwin.sel(year=slice(yr1,yr2)).values))/np.nanstd(Darwin.sel(year=slice(yr1,yr2)).values)
                    SOI=(sTahiti.values-sDarwin.values)/np.nanstd(sTahiti.sel(year=slice(yr1,yr2)).values-sDarwin.sel(year=slice(yr1,yr2)).values)

            df['SOI']= SOI
            # Save DataFrame to ASCII file
            # Calculate mean and standard deviation of 'ps'
            # Calculate and add the standardized values column
            #df['Standardized_NAO']  = (df['NAO']  - df['NAO' ].mean() ) / df['NAO'].std() 
            #df['Standardized_WEPA'] = (df['WEPA'] - df['WEPA'].mean() )/ df['WEPA'].std()
            #df['Standardized_EAP']  = (df['EAP'] - df['EAP'].mean()  )/ df['EAP'].std()
            df.dropna(inplace=True)
            if season == 'monthly':
                df.reset_index().to_csv(filout,sep=' ', columns=['Year', 'month', 'NAO', 'WEPA', 'EAP', 'PNA', 'WPO','TNH','NA','EP','SOI','SCAND','MedSCAND','WEPAbox','NAOi'], header=True, index=False)
            else:
                df.reset_index().to_csv(filout,sep=' ', columns=['Year', 'NAO', 'WEPA', 'EAP', 'PNA', 'WPO','TNH','NA','EP','SOI','SCAND','MedSCAND','WEPAbox','NAOi'], header=True, index=False)

# Record the end time
end_time = time.time()

# Calculate the execution time in seconds
execution_time = end_time - start_time

# Convert the execution time to hours, minutes, and seconds
hours, remainder = divmod(execution_time, 3600)
print ('execution_time=',execution_time, 'second')
