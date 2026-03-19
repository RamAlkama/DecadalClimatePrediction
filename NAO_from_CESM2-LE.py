"""
# Calculer les indice climatiques
example: NAO_index = press_Icelandic - press_Azores


1) NAO: Smith et al. (2020)

2) WEPA: L'indice WEPA est calculé comme la différence de pression au niveau de la mer mesurée entre la station de Valentia (Irlande, -10.34°W ; 51.93°N) et Santa Cruz de Tenerife (16.15°W ; 28.28°N), sur l'île des Canaries (Espagne).

(3) EAP et autre indices voir:
https://journals.ametsoc.org/view/journals/mwre/115/6/1520-0493_1987_115_1083_csapol_2_0_co_2.xml?tab_body=pdf

contact: ram.alkama@hotmail.fr
last update: 26/01/2024

"""

import xarray as xr, os, sys
import numpy as np, glob
from pathlib import Path
import time
import pandas as pd


season='DJFM'#'ONDJFM' # 'MAM'  'JJA'  'SON'  'DJF'  'DJFM'  'monthly'
refyrs='1981-2010'

var='psl'
if True:
    if True:
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

# ---------WEPAbox-----------
#Valentia_Irlande
lon_Valentia_IrlandeBox,lat_Valentia_IrlandeBox = (-21, 21),(47, 61)
#Santa Cruz de Tenerife
lon_Santa_CruzBox,lat_Santa_CruzBox = (-27, 0.),(22, 36)
# ---------MedSCAND-----------
lat_MedSCANDnord,lon_MedSCANDnord = (55,70),(3,22)
lat_MedSCANDsud,lon_MedSCANDsud = (30,46),(6,35)

# Latitude and longitude coordinates
# ---------NAO-----------North Atlantic Oscillation
# lisbone
lat_Azores,lon_Azores = (36,40),(-28,-20)#36.13 #-5.35
#Reykjavik
lat_Iceland,lon_Iceland = (63,70),(-25,-16)#64.08 #-21.56
# ---------WEPA-----------
#Valentia_Irlande
lon_Valentia_Irlande,lat_Valentia_Irlande = (-11,-9),(51,53)#-10.34 #51.93
#Santa Cruz de Tenerife
lon_Santa_Cruz,lat_Santa_Cruz = (-17,-15),(27,29)#-16.15 #28.28
# ---------EAP-----------East Atlantic Pattern
lat_EAPnord,lon_EAPnord=(53,58),(-35,-20)
lat_EAPsud,lon_EAPsud=(25,35),(-10,-0.5)
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
#  ----------SOI----------------Southern Oscillation Index (SOI)
# SOI=(sSLPTahiti - sSLPDarwin)/STDmonthly
lat_Tahiti, lon_Tahiti=(-18,-17),(-150,-149)#-17.67, -149.47
lat_Darwin, lon_Darwin=(-13,-12),(130,131)#-12.44, 130.84


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
        ds = xr.open_mfdataset( fichier[0][:-33] + '*.nc',  combine='by_coords'  )

    # Normalisation des longitudes
    ds = normalize_longitudes(ds)
    ds = ds.chunk({'time': -1})
    
    # ---- Traitement saisonnier ----
    if season in ['DJF', 'DJFM', 'ONDJFM', 'AMJJAS']:
        window = {'DJF': 3, 'DJFM': 4, 'ONDJFM': 6, 'AMJJAS': 6}[season]
        target_month = {'DJF': 1, 'DJFM': 2, 'ONDJFM': 1, 'AMJJAS': 7}[season]


        ds_season = ds.resample(time='ME').mean()
        ds_season = ds_season.chunk({'time': -1})
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
    ds = ds.chunk({'time': -1})
    
    # ---- Traitement saisonnier ----
    if season in ['DJF', 'DJFM', 'ONDJFM', 'AMJJAS']:
        window = {'DJF': 3, 'DJFM': 4, 'ONDJFM': 6, 'AMJJAS': 6}[season]
        target_month = {'DJF': 1, 'DJFM': 2, 'ONDJFM': 1, 'AMJJAS': 7}[season]

        ds_season = ds.resample(time='ME').mean()
        ds_season = ds_season.chunk({'time': -1})
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


# r1i1p1f1  la première réalisation, la première initialisation, le premier membre et le premier forçage

def limm(tab):
    RR=int(tab.split('r')[1].split('i')[0])
    II=int(tab.split('r')[1].split('i')[1].split('p')[0])
    PP=int(tab.split('r')[1].split('i')[1].split('p')[1].split('f')[0])
    FF=int(tab.split('r')[1].split('i')[1].split('p')[1].split('f')[1])
    return RR,II,PP,FF


# Record the start time
start_time = time.time()

version='CESM2'

filout='ATMOSindexBox_%s_%s.txt'%(version,season)
if True:
    if True:
        if True:
            if (not os.path.exists(filout)) :
                df=[]
                if True:
                    for fil in glob.glob('CESM2_*_psl_185001-210012.nc'):
                        epp=fil.split('_')[1]
                        run,ini,member,forcing=limm(epp)
                        if True:
                            if True:
                                fichier=[fil] 
                                print (fichier)
                                if len(fichier)>0:
                                    NAO,EAP,PNA,WPO,TNH,NA,EP,SCAND,WEPAbox,MedSCAND,NAOi=extract_NAObox(fichier)
                                    WEPA,Tahiti,Darwin=extract_NAOpt(fichier)
                                    if 'time' in NAO.coords: 
                                        df_tmp = NAO['time.year'].to_dataframe(name='Year')
                                    else:    
                                        df_tmp = NAO['year'].to_dataframe(name='Year')
                                    if season == 'monthly':
                                        df_tmp['month'] = NAO['time.month'].to_dataframe()['month']
                                    df_tmp['NAO'] = NAO.values 
                                    df_tmp['NAOi'] = NAOi.values 
                                    df_tmp['MedSCAND'] = MedSCAND.values
                                    df_tmp['SCAND']   = SCAND.values
                                    df_tmp['WEPAbox']  = WEPAbox.values
                                    df_tmp['member']   = [member]*(len(df_tmp))
                                    df_tmp['ini']      = [ini]*(len(df_tmp))
                                    df_tmp['forcing']  = [forcing]*(len(df_tmp))
                                    df_tmp['run']      = [run]*(len(df_tmp))
                                    df_tmp['WEPA']     = WEPA.values
                                    df_tmp['EAP']      = EAP.values
                                    df_tmp['PNA']      = PNA.values
                                    df_tmp['WPO']      = WPO.values
                                    df_tmp['TNH']      = TNH.values
                                    df_tmp['NA']      = NA.values
                                    df_tmp['EP']      = EP.values
                                    df_tmp['Tahiti']  = Tahiti.values
                                    df_tmp['Darwin']  = Darwin.values
                                    df_tmp.dropna(inplace=True)
                                    if len(df)==0:
                                        df = df_tmp
                                    else:
                                        df = pd.concat([df,df_tmp])               
                    # Save DataFrame to ASCII file
                    yr1=int(refyrs[0:-5])
                    yr2=int(refyrs[5:  ])
                    df['sTahiti']=(df['Tahiti']-np.nanmean(df['Tahiti'].loc[(df['Year'] >= yr1) & (df['Year'] <= yr2)]))/np.nanstd(df['Tahiti'].loc[(df['Year'] >= yr1) & (df['Year'] <= yr2)])
                    df['sDarwin']=(df['Darwin']-np.nanmean(df['Darwin'].loc[(df['Year'] >= yr1) & (df['Year'] <= yr2)]))/np.nanstd(df['Darwin'].loc[(df['Year'] >= yr1) & (df['Year'] <= yr2)])
                    df['SOI']=(df['sTahiti']-df['sDarwin'])/np.nanstd(df['sTahiti'].loc[(df['Year'] >= yr1) & (df['Year'] <= yr2)]-df['sDarwin'].loc[(df['Year'] >= yr1) & (df['Year'] <= yr2)])
                    df = df[df['Year'] >= 1961]
                    if season == 'monthly':
                        df.reset_index().to_csv(filout,sep=' ', columns=['Year', 'month', 'run', 'ini','member', 'forcing', 'NAO', 'WEPA', 'EAP', 'PNA', 'WPO','TNH','NA','EP','SOI','SCAND','MedSCAND','WEPAbox','NAOi'], header=True, index=False)
                    else:
                        df.reset_index().to_csv(filout,sep=' ', columns=['Year', 'run', 'ini','member', 'forcing', 'NAO', 'WEPA', 'EAP', 'PNA', 'WPO','TNH','NA','EP','SOI','SCAND','MedSCAND','WEPAbox','NAOi'], header=True, index=False)

# Record the end time
end_time = time.time()

# Calculate the execution time in seconds
execution_time = end_time - start_time

# Convert the execution time to hours, minutes, and seconds
hours, remainder = divmod(execution_time, 3600)
print ('execution_time=',execution_time, 'second')

