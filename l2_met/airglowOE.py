# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:09:38 2021

@author: kangsun
"""
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import datetime as dt
import sys, os, glob
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from astropy.convolution import convolve_fft
from scipy.interpolate import splrep, splev
from scipy.spatial import ConvexHull
import logging
import warnings
import inspect
from collections import OrderedDict
from calendar import monthrange
from scipy.stats import binned_statistic

# In this "robust" version of arange the grid doesn't suffer 
# from the shift of the nodes due to error accumulation.
# This effect is pronounced only if the step is sufficiently small.
def arange_(lower,upper,step,dtype=None):
    npnt = np.floor((upper-lower)/step)+1
    upper_new = lower + step*(npnt-1)
    if np.abs((upper-upper_new)-step) < 1e-10:
        upper_new += step
        npnt += 1    
    return np.linspace(lower,upper_new,int(npnt),dtype=dtype)

def F_generate_control(if_save_txt,
                       hapi_directory,
                       airglowOE_directory,
                       save_directory,
                       hitran_database_path,
                       sciamachy_l1b_path,
                       control_txt_fn=None,file_suffix='_airglow_level2',
                       if_save_single_pixel_file=False,
                       if_verbose=False,
                       if_use_msis=True,
                       delta_start_wavelength=1240.,
                       delta_end_wavelength=1300.,
                       delta_min_th=25.,
                       delta_max_th=100.,
                       delta_min_th_mlt=60.,
                       delta_max_th_mlt=120.,
                       delta_w1_step=-0.001,
                       delta_r_per_e=5e8,
                       delta_nO2Scale=99,
                       delta_nO2Scale_mlt=99,
                       delta_iy0=0,delta_iy1=100,
                       delta_ix0=0,delta_ix1=100,
                       if_A_band=True,
                       sigma_start_wavelength=759.,
                       sigma_end_wavelength=772.,
                       sigma_min_th=60.,
                       sigma_max_th=100.,
                       sigma_min_th_mlt=60.,
                       sigma_max_th_mlt=120.,
                       sigma_w1_step=-0.0002,
                       sigma_r_per_e=1e7,
                       sigma_nO2Scale=99,
                       sigma_nO2Scale_mlt=99,
                       sigma_iy0=0,sigma_iy1=100,
                       sigma_ix0=0,sigma_ix1=100,):
    import yaml
    control = {}
    control['hapi directory'] = hapi_directory
    control['airglowOE directory'] = airglowOE_directory
    control['save directory'] = save_directory
    control['file suffix'] = file_suffix
    control['if save single-pixel file'] = if_save_single_pixel_file
    control['hitran database path'] = hitran_database_path
    control['sciamachy file path'] = sciamachy_l1b_path
    control['if verbose'] = if_verbose
    # use msis model pressure and O2 number density or not
    control['if use msis'] = if_use_msis
    # start/end wavelengths in nm
    control['delta start wavelength'] = delta_start_wavelength
    control['delta end wavelength'] = delta_end_wavelength
    # min/max tangent heights in km
    control['delta min tangent height'] = delta_min_th
    control['delta max tangent height'] = delta_max_th
    # min/max tangent heights in km for mesosphere-lower thermosphere orbits
    control['delta min tangent height mlt'] = delta_min_th_mlt
    control['delta max tangent height mlt'] = delta_max_th_mlt
    # forward mode wavelength step, has to be negative
    control['delta w1 step'] = delta_w1_step
    # detector response, used to estimate shot noise
    control['delta radiance per electron'] = delta_r_per_e
    control['delta number of loosen O2 layers'] = delta_nO2Scale
    control['delta number of loosen O2 layers mlt'] = delta_nO2Scale_mlt
    
    control['delta start along-track (0-based)'] = delta_iy0
    control['delta end along-track (0-based)'] = delta_iy1
    
    control['delta start across-track (0-based)'] = delta_ix0
    control['delta end across-track (0-based)'] = delta_ix1
    
    control['if A band'] = if_A_band
    
    # start/end wavelengths in nm
    control['sigma start wavelength'] = sigma_start_wavelength
    control['sigma end wavelength'] = sigma_end_wavelength
    # min/max tangent heights in km
    control['sigma min tangent height'] = sigma_min_th
    control['sigma max tangent height'] = sigma_max_th
    # min/max tangent heights in km for mesosphere-lower thermosphere orbits
    control['sigma min tangent height mlt'] = sigma_min_th_mlt
    control['sigma max tangent height mlt'] = sigma_max_th_mlt
    # forward mode wavelength step, has to be negative
    control['sigma w1 step'] = sigma_w1_step
    # detector response, used to estimate shot noise
    control['sigma radiance per electron'] = sigma_r_per_e
    control['sigma number of loosen O2 layers'] = sigma_nO2Scale
    control['sigma number of loosen O2 layers mlt'] = sigma_nO2Scale_mlt
    
    control['sigma start along-track (0-based)'] = sigma_iy0
    control['sigma end along-track (0-based)'] = sigma_iy1
    
    control['sigma start across-track (0-based)'] = sigma_ix0
    control['sigma end across-track (0-based)'] = sigma_ix1
    if if_save_txt:
        if control_txt_fn is None:control_txt_fn='control.txt'
        with open(control_txt_fn,'w') as stream:
            yaml.dump(control,stream,sort_keys=False)
    else:
        return control

def datetime2datenum(python_datetime):
    '''
    convert python datetime to matlab datenum
    '''
    matlab_datenum = python_datetime.toordinal()\
                                        +python_datetime.hour/24.\
                                        +python_datetime.minute/1440.\
                                        +python_datetime.second/86400.+366.
    return matlab_datenum

def F_read_sofie(fn,varnames=['Temperature','Temperature_precision',
                              'Latitude_83km','Longitude_83km','Time_83km',
                              'Pressure','Altitude','CH4_vmr']):
    '''
    read sofie file name, fn
    return dict
    '''
    from scipy.io import netcdf
    f = netcdf.netcdf_file(fn,'r')
    outp = {}
    for varname in varnames:
        outp[varname] = f.variables[varname][:].copy()
        if hasattr(f.variables[varname],'fill_value'):
            outp[varname][outp[varname]==f.variables[varname].fill_value] = np.nan
        if varname == 'Time_83km':
            seconds_1970_2000 = (dt.datetime(2000,1,1)-dt.datetime(1970,1,1)).total_seconds()
            outp['times2000'] = np.array([s-seconds_1970_2000 for s in outp[varname]])
            outp['python_datetime'] = np.array([dt.datetime(1970,1,1)+dt.timedelta(seconds=s) for s in outp[varname]])
            outp['matlab_datenum'] = np.array([datetime2datenum(pdt) for pdt in outp['python_datetime']])
    f.close()
    return outp
def F_distance(lat1,lon1,lat2,lon2):
    from math import sin, cos, sqrt, atan2, radians
    
    # approximate radius of earth in km
    R = 6371.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    return distance

def F_scia_sofie_collocation(sofie_dir,
                             scia_path_pattern='/projects/academic/kangsun/data/SCIAMACHY/%Y/%m/%d/SCI_%Y%m%d*.nc',
                             save_csv_path=None,
                             time_hr=2.5,space_km=100,
                             start_year=None,start_month=None,start_day=None,
                             end_year=None,end_month=None,end_day=None,
                             if_exclude_MLT=True):
    collocation_info = []
    datetime_2000 = dt.datetime(2000,1,1)
    datenum_2000 = datetime2datenum(datetime_2000)
    if start_year is not None:
        start_date = dt.date(start_year,start_month,start_day)
        end_date = dt.date(end_year,end_month,end_day)
    else:
        start_date = None
        end_date = None
    for sofie_fn in glob.glob(os.path.join(sofie_dir,'SOFIE_Level2_*_01.3.nc')):
        sofie_year = int(os.path.split(sofie_fn)[-1][13:17])
        sofie_doy = int(os.path.split(sofie_fn)[-1][17:20])
        sofie_date = dt.datetime(sofie_year, 1, 1) + dt.timedelta(sofie_doy - 1)
        sofie_date = sofie_date.date()
        if start_date is not None:
            if sofie_date < start_date:
                continue
        if end_date is not None:
            if sofie_date > end_date:
                continue
        logging.info('loading {}'.format(sofie_fn))
        sofie = F_read_sofie(sofie_fn)
        sofie['Longitude_83km'][sofie['Longitude_83km']>180]=sofie['Longitude_83km'][sofie['Longitude_83km']>180]-360
        # just in case there are scia pixels in files one day before and one day after
        scia_flist = glob.glob(sofie_date.strftime(scia_path_pattern))
        scia_flist_tomorrow = glob.glob((sofie_date+dt.timedelta(days=1)).strftime(scia_path_pattern))
        if len(scia_flist_tomorrow) > 0:
            scia_flist_tomorrow_dt = [dt.datetime.strptime(os.path.split(fn)[-1][4:19],'%Y%m%dT%H%M%S') for fn in scia_flist_tomorrow]
            scia_flist.append(scia_flist_tomorrow[np.argmin(scia_flist_tomorrow_dt)])
        scia_flist_yesterday = glob.glob((sofie_date-dt.timedelta(days=1)).strftime(scia_path_pattern))
        if len(scia_flist_yesterday) > 0:
            scia_flist_yesterday_dt = [dt.datetime.strptime(os.path.split(fn)[-1][4:19],'%Y%m%dT%H%M%S') for fn in scia_flist_yesterday]
            scia_flist = [scia_flist_yesterday[np.argmax(scia_flist_yesterday_dt)]]+scia_flist
        for scia_fn in scia_flist:
            logging.info('loading {}'.format(scia_fn))
            s = sciaOrbit(scia_fn)
            try:
                ifMLT = s.ifMLT()
            except:
                logging.warning(f'{scia_fn} gives error!')
                continue
            if if_exclude_MLT:
                if ifMLT:
                    logging.info('this orbit appears to be MLT and will be skipped')
                    continue
            s.loadData()
            granules = s.divideProfiles()
            orbitLat = np.array([g['latitude'][0,] for g in granules])
            orbitLon = np.array([g['longitude'][0,] for g in granules])
            orbitDatenum = np.array([g['time'][0,]/86400+datenum_2000 for g in granules])
            orbitDatenum=np.repeat(orbitDatenum[...,np.newaxis],orbitLat.shape[1],axis=1)
            for (isofie,sofie_datenum) in enumerate(sofie['matlab_datenum']):
                timemask = np.abs(sofie['matlab_datenum'][isofie]-orbitDatenum)*24 <= time_hr
                if np.sum(timemask) == 0:
                    continue
                sofie_lat = sofie['Latitude_83km'][isofie]
                sofie_lon = sofie['Longitude_83km'][isofie]
                dists = []
                for ltrack in range(orbitLat.shape[0]):
                    dist=np.array([F_distance(sofie_lat,sofie_lon,slat,slon) for slat,slon in zip(orbitLat[ltrack,],orbitLon[ltrack,])]) 
                    dists.append(dist)
                dists = np.array(dists)
                spacemask = dists <= space_km
                minmask = dists==np.nanmin(dists)
                allmask = minmask&spacemask&timemask
                if np.sum(allmask) == 0:
                    continue
                scia_collocation = np.where(allmask)
                scia_iy = scia_collocation[0][0]
                scia_ix = scia_collocation[1][0]
                collocation_info.append({'scia path':scia_fn,
                                        'scia iy':scia_iy,
                                        'scia ix':scia_ix,
                                        'scia lat':orbitLat[scia_iy,scia_ix],
                                        'scia lon':orbitLon[scia_iy,scia_ix],
                                        'scia datenum':orbitDatenum[scia_iy,scia_ix],
                                        'sofie path':sofie_fn,
                                        'sofie pixel':isofie,
                                        'sofie lat':sofie_lat,
                                        'sofie lon':sofie_lon,
                                        'sofie datenum':sofie_datenum,
                                        'distance (km)':dists[scia_iy,scia_ix]})
    df = pd.DataFrame.from_dict(collocation_info)
    if save_csv_path is None:
        return df
    else:
        df.to_csv(save_csv_path)

def F_read_ace(fn,varnames=['temperature','temperature_fit','pressure'],
               start_datetime=None,end_datetime=None,max_quality_flag=4):
    ace_fid = Dataset(fn)
    years = np.array(ace_fid['year'][:].squeeze(),dtype=np.int)
    months = np.array(ace_fid['month'][:].squeeze(),dtype=np.int)
    days = np.array(ace_fid['day'][:].squeeze(),dtype=np.int)
    hours = np.array(ace_fid['hour'][:].squeeze(),dtype=np.float)
    ace_datetime = np.array([dt.datetime(years[i],months[i],days[i])+dt.timedelta(hours=hours[i])
                             for i in range(len(ace_fid['year'][:]))])
    ace_datenum = np.array([datetime2datenum(d) for d in ace_datetime])
    ace_index = np.arange(len(ace_datenum))
    if start_datetime is not None:
        time_mask = (ace_datenum > datetime2datenum(start_datetime)) \
            & (ace_datenum < datetime2datenum(end_datetime))
    else:
        time_mask = np.ones(ace_datenum.shape,dtype=np.bool)
    ace_lon = ace_fid['longitude'][:].filled(np.nan).squeeze()[time_mask]
    ace_lon[ace_lon<-180] = ace_lon[ace_lon<-180]+360
    ace_lat = ace_fid['latitude'][:].filled(np.nan).squeeze()[time_mask]
    ace = {}
    ace['lon'] = ace_lon
    ace['lat'] = ace_lat
    ace['index'] = ace_index[time_mask]
    ace['H'] = ace_fid['altitude'][:].filled(np.nan)
    ace['matlab_datenum'] = ace_datenum[time_mask]
    ace['python_datetime'] = ace_datetime[time_mask]
    ace['orbit'] = ace_fid['orbit'][:].filled(np.nan).squeeze()[time_mask]
    
    for var in varnames:
        if ace_fid[var][:].shape != (ace_fid['longitude'][:].shape[0],ace_fid['altitude'][:].shape[0]):
            logging.warning('{}''s shape is not compatible, skipping'.format(var))
            continue
        ace[var] = ace_fid[var][:].filled(np.nan)
        # ace[var][ace_fid['quality_flag'][:]>max_quality_flag] = np.nan
        ace[var] = ace[var][time_mask,]
        
    # ace['pressure'] = ace_fid['pressure'][:].filled(np.nan)[time_mask,]
    ace_fid.close()
    return ace
    
def F_scia_ace_collocation(ace_path,
                           scia_path_pattern,
                           save_csv_path=None,
                           time_hr=2.5,space_km=500,
                           start_year=None,start_month=None,start_day=None,
                           end_year=None,end_month=None,end_day=None,
                           if_exclude_MLT=True):
    collocation_info = []
    datetime_2000 = dt.datetime(2000,1,1)
    datenum_2000 = datetime2datenum(datetime_2000)
    if start_year is not None:
        start_datetime = dt.datetime(start_year,start_month,start_day)
        end_datetime = dt.datetime(end_year,end_month,end_day,23,59,59)
    else:
        start_datetime = None
        end_datetime = None
    ace = F_read_ace(ace_path,varnames=[],
                     start_datetime=start_datetime,end_datetime=end_datetime)
    if start_datetime is None:
        start_datetime = ace['python_datetime'].min()
        end_datetime = ace['python_datetime'].max()
    start_date = start_datetime.date()
    end_date = end_datetime.date()
    days = (end_date-start_date).days+1
    DATES = [start_date + dt.timedelta(days=d) for d in range(days)] 
    scia_flist = []
    for DATE in DATES:
        scia_flist = scia_flist+glob.glob(DATE.strftime(scia_path_pattern))
    for scia_fn in scia_flist:
        logging.info('loading {}'.format(scia_fn))
        s = sciaOrbit(scia_fn)
        try:
            ifMLT = s.ifMLT()
        except:
            logging.warning(f'{scia_fn} gives error!')
            continue
        if if_exclude_MLT:
            if ifMLT:
                logging.info('this orbit appears to be MLT and will be skipped')
                continue
        s.loadData()
        granules = s.divideProfiles()
        orbitLat = np.array([g['latitude'][0,] for g in granules])
        orbitLon = np.array([g['longitude'][0,] for g in granules])
        orbitDatenum = np.array([g['time'][0,]/86400+datenum_2000 for g in granules])
        orbitDatenum=np.repeat(orbitDatenum[...,np.newaxis],orbitLat.shape[1],axis=1)
        ace_timemask = (ace['matlab_datenum'] > orbitDatenum.min()-time_hr/24) &\
            (ace['matlab_datenum'] < orbitDatenum.max()+time_hr/24) & \
                (~np.isnan(ace['lon'])) & (~np.isnan(ace['lat']))
        ace_lon = ace['lon'][ace_timemask]
        ace_lat = ace['lat'][ace_timemask]
        ace_orbit = ace['orbit'][ace_timemask]
        ace_index = ace['index'][ace_timemask]
        ace_datenum = ace['matlab_datenum'][ace_timemask]
        for iace, (alon,alat) in enumerate(zip(ace_lon,ace_lat)):
            dists = np.array([[F_distance(slat,slon,alat,alon) for (slat,slon) in zip(lineLat,lineLon)]
                     for (lineLat,lineLon) in zip(orbitLat,orbitLon)])
            spacemask = dists <= space_km
            timemask = np.abs(ace_datenum[iace]-orbitDatenum) < time_hr/24
            spacetimemask = spacemask&timemask
            if np.sum(spacetimemask ) == 0:
                continue
            dists[~spacetimemask] = np.nan
            logging.info('min dist = {}'.format(np.nanmin(dists)))
            scia_collocation = np.where(dists==np.nanmin(dists))
            scia_iy = scia_collocation[0][0]
            scia_ix = scia_collocation[1][0]
            collocation_info.append({'scia path':scia_fn,
                                     'scia iy':scia_iy,
                                     'scia ix':scia_ix,
                                     'scia lat':orbitLat[scia_iy,scia_ix],
                                     'scia lon':orbitLon[scia_iy,scia_ix],
                                     'scia datenum':orbitDatenum[scia_iy,scia_ix],
                                     'ace index':ace_index[iace],
                                     'ace orbit':ace_orbit[iace],
                                     'ace lat':alat,
                                     'ace lon':alon,
                                     'ace datenum':ace_datenum[iace],
                                     'distance (km)':dists[scia_iy,scia_ix]})
    df = pd.DataFrame.from_dict(collocation_info)
    if save_csv_path is None:
        return df
    else:
        df.to_csv(save_csv_path)  

def F_ncread_selective(fn,varnames,varnames_short=None):
    """
    very basic netcdf reader, similar to F_ncread_selective.m
    created on 2019/08/13
    """
    from netCDF4 import Dataset
    ncid = Dataset(fn,'r')
    outp = {}
    if varnames_short is None:
        varnames_short = varnames
    for (i,varname) in enumerate(varnames):
        try:
            outp[varnames_short[i]] = ncid[varname][:].filled(np.nan)
        except:
            logging.debug('{} cannot be filled by nan or is not a masked array'.format(varname))
            outp[varnames_short[i]] = ncid[varname][:]
    ncid.close()
    return outp

def F_scia_mipas_collocation(mipas_path_pattern,
                             scia_path_pattern,
                             save_csv_path=None,
                             time_hr=1,space_km=500,
                             start_year=None,start_month=None,start_day=None,
                             end_year=None,end_month=None,end_day=None,
                             if_exclude_MLT=False):
    collocation_info = []
    datetime_2000 = dt.datetime(2000,1,1)
    datenum_2000 = datetime2datenum(datetime_2000)
    if start_year is not None:
        start_date = dt.date(start_year,start_month,start_day)
        end_date = dt.date(end_year,end_month,end_day)
    else:
        start_date = dt.date(2002,3,1)
        end_date = dt.date(2012,4,8)
    days = (end_date-start_date).days+1
    DATES = [start_date + dt.timedelta(days=d) for d in range(days)]
    for DATE in DATES:
        scia_flist = glob.glob(DATE.strftime(scia_path_pattern))
        mipas_flist = np.array(glob.glob(DATE.strftime(mipas_path_pattern)))
        for scia_path in scia_flist:
            s = sciaOrbit(scia_path)
            try:
                ifMLT = s.ifMLT()
            except:
                logging.warning(f'{scia_path} gives error!')
                continue
            if if_exclude_MLT:
                if ifMLT:
                    logging.info('this orbit appears to be MLT and will be skipped')
                    continue
            scia_orbit_number = os.path.split(scia_path)[-1][-8:-3]
            mask = np.array([os.path.split(mipas_path)[-1][-11:-6]==scia_orbit_number for mipas_path in mipas_flist])
            if np.sum(mask) == 0:
                continue
            mipas_path = mipas_flist[mask][0]
            logging.info(f'loading scia file {os.path.split(scia_path)[-1]}')
            logging.info(f'loading mipas file {os.path.split(mipas_path)[-1]}')
            mipas = F_ncread_selective(mipas_path,['time','latitude','longitude','height','profile','profile_error','day_night','quality_flag'])
            mipas['pixel'] = np.arange(len(mipas['time']))
            mask = (mipas['quality_flag'] == 0) & (mipas['day_night'] == 1)
            mipas = {k:v[mask,] for (k,v) in mipas.items()}
            mipas['matlab_datenum'] = mipas['time']/86400+datenum_2000
            s.loadData()
            granules = s.divideProfiles()
            orbitLat = np.array([g['latitude'][0,] for g in granules])
            orbitLon = np.array([g['longitude'][0,] for g in granules])
            orbitDatenum = np.array([g['time'][0,]/86400+datenum_2000 for g in granules])
            orbitDatenum=np.repeat(orbitDatenum[...,np.newaxis],orbitLat.shape[1],axis=1)
            for imipas in range(len(mipas['time'])):
                mipas_lon = mipas['longitude'][imipas]
                mipas_lat = mipas['latitude'][imipas]
                mipas_datenum = mipas['matlab_datenum'][imipas]
                dists = np.array([[F_distance(slat,slon,mipas_lat,mipas_lon) for (slat,slon) in zip(lineLat,lineLon)]
                                  for (lineLat,lineLon) in zip(orbitLat,orbitLon)])
                spacemask = dists <= space_km
                timemask = np.abs(mipas_datenum-orbitDatenum) < time_hr/24
                spacetimemask = spacemask&timemask
                if np.sum(spacetimemask ) == 0:
                    continue
                dists[~spacetimemask] = np.nan
                logging.info('min dist = {}'.format(np.nanmin(dists)))
                scia_collocation = np.where(dists==np.nanmin(dists))
                scia_iy = scia_collocation[0][0]
                scia_ix = scia_collocation[1][0]
                collocation_info.append({'scia path':scia_path,
                                         'scia iy':scia_iy,
                                         'scia ix':scia_ix,
                                         'scia lat':orbitLat[scia_iy,scia_ix],
                                         'scia lon':orbitLon[scia_iy,scia_ix],
                                         'scia datenum':orbitDatenum[scia_iy,scia_ix],
                                         'mipas path':mipas_path,
                                         'mipas pixel':mipas['pixel'][imipas],
                                         'mipas lat':mipas_lat,
                                         'mipas lon':mipas_lon,
                                         'mipas datenum':mipas_datenum,
                                         'distance (km)':dists[scia_iy,scia_ix]})
    df = pd.DataFrame.from_dict(collocation_info)
    if save_csv_path is None:
        return df
    else:
        df.to_csv(save_csv_path)                 

class Collocated_Profile(dict):
    def __init__(self,df_row):
        self.logger = logging.getLogger(__name__)
        self.scia_ix = df_row['scia ix']
        self.scia_iy = df_row['scia iy']
        self.scia_lon = df_row['scia lon']
        self.scia_lat = df_row['scia lat']
        self.scia_datenum = df_row['scia datenum']
        self.scia_l1b_path = df_row['scia path']
        if 'ace lat' in df_row.keys():
            self.ace_lat = df_row['ace lat']
            self.ace_lon = df_row['ace lon']
            self.ace_datenum = df_row['ace datenum']
            self.ace_index = df_row['ace index']
            self.logger.info('this appears to be collocation between scia and ace')
            self.which_sensor = 'ace'
        if 'sofie lat' in df_row.keys():
            self.sofie_lat = df_row['sofie lat']
            self.sofie_lon = df_row['sofie lon']
            self.sofie_datenum = df_row['sofie datenum']
            self.sofie_pixel = df_row['sofie pixel']
            self.sofie_path = df_row['sofie path']
            self.logger.info('this appears to be collocation between scia and sofie')
            self.which_sensor = 'sofie'
        if 'mipas lat' in df_row.keys():
            self.mipas_lat = df_row['mipas lat']
            self.mipas_lon = df_row['mipas lon']
            self.mipas_datenum = df_row['mipas datenum']
            self.mipas_pixel = df_row['mipas pixel']
            self.mipas_path = df_row['mipas path']
            self.logger.info('this appears to be collocation between scia and mipas')
            self.which_sensor = 'mipas'
    def read_scia_mat(self,scia_single_pixel_dir,sigma_or_delta='delta'):
        from scipy.io import loadmat
        scia_fn = os.path.splitext(os.path.split(self.scia_l1b_path)[-1])[0]+\
            '_{}_{}_'.format(self.scia_iy,self.scia_ix)+sigma_or_delta+'.mat'
        fn = os.path.join(scia_single_pixel_dir,scia_fn)
        scia = loadmat(fn,squeeze_me=True)
        th_nan_mask = (scia['tangent_height'] < 130)#~np.isnan(scia['tangent_height'])
        self['th']=scia['tangent_height'][th_nan_mask]
        self['Hlayer'] = F_th_to_Hlayer(self['th'])
        self['Hlevel'] = F_th_to_Hlevel(self['th'])
        self[sigma_or_delta+'_T']=scia['temperature'][th_nan_mask]
        self[sigma_or_delta+'_T_dofs']=scia['temperature_dofs'][th_nan_mask]
        self['msis_T']=scia['temperature_msis'][th_nan_mask]
        self['lat'] = scia['latitude'][th_nan_mask]
        self['lon'] = scia['longitude'][th_nan_mask]
        self[sigma_or_delta+'_chi2'] = scia['chi2']
        self[sigma_or_delta+'_rmse'] = scia['rmse']
    def read_ace(self,ace_dict=None,fn=None):
        if ace_dict is None:
            ace = F_read_ace(fn)
        else:
            ace = ace_dict
        self['ace_H'] = ace['H']
        self['ace_T'] = ace['temperature'][self.ace_index,]
        self['ace_T'][ace['temperature_fit'][self.ace_index,]==0] = np.nan
    def read_sofie(self,sofie_dir=None):
        if sofie_dir is None:
            sofie_path = self.sofie_path
        else:
            sofie_path_split = os.path.split(self.sofie_path)
            sofie_path = os.path.join(sofie_dir,sofie_path_split[-1])
        sofie = F_read_sofie(sofie_path,varnames=['Temperature','Altitude'])
        self['sofie_H'] = sofie['Altitude']
        self['sofie_T'] = sofie['Temperature'][self.sofie_pixel,]
    def read_mipas(self,mipas_dir=None):
        if mipas_dir is None:
            mipas_path=self.mipas_path
        else:
            mipas_path_split=os.path.split(self.mipas_path)
            mipas_path=os.path.join(mipas_dir,mipas_path_split[-1])
        mipas = F_ncread_selective(mipas_path,['height','profile'])
        self['mipas_H']=mipas['height'][self.mipas_pixel,:]
        self['mipas_T']=mipas['profile'][self.mipas_pixel,:]
    def compare(self,min_dofs=0.5):
        
        if 'ace_H' in self.keys():
            self.logger.info('matching ace with scia')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self['ace_binned_T'],_,_ = binned_statistic(self['ace_H'],self['ace_T'],statistic=np.nanmean,bins=self['Hlevel'])
            if 'delta_T' in self.keys():
                self['delta_minus_ace_T'] = self['delta_T']-self['ace_binned_T']
                self['delta_minus_ace_T'][self['delta_T_dofs']<min_dofs] = np.nan
            if 'sigma_T' in self.keys():
                self['sigma_minus_ace_T'] = self['sigma_T']-self['ace_binned_T']
                self['sigma_minus_ace_T'][self['sigma_T_dofs']<min_dofs] = np.nan
            if 'msis_T' in self.keys():
                self['msis_minus_ace_T'] = self['msis_T']-self['ace_binned_T']
        if 'sofie_H' in self.keys():
            self.logger.info('matching sofie with scia')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self['sofie_binned_T'],_,_ = binned_statistic(self['sofie_H'],self['sofie_T'],statistic=np.nanmean,bins=self['Hlevel'])
            if 'delta_T' in self.keys():
                self['delta_minus_sofie_T'] = self['delta_T']-self['sofie_binned_T']
                self['delta_minus_sofie_T'][self['delta_T_dofs']<min_dofs] = np.nan
            if 'sigma_T' in self.keys():
                self['sigma_minus_sofie_T'] = self['sigma_T']-self['sofie_binned_T']
                self['sigma_minus_sofie_T'][self['sigma_T_dofs']<min_dofs] = np.nan
            if 'msis_T' in self.keys():
                self['msis_minus_sofie_T'] = self['msis_T']-self['sofie_binned_T']
        if 'mipas_H' in self.keys():
            self.logger.info('matching mipas with scia')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self['mipas_binned_T'],_,_ = binned_statistic(self['mipas_H'],self['mipas_T'],statistic=np.nanmean,bins=self['Hlevel'])
            if 'delta_T' in self.keys():
                self['delta_minus_mipas_T'] = self['delta_T']-self['mipas_binned_T']
                self['delta_minus_mipas_T'][self['delta_T_dofs']<min_dofs] = np.nan
            if 'sigma_T' in self.keys():
                self['sigma_minus_mipas_T'] = self['sigma_T']-self['mipas_binned_T']
                self['sigma_minus_mipas_T'][self['sigma_T_dofs']<min_dofs] = np.nan
            if 'msis_T' in self.keys():
                self['msis_minus_mipas_T'] = self['msis_T']-self['mipas_binned_T']
    def plot_raw_profiles(self,existing_ax=None,
                          which_sensor=None,
                          sigma_or_delta='delta',
                          size_legend=True):
        if existing_ax is None:
            self.logger.info('axes not supplied, creating one')
            fig,ax = plt.subplots(1,1,figsize=(10,5))
        else:
            fig = None
            ax = existing_ax
        if which_sensor is None:
            which_sensor = self.which_sensor
        figout = {}
        figout['fig'] = fig
        figout['ax'] = ax
        profile_color = {}
        profile_color['delta'] = 'red'
        profile_color['sigma'] = 'blue'
        leg_dict = {'msis_line':'Prior (MSIS)',
                    'delta_line':r'O$_2$ $\Delta$',
                    'sigma_line':r'O$_2$ $\Sigma$',
                    'ace_line':'ACE-FTS',
                    'sofie_line':'SOFIE'}
        figout[which_sensor.lower()+'_line'] =ax.plot(self[which_sensor.lower()+'_T'],self[which_sensor.lower()+'_H'],'-k')
        if 'T_msis' in self.keys():
            figout['msis_line'] = ax.plot(self['T_msis'],self['Hlayer'],linewidth=2,color='gray',alpha=0.5)
        ref_dofs = 1
        ref_size = 50
        if sigma_or_delta is None:
            figout['delta_line'] = ax.plot(self['delta_T'],self['Hlayer'],'-',
                    color=profile_color['delta'],linewidth=1)
            figout['sigma_line'] = ax.plot(self['sigma_T'],self['Hlayer'],'-',
                    color=profile_color['sigma'],linewidth=1)
            figout['delta_scatter'] = ax.scatter(self['delta_T'],self['Hlayer'],
                       s=self['delta_T_dofs']/ref_dofs*ref_size,color=profile_color['delta'])
            figout['sigma_scatter'] = ax.scatter(self['sigma_T'],self['Hlayer'],
                       s=self['sigma_T_dofs']/ref_dofs*ref_size,color=profile_color['sigma'])
            # ax.legend([which_sensor.upper(),r'O$_2$ $\Delta$',r'O$_2$ $\Sigma$'])
        else:
            figout[sigma_or_delta+'_line'] = ax.plot(self[sigma_or_delta+'_T'],self['Hlayer'],'-',
                    color=profile_color[sigma_or_delta],linewidth=1)
            figout[sigma_or_delta+'_scatter'] = ax.scatter(self[sigma_or_delta+'_T'],self['Hlayer'],
                       s=self[sigma_or_delta+'_T_dofs']/ref_dofs*ref_size,color=profile_color[sigma_or_delta])
        handle_list = []
        lname_list = []
        for (k,v) in figout.items():
            if k in leg_dict.keys():
                handle_list.append(v[0])
                lname_list.append(leg_dict[k])
        ax.legend(handle_list,lname_list)
        ax.set_xlabel('Temperature [K]');
        ax.set_ylabel('Altitude [km]');
        ax.set_title('SCIAMACHY (lat: {:.2f}, lon: {:.2f}), {} (lat: {:.2f}, lon: {:.2f})'.format(
            self.scia_lat,self.scia_lon,which_sensor.upper(),
            getattr(self,which_sensor.lower()+'_lat'),getattr(self,which_sensor.lower()+'_lon')));
        if size_legend:
            from matplotlib.lines import Line2D
            insert_ax = fig.add_axes([0.15,0.15,0.2,0.1])
            xxx = np.array([0.1,0.25,0.5,0.75,1])
            insert_ax.scatter(xxx,np.zeros(xxx.shape),s=xxx/ref_dofs*ref_size,color='gray')
            insert_ax.set_xticks(xxx)
            insert_ax.set_xlabel('DOFS')
            insert_ax.set_ylim((-0.15,0.15));
            insert_ax.set_xlim((np.min(xxx)-.1,np.max(xxx)+.1));
            insert_ax.set_frame_on(False)
            insert_ax.axes.get_yaxis().set_visible(False)
            xmin,xmax = insert_ax.get_xaxis().get_view_interval()
            ymin,ymax = insert_ax.get_yaxis().get_view_interval()
            insert_ax.add_artist(Line2D((xmin,xmax),(ymin,ymin),color='k',linewidth=2))
            figout['insert_ax'] = insert_ax
        return figout
    
class Collocated_Profiles(list):
    def __init__(self,input_list=[]):
        self.logger = logging.getLogger(__name__)
        self.npair = 0
        for cp in input_list:
            self.append_Collocated_Profile(cp)
    def append_Collocated_Profile(self,cp):
        self.append(cp)
        if not hasattr(self,'which_sensor'):
            self.which_sensor = cp.which_sensor
        elif self.which_sensor != cp.which_sensor:
            self.logger.error('this should not happen. cps and cp conflict in which_sensor')
        self.npair += 1
    def bin_bias_profiles(self,which_sensor=None,
                          sigma_or_delta='delta',
                          vertical_edges=None,func=np.nanmean):
        '''' the following doesn't work if Hlayer has not the same length
        Hlayer_all = np.array([cp['Hlayer'] for cp in self]).ravel()
        bias_all = np.array([cp[sigma_or_delta.lower()+'_minus_{}_T'.format(which_sensor.lower())] for cp in self]).ravel()
        '''
        if which_sensor is None: which_sensor = self.which_sensor
        Hlayer_all = np.hstack([cp['Hlayer'] for cp in self])
        bias_all = np.hstack([cp[sigma_or_delta.lower()+'_minus_{}_T'.format(which_sensor.lower())] for cp in self])
        if vertical_edges is None:
            vertical_edges = np.linspace(Hlayer_all.min(),Hlayer_all.max(),51)
        vertical_mid = vertical_edges[0:-1]+np.diff(vertical_edges)/2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            profile,_,_ = binned_statistic(Hlayer_all, bias_all, statistic=func,bins=vertical_edges)
        outp = {'H':vertical_mid,'dT':profile}
        return outp

class layer():
    
    def __init__(self,dz=None,p=1.01325e5,T=296.,
                 minWavelength=1240.,maxWavelength=1300.,
                 nO2s=0.,nO2=None,einsteinA=None):
        '''
        initialize basic properties of the layer
        nO2s:
            singlet delta state O2 in molecules/cm3
        '''
        # layer thickness in m
        self.dz = dz
        # layer temperature in K
        self.T = T
        # layer pressure in Pa
        self.p = p
        # O2 number density in molecules/cm3
        if nO2 is None:
            self.nO2 = p/T/1.38065e-23*0.2095*1e-6
        else:
            self.nO2 = nO2
        self.nO2s = nO2s
        # Einstein A coefficient for the band in s-1
        if maxWavelength < 800:
            if einsteinA is None:
                self.einsteinA = 0.08693#10.1016/j.jqsrt.2010.05.011
            else:
                self.einsteinA = einsteinA
        elif minWavelength > 1200:
            if einsteinA is None:
                self.einsteinA = 2.27e-4
            else:
                self.einsteinA = einsteinA
        self.minWavelength = minWavelength
        self.maxWavelength = maxWavelength
        self.minWavenumber = 1e7/maxWavelength
        self.maxWavenumber = 1e7/minWavelength
        
    def getAbsorption(self,nu=None,
                      finiteDifference=True,dT=0.01,
                      WavenumberWing=3.,sourceTableName=None):
        '''
        call hapi line by line function to calculate absorption
        nu:
            wavenumber grid, if None, construct from min/maxWavenumber
        finiteDifference:
            True means dsigma/dT through finite difference
        dT:
            dT in finite difference
        WavenumberWing:
            window width input to hapi to calculate line profile
        '''
        from hapi import absorptionCoefficient_Voigt, absorptionCoefficient_Voigt_jac
        if sourceTableName == None:
            sourceTableName = 'O2_{:.1f}-{:.1f}'.format(self.minWavelength,self.maxWavelength)
        if nu is None:
            if finiteDifference:
                nu, sigma = absorptionCoefficient_Voigt(SourceTables=sourceTableName, 
                                                    Environment={'p':self.p/1.01325e5,'T':self.T}, 
                                                    OmegaRange=[self.minWavenumber,self.maxWavenumber],
                                                    OmegaStep=0.005,
                                                    WavenumberWing=WavenumberWing)
                _, sigma1 = absorptionCoefficient_Voigt(SourceTables=sourceTableName, 
                                                    Environment={'p':self.p/1.01325e5,'T':self.T+dT}, 
                                                    OmegaRange=[self.minWavenumber,self.maxWavenumber],
                                                    OmegaStep=0.005,
                                                    WavenumberWing=WavenumberWing)
                dsigmadT = (sigma1-sigma)/dT
            else:
                nu, sigma, dsigmadT = absorptionCoefficient_Voigt_jac(SourceTables=sourceTableName, 
                                                    Environment={'p':self.p/1.01325e5,'T':self.T}, 
                                                    OmegaRange=[self.minWavenumber,self.maxWavenumber],
                                                    OmegaStep=0.005,
                                                    WavenumberWing=WavenumberWing)
        else:
            if finiteDifference:
                nu, sigma = absorptionCoefficient_Voigt(SourceTables=sourceTableName, 
                                                    Environment={'p':self.p/1.01325e5,'T':self.T},
                                                    OmegaGrid=nu,
                                                    WavenumberWing=WavenumberWing)
                _, sigma1 = absorptionCoefficient_Voigt(SourceTables=sourceTableName, 
                                                    Environment={'p':self.p/1.01325e5,'T':self.T+dT},
                                                    OmegaGrid=nu,
                                                    WavenumberWing=WavenumberWing)
                dsigmadT = (sigma1-sigma)/dT
            else:
                nu, sigma, dsigmadT = absorptionCoefficient_Voigt_jac(SourceTables=sourceTableName, 
                                                    Environment={'p':self.p/1.01325e5,'T':self.T},
                                                    OmegaGrid=nu,
                                                    WavenumberWing=WavenumberWing)
        # sigma: absorption cross section in cm2; dz: layer thickness in m; nO2: O2 number density in molc/cm3
        if self.dz is not None:
            self.tau = sigma*(self.dz*100)*self.nO2
        self.sigma = sigma
        self.dsigmadT = dsigmadT
        self.nu = nu
        self.wvl = 1e7/nu
        
    def getAirglowEmission(self,nO2s=None):
        '''
        calculate airglow volume emission rate spectra
        have to run getAbsorption first to get sigma
        '''
        if nO2s is not None:
            self.nO2s = nO2s
        c2 = 1.4387769
        y = self.sigma*np.power(self.nu,2)/(np.exp(c2*self.nu/self.T)-1)
        dydT = np.power(self.nu,2)*(self.dsigmadT/(np.exp(c2*self.nu/self.T)-1)\
                        +self.sigma/np.power(np.exp(c2*self.nu/self.T)-1,2)*np.exp(c2*self.nu/self.T)*c2*self.nu/np.power(self.T,2))
        int_y = np.abs(np.trapz(y,self.nu))
        int_dydT = np.trapz(dydT,self.nu)
        yn = y/int_y
        dyndT = (dydT*int_y-int_dydT*y)/np.power(int_y,2)
        # airglow ver spectrum in photons/cm3/s/cm-1
        self.emission_nu = yn*self.nO2s*self.einsteinA
        # airglow ver spectrum in photons/cm3/s/nm
        self.emission = self.emission_nu*self.nu/self.wvl
        self.dedT = dyndT*self.nO2s*self.einsteinA*self.nu/self.wvl
        self.dednO2s = self.emission/self.nO2s
                
    def plotSpectrum(self,xlim=None,whichVariable='sigma'):
        '''
        plot spectra saved as attributes of the layer class object
        '''
        import matplotlib.pyplot as plt
        plt.plot(self.nu,getattr(self,whichVariable))
        if whichVariable == 'sigma':
            ylabel = r'$\sigma$ [cm$^2$/molecule]'
            longName = 'Absorption coefficient'
        elif whichVariable == 'tau':
            ylabel = r'$\tau$'
            longName = 'Optical thickness'
        elif whichVariable == 'emission':
            ylabel = r'$\varepsilon$'
            longName = 'Emission'
        else:
            ylabel = whichVariable
            longName = whichVariable
        plt.xlabel(r'$\nu$ [cm$^{-1}$]')
        plt.ylabel(ylabel)
        plt.title(longName+' at p = {} hPa, T = {} K'.format(self.p/100, self.T), fontsize=10)
        if xlim is None:
            xlim = (self.minWavenumber,self.maxWavenumber)
        plt.xlim(xlim)

def compare_nan_array(func, a, thresh):
    out = ~np.isnan(a)
    out[out] = func(a[out] , thresh)
    return out

class sciaOrbit():
    def __init__(self,sciaPath):
        self.sciaPath = sciaPath
        self.nc = Dataset(sciaPath)
    
    def ifMLT(self):
        allTH = self.nc['/limb__20/tangent_height'][:].filled(np.nan)
        n_above_100 = np.sum(compare_nan_array(np.greater,allTH.ravel(),100))
        n_allTH = len(allTH[~np.isnan(allTH)].ravel())
        if n_above_100/n_allTH > 0.2:
            if_mlt_or_not = True
        else:
            if_mlt_or_not = False
        return if_mlt_or_not
        
    def loadData(self,if_close_file=True,startWavelength=1200,endWavelength=1340):
        varnames = ['radiance','wavelength','latitude','longitude','time',
                    'solar_zenith_angle','latitude_bounds','longitude_bounds',
                    'tangent_height']#'pixel_quality_flag' is useless
        singletDeltaData = {}
        self.startWavelength = startWavelength
        self.endWavelength = endWavelength
        if endWavelength < 800:
            bandStr = '/limb__20/'
        elif startWavelength > 1100:
            bandStr = '/limb__30/'
        for varname in varnames:
            singletDeltaData[varname] = self.nc[bandStr+varname][:]
        self.singletDeltaData = singletDeltaData
        if if_close_file:
            self.nc.close()
    
    def divideProfiles(self,badPixels=(348,374),radiancePerElectron=None):
        data = self.singletDeltaData
        if self.endWavelength < 800:
            b1=750;b2=759;b3=767;b4=780
            badPixels = ()
            if radiancePerElectron == None:
                radiancePerElectron = 5e7
        elif self.startWavelength > 1100:
            b1=1210;b2=1240;b3=1300;b4=1340
            if radiancePerElectron == None:
                radiancePerElectron = 1e9
        meanWvl = np.nanmedian(data['wavelength'],axis=0)
        wvlMask = (meanWvl > self.startWavelength) & (meanWvl < self.endWavelength)
        TH = data['tangent_height'].filled(np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            meanTH = np.nanmean(TH,axis=1)
        diffMeanTH = np.diff(meanTH)
        diffMeanTH = np.append(diffMeanTH,np.nan)
        n_positive_diff = np.sum(compare_nan_array(np.greater,diffMeanTH,0))
        n_negative_diff = np.sum(compare_nan_array(np.less,diffMeanTH,0))
        if n_positive_diff > n_negative_diff:# scan appears to be bottom-up
            turningMask = compare_nan_array(np.less,diffMeanTH,0)
        else:# scan appears to be top down
            turningMask = compare_nan_array(np.greater,diffMeanTH,0)
        indexArray = np.arange(len(meanTH))
        granules = []
        idx = 0
        while idx <= len(meanTH):
            tmp = indexArray[(turningMask | np.isnan(diffMeanTH))&(indexArray>idx)]
            if len(tmp)==0:
                break
            next_idx = tmp[0]
            # short profile indicates all nan
            if next_idx >= idx+2:
                granule = {}
                # second condition removes 16th levels in mlt
                if all(data['tangent_height'].mask[idx,:]) or \
                (np.ptp(data['tangent_height'][idx,:])> 100 and data['tangent_height'][idx,:].min()<100):
                    start_idx = idx+1
                else:
                    start_idx = idx
                for key in data.keys():
                    granule[key] = data[key][start_idx:next_idx+1,].filled(np.nan)
                # manually remove bad pixels
                granule['radiance'][:,:,badPixels] = np.nan
                # trim radiance and wavelength
                granule['radiance'] = granule['radiance'][:,:,wvlMask]
                granule['wavelength'] = granule['wavelength'][:,wvlMask]
                granule['radiance_error'] = np.ones_like(granule['radiance'])
                # remove background radiance defined at 130-140 km for A band
                if self.endWavelength < 800:# A band
                    bg_radiance = np.full_like(granule['wavelength'],np.nan)
                    bg_shoulder = np.full(granule['radiance'].shape[1],np.nan)
                    v_wavelength = np.nanmedian(granule['wavelength'],axis=0)
                    for ift in range(granule['radiance'].shape[1]):
                        bg_radiance[ift,:] = np.nanmean(granule['radiance'][((granule['tangent_height'][:,ift]>130) & (granule['tangent_height'][:,ift]<150)),ift,:],axis=0)
                        waveMask = ((v_wavelength >= b1) & (v_wavelength <= b2)) |\
                        ((v_wavelength >= b3) & (v_wavelength <= b4))
                        bg_shoulder[ift] = np.nanmean(bg_radiance[ift,waveMask])
                for ith in range(granule['radiance'].shape[0]):
                    waveMask = ((granule['wavelength'][ith,:] >= b1) & (granule['wavelength'][ith,:] <= b2)) |\
                    ((granule['wavelength'][ith,:] >= b3) & (granule['wavelength'][ith,:] <= b4))
                    
                    xx = granule['wavelength'][ith,:][waveMask]
#                    print(ith)
                    for ift in range(granule['radiance'].shape[1]):
                        yy = granule['radiance'][ith,ift,:].squeeze()[waveMask]
                        if self.endWavelength < 800:
                            rad_shoulder = np.nanmean(yy)
                            granule['radiance'][ith,ift,:] = granule['radiance'][ith,ift,:]-bg_radiance[ift,:]/bg_shoulder[ift]*rad_shoulder
                            yy = granule['radiance'][ith,ift,:].squeeze()[waveMask]
                        if all(np.isnan(yy)):
                            granule['radiance_error'][ith,ift,:] = np.nan*granule['radiance_error'][ith,ift,:]
                            continue
                        baseLinePoly = np.polyfit(xx,yy,1)
                        granule['radiance'][ith,ift,:] = granule['radiance'][ith,ift,:]-\
                        np.polyval(baseLinePoly,granule['wavelength'][ith,:])
                        readOutNoise = np.nanstd(yy-np.polyval(baseLinePoly,xx))
                        allNoise = np.sqrt(np.abs(granule['radiance'][ith,ift,:])*radiancePerElectron+
                        readOutNoise**2)
                        allNoise[(allNoise<readOutNoise)|np.isnan(allNoise)] = readOutNoise
                        granule['radiance_error'][ith,ift,:] = allNoise
                
                granules.append(granule)
            idx = next_idx+1
        return granules
        
    def plotBounds(self,granule,THLimit=(20,200),alpha=0.8):
        from matplotlib.collections import PolyCollection
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 22})
        THFilter = (np.nanmean(granule['tangent_height'],axis=1)>=THLimit[0]) &\
        (np.nanmean(granule['tangent_height'],axis=1)<=THLimit[1])
        tangent_height = granule['tangent_height'][THFilter,]
        lat_r = granule['latitude_bounds'][THFilter,]
        lon_r = granule['longitude_bounds'][THFilter,]
        lat_c = granule['latitude'][THFilter,]
        lon_c = granule['longitude'][THFilter,]
        plt.plot(lon_c,lat_c,color='none')
        nth = lat_r.shape[0]
        for ift in range(8):
            verts = []
            for ith in range(nth):
                xs = lon_r[ith,ift,(0,2,3,1)].squeeze();ys = lat_r[ith,ift,(0,2,3,1)].squeeze()
                verts.append(list(zip(xs,ys)))
            collection = PolyCollection(verts,
                        array=tangent_height[:,ift],cmap='rainbow_r',edgecolors='k')
            collection.set_alpha(alpha)
            plt.gca().add_collection(collection)
        
    def plotGranule(self,granule,waveLimit=(),THLimit=(20,500)):
        import matplotlib.pyplot as plt
        if len(waveLimit) == 0:
            waveLimit = (self.startWavelength,self.endWavelength)
        THFilter = (np.nanmean(granule['tangent_height'],axis=1)>=THLimit[0]) &\
        (np.nanmean(granule['tangent_height'],axis=1)<=THLimit[1])
        waveFilter = (np.nanmean(granule['wavelength'],axis=0)>=waveLimit[0]) &\
        (np.nanmean(granule['wavelength'],axis=0)<=waveLimit[1])
        fig, axs = plt.subplots(2,4)
        axs = axs.ravel()
        for ift in range(8):
            axs[ift].plot(granule['wavelength'][np.ix_(THFilter,waveFilter)].T,
               granule['radiance'][:,ift,:].squeeze()[np.ix_(THFilter,waveFilter)].T)
            axs[ift].plot(granule['wavelength'][np.ix_(THFilter,waveFilter)].T,
               granule['radiance_error'][:,ift,:].squeeze()[np.ix_(THFilter,waveFilter)].T,'-k')
            axs[ift].legend(['{:.1f}'.format(th) for th in np.nanmean(granule['tangent_height'],axis=1)[THFilter]])

def F_airglow_forward_model(w1,wavelength,L,p_profile,
                            nO2s_profile,T_profile,HW1E,w_shift,T_profile_reference=None,
                            nu=None,einsteinA=None,nO2_profile=None):
    '''
    forward model to simulate scia-observed limb spectra for a profile scan
    output jacobians
    '''
    nu = nu or 1e7/w1
    T_profile_reference = T_profile_reference or T_profile.copy()
    dw1 = np.abs(np.median(np.diff(w1)))
    ndx = np.ceil(HW1E*3/dw1);
    xx = np.arange(ndx*2)*dw1-ndx*dw1;
    ILS = 1/np.sqrt(np.pi)/HW1E*np.exp(-np.power(xx/HW1E,2))*dw1
    dILSdHW1E = 1/np.sqrt(np.pi)*(-1/np.power(HW1E,2)+2*np.power(xx,2)/np.power(HW1E,4))*np.exp(-np.power(xx/HW1E,2))*dw1
    nw1 = len(nu)
    nw2 = wavelength.shape[1]
    nth = len(p_profile)
    sigma_ = np.zeros((nth,nw1))
    dsigmadT_ = np.zeros((nth,nw1))
    emission_ = np.zeros((nth,nw1))
    dedT_ = np.zeros((nth,nw1))
    dednO2s_ = np.zeros((nth,nw1))
    if nO2_profile is None:
        nO2_profile = [None]*nth
    # get xsection (sigma_), emission, and jacobians of emission for each layer
    for ith in range(nth):
        l = layer(p=p_profile[ith],T=T_profile[ith],
                  minWavelength=np.min(w1),maxWavelength=np.max(w1),einsteinA=einsteinA,nO2=nO2_profile[ith])
        l.getAbsorption(nu=nu)
        sigma_[ith,] = l.sigma*l.nO2# this is optical depth divided by length
        dsigmadT_[ith,] = l.dsigmadT*l.nO2#-l.sigma*l.p/1.38065e-23*0.2095*1e-6/l.T**2# this is d(optical depth divided by length)/dT
        l.getAirglowEmission(nO2s=nO2s_profile[ith])
        emission_[ith,] = l.emission
        dedT_[ith,] = l.dedT
        dednO2s_[ith,] = l.dednO2s
    
    # high res attenuated emission at each tangent height
    obs_R1 = np.zeros(emission_.shape)
    # high res jacobians to layer temperature
    obs_dR1dT = np.zeros(emission_.shape+(nth,))
    # high res jacobians to layer excited o2 density
    obs_dR1dnO2s = np.zeros(emission_.shape+(nth,))
    # low res radiance at each tangent height
    obs_R2 = np.zeros((nth,nw2))
    # jacobian to ils hw1e
    obs_dR2dHW1E = np.zeros((nth,nw2))
    # jacobian to wavelength shift
    obs_dR2dw_shfit = np.zeros((nth,nw2))
    # low res jacobians to layer temperature
    obs_dR2dT = np.zeros((nth,nw2,nth))
    # low res jacobians to layer excited o2 density
    obs_dR2dnO2s = np.zeros((nth,nw2,nth))
    
    for i in range(nth):# observation at tangent height i
        # piercing through the onion from close side to far side
        absorbing_layer_idx = np.hstack((np.arange(nth-1,i-1,-1),np.arange(i,nth-1)))
        emitting_layer_idx = np.hstack((np.arange(nth-1,i-1,-1),np.arange(i,nth)))
        cum_tau = np.cumsum(np.array([sigma_[k,]*L[i,k] for k in absorbing_layer_idx]),axis=0)
        emitting_layer_tau = np.array([-np.log((1-np.exp(-sigma_[k,]*L[i,k]))/(sigma_[k,]*L[i,k])) 
                                       for k in emitting_layer_idx])
        # d(emitting layer tau)/dT = d(emitting layer tau)/d(extinction) * d(extinction)/dT
        detau_dT = np.array([(-(sigma_[k,]*L[i,k]*np.exp(-sigma_[k,]*L[i,k])+np.exp(-sigma_[k,]*L[i,k])-1)\
                              /(sigma_[k,]*(1-np.exp(-sigma_[k,]*L[i,k]))))\
                            *dsigmadT_[k,] for k in emitting_layer_idx])
        
        for (count,j) in enumerate(emitting_layer_idx):
            if count == 0:
                total_tau = emitting_layer_tau[count,]
            else:
                # downstream layers' tau+emitting layer tau
                total_tau = cum_tau[count-1,]+emitting_layer_tau[count,]
            
            # radiance
            obs_R1[i,] = obs_R1[i,]+emission_[j,]/(4*np.pi)*L[i,j]*np.exp(-total_tau)
            
            # d(radiance)/d(emitting layer nO2s)
            obs_dR1dnO2s[i,:,j] = obs_dR1dnO2s[i,:,j]+dednO2s_[j,]/(4*np.pi)*L[i,j]*np.exp(-total_tau)
            
            # d(radiance)/d(emitting layer temperature), without accounting for upstream layers
            obs_dR1dT[i,:,j] = obs_dR1dT[i,:,j]+dedT_[j,]/(4*np.pi)*L[i,j]*np.exp(-total_tau)\
            +emission_[j,]/(4*np.pi)*L[i,j]*np.exp(-total_tau)*(-detau_dT[count,])
            
            # loop over upstream layers
            for (count1,k) in enumerate(emitting_layer_idx[count+1:]):
                total_tau = cum_tau[count+count1,]+emitting_layer_tau[count+count1+1,]
                # d(radiance)/d(emitting layer temperature) due to altered emission from upstream layers
                obs_dR1dT[i,:,j] = obs_dR1dT[i,:,j]\
                +emission_[k,]/(4*np.pi)*L[i,k]*np.exp(-total_tau)*(-L[i,j]*dsigmadT_[j,])
        
        R1_oversampled_fft = convolve_fft(obs_R1[i,::-1],ILS,normalize_kernel=True)
        
        dR1dHW1E_oversampled_fft = convolve_fft(obs_R1[i,::-1],dILSdHW1E,normalize_kernel=False)
        bspline_coef = splrep(w1[::-1],R1_oversampled_fft)
        obs_R2[i,] = splev(wavelength[i,]+w_shift,bspline_coef)
        obs_dR2dw_shfit[i,] = splev(wavelength[i,]+w_shift,bspline_coef,der=1)
        
        bspline_coef = splrep(w1[::-1],dR1dHW1E_oversampled_fft)
        obs_dR2dHW1E[i,] = splev(wavelength[i,]+w_shift,bspline_coef)
        
        for j in range(i,nth):
            dR1dT_oversampled_fft = convolve_fft(obs_dR1dT[i,::-1,j],ILS,normalize_kernel=True)
            dR1dnO2s_oversampled_fft = convolve_fft(obs_dR1dnO2s[i,::-1,j],ILS,normalize_kernel=True)
            bspline_coef = splrep(w1[::-1],dR1dT_oversampled_fft)
            obs_dR2dT[i,:,j] = splev(wavelength[i,]+w_shift,bspline_coef)
            bspline_coef = splrep(w1[::-1],dR1dnO2s_oversampled_fft)
            obs_dR2dnO2s[i,:,j] = splev(wavelength[i,]+w_shift,bspline_coef)
    jacobians = {}
    jacobians['nO2s_profile'] = obs_dR2dnO2s.reshape(-1,nth)
    jacobians['T_profile'] = obs_dR2dT.reshape(-1,nth)
    jacobians['HW1E'] = obs_dR2dHW1E.reshape(-1,1)
    jacobians['w_shift'] = obs_dR2dw_shfit.reshape(-1,1)
    return obs_R2,jacobians

def F_airglow_forward_model_nO2Scale(w1,wavelength,L,p_profile,
                               nO2s_profile,T_profile,HW1E,w_shift,
                               nO2Scale_profile,T_profile_reference=None,
                               nu=None,einsteinA=None,nO2_profile=None):
    '''
    forward model to simulate scia-observed limb spectra for a profile scan
    output jacobians
    '''
    nu = nu or 1e7/w1
    T_profile_reference = T_profile_reference or T_profile.copy()
    dw1 = np.abs(np.median(np.diff(w1)))
    ndx = np.ceil(HW1E*3/dw1);
    xx = np.arange(ndx*2)*dw1-ndx*dw1;
    ILS = 1/np.sqrt(np.pi)/HW1E*np.exp(-np.power(xx/HW1E,2))*dw1
    dILSdHW1E = 1/np.sqrt(np.pi)*(-1/np.power(HW1E,2)+2*np.power(xx,2)/np.power(HW1E,4))*np.exp(-np.power(xx/HW1E,2))*dw1
    nw1 = len(nu)
    nw2 = wavelength.shape[1]
    nth = len(p_profile)
    sigma_ = np.zeros((nth,nw1))
    dsigmadT_ = np.zeros((nth,nw1))
    dsigmadnO2_ = np.zeros((nth,nw1))
    emission_ = np.zeros((nth,nw1))
    dedT_ = np.zeros((nth,nw1))
    dednO2s_ = np.zeros((nth,nw1))
    if nO2_profile is None:
        # this is O2 number density from ideal gas law
        nO2_profile_full = np.array([p_profile[i]/T_profile_reference[i]/1.38065e-23*0.2095*1e-6 for i in range(len(T_profile))])
    else:
        nO2_profile_full = nO2_profile
    n_nO2 = len(nO2Scale_profile)
    nO2Scale_profile_full = np.ones_like(T_profile)
    nO2Scale_profile_full[0:n_nO2] = nO2Scale_profile
    # get xsection (sigma_), emission, and jacobians of emission for each layer
    for ith in range(nth):
        l = layer(p=p_profile[ith],T=T_profile[ith],
                  minWavelength=np.min(w1),maxWavelength=np.max(w1),
                  nO2=nO2_profile_full[ith]*nO2Scale_profile_full[ith],
                  einsteinA=einsteinA)
        l.getAbsorption(nu=nu)
        # this is extinction (o2 number density x o2 absorption cross section)
        sigma_[ith,] = l.sigma*l.nO2
        # this is d(extinction)/dnO2 for non-emitting layers
        dsigmadnO2_[ith,] = l.sigma
        # this is d(extinction)/dT
        dsigmadT_[ith,] = l.dsigmadT*l.nO2
        l.getAirglowEmission(nO2s=nO2s_profile[ith])
        # emissivity
        emission_[ith,] = l.emission
        # d(emissivity)/dT
        dedT_[ith,] = l.dedT
        # d(emissivity)/dnO2s
        dednO2s_[ith,] = l.dednO2s
    
    # high res attenuated emission at each tangent height
    obs_R1 = np.zeros(emission_.shape)
    # high res jacobians to layer temperature
    obs_dR1dT = np.zeros(emission_.shape+(nth,))
    # high res jacobians to layer excited o2 density
    obs_dR1dnO2s = np.zeros(emission_.shape+(nth,))
    # high res jacobians to layer o2 density
    obs_dR1dnO2 = np.zeros(emission_.shape+(nth,))
    # low res radiance at each tangent height
    obs_R2 = np.zeros((nth,nw2))
    # jacobian to ils hw1e
    obs_dR2dHW1E = np.zeros((nth,nw2))
    # jacobian to wavelength shift
    obs_dR2dw_shfit = np.zeros((nth,nw2))
    # low res jacobians to layer temperature
    obs_dR2dT = np.zeros((nth,nw2,nth))
    # low res jacobians to layer excited o2 density
    obs_dR2dnO2s = np.zeros((nth,nw2,nth))
    # low res jacobians to layer o2 density scaling factor
    obs_dR2dnO2Scale = np.zeros((nth,nw2,nth))
    
    for i in range(nth):# observation at tangent height i
        # piercing through the onion from close side to far side
        absorbing_layer_idx = np.hstack((np.arange(nth-1,i-1,-1),np.arange(i,nth-1)))
        emitting_layer_idx = np.hstack((np.arange(nth-1,i-1,-1),np.arange(i,nth)))
        cum_tau = np.cumsum(np.array([sigma_[k,]*L[i,k] for k in absorbing_layer_idx]),axis=0)
        emitting_layer_tau = np.array([-np.log((1-np.exp(-sigma_[k,]*L[i,k]))/(sigma_[k,]*L[i,k])) 
                                       for k in emitting_layer_idx])
        # d(emitting layer tau)/dT = d(emitting layer tau)/d(extinction) * d(extinction)/dT
        detau_dT = np.array([(-(sigma_[k,]*L[i,k]*np.exp(-sigma_[k,]*L[i,k])+np.exp(-sigma_[k,]*L[i,k])-1)\
                              /(sigma_[k,]*(1-np.exp(-sigma_[k,]*L[i,k]))))\
                            *dsigmadT_[k,] for k in emitting_layer_idx])
        # d(emitting layer tau)/dnO2
        detau_dnO2 = np.array([(-(sigma_[k,]*L[i,k]*np.exp(-sigma_[k,]*L[i,k])+np.exp(-sigma_[k,]*L[i,k])-1)\
                              /(nO2_profile_full[k]*nO2Scale_profile_full[k]*(1-np.exp(-sigma_[k,]*L[i,k]))))\
                               for k in emitting_layer_idx])
        
        for (count,j) in enumerate(emitting_layer_idx):
            if count == 0:
                total_tau = emitting_layer_tau[count,]
            else:
                # downstream layers' tau+emitting layer tau
                total_tau = cum_tau[count-1,]+emitting_layer_tau[count,]
            
            # radiance
            obs_R1[i,] = obs_R1[i,]+emission_[j,]/(4*np.pi)*L[i,j]*np.exp(-total_tau)
            
            # d(radiance)/d(emitting layer nO2s)
            obs_dR1dnO2s[i,:,j] = obs_dR1dnO2s[i,:,j]+dednO2s_[j,]/(4*np.pi)*L[i,j]*np.exp(-total_tau)
            
            # d(radiance)/d(emitting layer temperature), without accounting for upstream layers
            obs_dR1dT[i,:,j] = obs_dR1dT[i,:,j]+dedT_[j,]/(4*np.pi)*L[i,j]*np.exp(-total_tau)\
            +emission_[j,]/(4*np.pi)*L[i,j]*np.exp(-total_tau)*(-detau_dT[count,])

            # d(radiance)/d(emitting layer nO2), without accounting for upstream layers
            obs_dR1dnO2[i,:,j] = obs_dR1dnO2[i,:,j]\
            +emission_[j,]/(4*np.pi)*L[i,j]*np.exp(-total_tau)*(-detau_dnO2[count,])

            # loop over upstream layers
            for (count1,k) in enumerate(emitting_layer_idx[count+1:]):
                total_tau = cum_tau[count+count1,]+emitting_layer_tau[count+count1+1,]
                # d(radiance)/d(emitting layer temperature) due to altered emission from upstream layers
                obs_dR1dT[i,:,j] = obs_dR1dT[i,:,j]\
                +emission_[k,]/(4*np.pi)*L[i,k]*np.exp(-total_tau)*(-L[i,j]*dsigmadT_[j,])
                # d(radiance)/d(emitting layer nO2) due to altered emission from upstream layers
                obs_dR1dnO2[i,:,j] = obs_dR1dnO2[i,:,j]\
                +emission_[k,]/(4*np.pi)*L[i,k]*np.exp(-total_tau)*(-L[i,j]*dsigmadnO2_[j,])
        
        R1_oversampled_fft = convolve_fft(obs_R1[i,::-1],ILS,normalize_kernel=True)
        
        dR1dHW1E_oversampled_fft = convolve_fft(obs_R1[i,::-1],dILSdHW1E,normalize_kernel=False)
        bspline_coef = splrep(w1[::-1],R1_oversampled_fft)
        obs_R2[i,] = splev(wavelength[i,]+w_shift,bspline_coef)
        obs_dR2dw_shfit[i,] = splev(wavelength[i,]+w_shift,bspline_coef,der=1)
        
        bspline_coef = splrep(w1[::-1],dR1dHW1E_oversampled_fft)
        obs_dR2dHW1E[i,] = splev(wavelength[i,]+w_shift,bspline_coef)
        
        for j in range(i,nth):
            dR1dT_oversampled_fft = convolve_fft(obs_dR1dT[i,::-1,j],ILS,normalize_kernel=True)
            dR1dnO2s_oversampled_fft = convolve_fft(obs_dR1dnO2s[i,::-1,j],ILS,normalize_kernel=True)
            bspline_coef = splrep(w1[::-1],dR1dT_oversampled_fft)
            obs_dR2dT[i,:,j] = splev(wavelength[i,]+w_shift,bspline_coef)
            bspline_coef = splrep(w1[::-1],dR1dnO2s_oversampled_fft)
            obs_dR2dnO2s[i,:,j] = splev(wavelength[i,]+w_shift,bspline_coef)
            dR1dnO2Scale_oversampled_fft = convolve_fft(obs_dR1dnO2[i,::-1,j]*nO2_profile_full[j],ILS,normalize_kernel=True)
            bspline_coef = splrep(w1[::-1],dR1dnO2Scale_oversampled_fft)
            obs_dR2dnO2Scale[i,:,j] = splev(wavelength[i,]+w_shift,bspline_coef)
    obs_dR2dnO2Scale = obs_dR2dnO2Scale[:,:,0:n_nO2]
    jacobians = OrderedDict()
    jacobians['nO2s_profile'] = obs_dR2dnO2s.reshape(-1,nth)
    jacobians['T_profile'] = obs_dR2dT.reshape(-1,nth)
    jacobians['HW1E'] = obs_dR2dHW1E.reshape(-1,1)
    jacobians['w_shift'] = obs_dR2dw_shfit.reshape(-1,1)
    jacobians['nO2Scale_profile'] = obs_dR2dnO2Scale.reshape(-1,n_nO2)
    return obs_R2,jacobians

def F_airglow_forward_model_no_absorption(w1,wavelength,L,p_profile,
                                          nO2s_profile,T_profile,HW1E,w_shift,T_profile_reference=None,
                                          nu=None,einsteinA=None,nO2_profile=None):
    '''
    forward model to simulate scia-observed limb spectra for a profile scan
    output jacobians
    '''
    nu = nu or 1e7/w1
    T_profile_reference = T_profile_reference or T_profile.copy()
    dw1 = np.abs(np.median(np.diff(w1)))
    ndx = np.ceil(HW1E*3/dw1);
    xx = np.arange(ndx*2)*dw1-ndx*dw1;
    ILS = 1/np.sqrt(np.pi)/HW1E*np.exp(-np.power(xx/HW1E,2))*dw1
    dILSdHW1E = 1/np.sqrt(np.pi)*(-1/np.power(HW1E,2)+2*np.power(xx,2)/np.power(HW1E,4))*np.exp(-np.power(xx/HW1E,2))*dw1
    nw1 = len(nu)
    nw2 = wavelength.shape[1]
    nth = len(p_profile)
    sigma_ = np.zeros((nth,nw1))
    emission_ = np.zeros((nth,nw1))
    dedT_ = np.zeros((nth,nw1))
    dednO2s_ = np.zeros((nth,nw1))
    if nO2_profile is None:
        nO2_profile = [None]*nth
    # get xsection (sigma_), emission, and jacobians of emission for each layer
    for ith in range(nth):
        l = layer(p=p_profile[ith],T=T_profile[ith],
                  minWavelength=np.min(w1),maxWavelength=np.max(w1),einsteinA=einsteinA,nO2=nO2_profile[ith])
        l.getAbsorption(nu=nu)
        sigma_[ith,] = l.sigma*l.nO2# this is optical depth divided by length
        l.getAirglowEmission(nO2s=nO2s_profile[ith])
        emission_[ith,] = l.emission
        dedT_[ith,] = l.dedT
        dednO2s_[ith,] = l.dednO2s
    
    # high res attenuated emission at each tangent height
    obs_R1 = np.zeros(emission_.shape)
    # high res jacobians to layer temperature
    obs_dR1dT = np.zeros(emission_.shape+(nth,))
    # high res jacobians to layer excited o2 density
    obs_dR1dnO2s = np.zeros(emission_.shape+(nth,))
    # low res radiance at each tangent height
    obs_R2 = np.zeros((nth,nw2))
    # jacobian to ils hw1e
    obs_dR2dHW1E = np.zeros((nth,nw2))
    # jacobian to wavelength shift
    obs_dR2dw_shfit = np.zeros((nth,nw2))
    # low res jacobians to layer temperature
    obs_dR2dT = np.zeros((nth,nw2,nth))
    # low res jacobians to layer excited o2 density
    obs_dR2dnO2s = np.zeros((nth,nw2,nth))
    
    for i in range(nth):# observation at tangent height i
        # piercing through the onion from close side to far side
        emitting_layer_idx = np.hstack((np.arange(nth-1,i-1,-1),np.arange(i,nth)))
        for (count,j) in enumerate(emitting_layer_idx):
            obs_R1[i,] = obs_R1[i,]+emission_[j,]/(4*np.pi)*L[i,j]# no absorption for the closest shell
            obs_dR1dT[i,:,j] = obs_dR1dT[i,:,j]+dedT_[j,]/(4*np.pi)*L[i,j]
            obs_dR1dnO2s[i,:,j] = obs_dR1dnO2s[i,:,j]+dednO2s_[j,]/(4*np.pi)*L[i,j]
        R1_oversampled_fft = convolve_fft(obs_R1[i,::-1],ILS,normalize_kernel=True)
        
        dR1dHW1E_oversampled_fft = convolve_fft(obs_R1[i,::-1],dILSdHW1E,normalize_kernel=False)
        bspline_coef = splrep(w1[::-1],R1_oversampled_fft)
        obs_R2[i,] = splev(wavelength[i,]+w_shift,bspline_coef)
        obs_dR2dw_shfit[i,] = splev(wavelength[i,]+w_shift,bspline_coef,der=1)
        
        bspline_coef = splrep(w1[::-1],dR1dHW1E_oversampled_fft)
        obs_dR2dHW1E[i,] = splev(wavelength[i,]+w_shift,bspline_coef)
        
        for j in range(i,nth):
            dR1dT_oversampled_fft = convolve_fft(obs_dR1dT[i,::-1,j],ILS,normalize_kernel=True)
            dR1dnO2s_oversampled_fft = convolve_fft(obs_dR1dnO2s[i,::-1,j],ILS,normalize_kernel=True)
            bspline_coef = splrep(w1[::-1],dR1dT_oversampled_fft)
            obs_dR2dT[i,:,j] = splev(wavelength[i,]+w_shift,bspline_coef)
            bspline_coef = splrep(w1[::-1],dR1dnO2s_oversampled_fft)
            obs_dR2dnO2s[i,:,j] = splev(wavelength[i,]+w_shift,bspline_coef)
    jacobians = OrderedDict()
    jacobians['nO2s_profile'] = obs_dR2dnO2s.reshape(-1,nth)
    jacobians['T_profile'] = obs_dR2dT.reshape(-1,nth)
    jacobians['HW1E'] = obs_dR2dHW1E.reshape(-1,1)
    jacobians['w_shift'] = obs_dR2dw_shfit.reshape(-1,1)
    return obs_R2,jacobians

class Parameter:
    def __init__(self, name,prior=None,value=None,prior_error=None,
                 p_profile=None,correlation_scaleHeight=None,
                 vmin=-np.inf,vmax=np.inf,vary=True):
        self.name = name
        self.prior = prior
        self.vmin = vmin
        self.vmax = vmax
        self.vary = vary
        if value is None:
            value = prior
        self.value = value
        if np.isscalar(prior):
            self.prior_error_matrix = prior_error**2
            self.nstate = 1
            return
        self.nstate = len(prior)
        # prevent zero profile prior error
        mask = prior_error == 0
        prior_error[mask] = np.min(prior_error[~mask])
        self.prior_error_matrix = np.diag(prior_error**2)
        if correlation_scaleHeight is not None and p_profile is not None:
            log_p_profile = np.log(p_profile)
            for (i,logp1) in enumerate(log_p_profile):
                for (j,logp2) in enumerate(log_p_profile):
                    if i == j:
                        continue
                    self.prior_error_matrix[i,j] = prior_error[i]*prior_error[j]*np.exp(-np.abs(logp1-logp2)/correlation_scaleHeight)
            
class Parameters(OrderedDict):
    def __init__(self):
        pass
    
    def add(self,param):
        OrderedDict.__setitem__(self,param.name,param)
    
    def flatten_values(self,field_to_flatten):
        
#        nstates = np.sum(np.array([par.nstate for (name,par) in self.items()]))
        nstates = 0
        for (name,par) in self.items():
            if not par.vary:
                continue
            nstates = nstates+par.nstate
        Sa = np.zeros((nstates,nstates))
        count = 0
        beta0 = np.zeros(nstates)
        params_names = []
        for (name,par) in self.items():
            if not par.vary:
                continue
            params_names.append(name)
            Sa[count:count+par.nstate,count:count+par.nstate] = par.prior_error_matrix
            beta0[count:count+par.nstate] = getattr(par,field_to_flatten)
            count = count+par.nstate
        return beta0, Sa, nstates, params_names
    
    def update_vectors(self,vector_name,vector):
        count = 0
        for (name,par) in self.items():
            if not par.vary:
                continue
            new_values = vector[count:count+par.nstate]
            if vector_name == 'value':
                new_values[new_values<par.vmin] = par.vmin
                new_values[new_values>par.vmax] = par.vmax
            setattr(self[name],vector_name,new_values)
            count = count+par.nstate
    
    def update_matrices(self,matrix_name,matrix):
        count = 0
        for (name,par) in self.items():
            setattr(self[name],matrix_name,matrix[count:count+par.nstate,count:count+par.nstate])
            count = count+par.nstate

class Retrieval_Results(object):
    def __init__(self):
        pass
    
    def plot_radiances(self):
        from matplotlib.collections import PolyCollection
        fig,ax = plt.subplots(2,1,constrained_layout=True,figsize=(9,5),sharex=True)
        ax_y = ax[0]#fig.add_subplot(gs[0,0:2])
        ax_r = ax[1]#fig.add_subplot(gs[1,0:2])
#         mngr = plt.get_current_fig_manager()
#         geom = mngr.window.geometry()
#         x,y,dx,dy = geom.getRect()
#         mngr.window.setGeometry(0,100,dx,dy)
        yy = self.yy
        yhat = self.yhat
        y0 = self.y0
        nth = self.nth
        nw2 = self.nw2
        xx = np.arange(len(yy))
        ax_y.plot(xx,yy,'ok',xx,y0,'-b',xx,yhat,'-r')
        ax_y.set_xlim([0,len(yy)])
        ax_y.set_ylabel('Radiance')
        verts = []
        ylim = ax_y.get_ylim()
        ys = np.array([ylim[0],ylim[1],ylim[1],ylim[0]])
        for ith in range(nth):
            x1 = ith*nw2;x2=ith*nw2+nw2
            xs = np.array([x1,x1,x2,x2])
            verts.append(list(zip(xs,ys)))
            collection = PolyCollection(verts,
                            array=self.tangent_height,cmap='rainbow_r',edgecolors='none',alpha=0.65)
        ax_y.add_collection(collection)
        for ith in range(nth):
            ax_y.text(ith*nw2+nw2/2,ylim[1]*0.9,'{:.1f} km'.format(self.tangent_height[ith]),
             horizontalalignment='center',fontsize=16,zorder=1)
        ax_y.legend(['Observation','Prior','Posterior'],fontsize=14,loc='center right')
        
        ax_r.plot(xx,yy-yhat,'-ok')
        verts = []
        ylim = ax_r.get_ylim()
        ys = np.array([ylim[0],ylim[1],ylim[1],ylim[0]])
        for ith in range(nth):
            x1 = ith*nw2;x2=ith*nw2+nw2
            xs = np.array([x1,x1,x2,x2])
            verts.append(list(zip(xs,ys)))
            collection = PolyCollection(verts,
                            array=self.tangent_height,cmap='rainbow_r',edgecolors='none',alpha=0.65)
            ax_r.add_collection(collection)
            ax_r.set_ylabel('Residual radiance')
        ax_r.set_title(r'$\chi^2$={:.3f}'.format(self.chi2))

class Forward_Model(object):
    def __init__(self,func,independent_vars,param_names):
        
        self.logger = logging.getLogger(__name__)
        self.logger.info('creating an instance of Forward_Model')
        self.func = func
        self.independent_vars = independent_vars
        self._param_names = param_names
        self._func_allargs = []
        self._func_haskeywords = False
        self.param_hints = OrderedDict()
        self._parse_params()
    
    def param_names(self):
        """Return the parameter names of the Model."""
        return self._param_names
    
    def _parse_params(self):
        """Build parameters from function arguments."""
        pos_args = []
        kw_args = {}
        keywords_ = None
        sig = inspect.signature(self.func)
        for fnam, fpar in sig.parameters.items():
            if fpar.kind == fpar.VAR_KEYWORD:
                keywords_ = fnam
            elif fpar.kind == fpar.POSITIONAL_OR_KEYWORD:
                if fpar.default == fpar.empty:
                    pos_args.append(fnam)
                else:
                    kw_args[fnam] = fpar.default
            elif fpar.kind == fpar.VAR_POSITIONAL:
                raise ValueError("varargs '*%s' is not supported" % fnam)
        # inspection done

        self._func_haskeywords = keywords_ is not None
        self._func_allargs = pos_args + list(kw_args.keys())
        
    def make_funcargs(self, params=None, kwargs=None):
        """Convert parameter values and keywords to function arguments."""
        if params is None:
            params = {}
        if kwargs is None:
            kwargs = {}
        out = {}
        for name, par in params.items():
            if name in self._func_allargs or self._func_haskeywords:
                out[name] = par.value

        # kwargs handled slightly differently -- may set param value too!
        for name, val in kwargs.items():
            if name in self._func_allargs or self._func_haskeywords:
                out[name] = val
        return out
    
    def evaluate(self,params=None,**kwargs):
        return self.func(**self.make_funcargs(params, kwargs))
    
    def set_prior(self,name,**kwargs):
        if name not in self.param_hints:
            self.param_hints[name] = OrderedDict()

        for key, val in kwargs.items():
            self.param_hints[name][key] = val
    
    def make_params(self):
        params = Parameters()
        for name in self.param_names():
            par = Parameter(name,**self.param_hints[name])
            params.add(par)
        return params
    
    def retrieve(self,radiance,radiance_error,
                 params=None,max_iter=100,use_LM=False,
                 max_diverging_step=5,converge_criterion_scale=1,**kwargs):
        
        if params is None:
            params = self.make_params()
        nth = radiance.shape[0]
        nw2 = radiance.shape[1]
        beta0, Sa, nstates, params_names = params.flatten_values(field_to_flatten='prior')
        beta = beta0.copy()
        
        Sa_inv = np.linalg.inv(Sa)
        yy = radiance.ravel()
        Sy = np.diag(radiance_error.ravel()**2)
        Sy_inv = np.diag(1/radiance_error.ravel()**2)
        count = 0
        count_div = 0
        dsigma2 = np.inf
        result = Retrieval_Results()
        result.if_success = True
        if use_LM:
            LM_gamma = 10.
        while(dsigma2 > nstates*converge_criterion_scale and count < max_iter):
            self.logger.info('Iteration {}'.format(count))
            if count != 0:
                params.update_vectors(vector_name='value',vector=beta)
            obs_R2,jacobians = self.evaluate(params,**kwargs)
            yhat = obs_R2.ravel()
            all_jacobians = [jacobians[name] for name in params_names]
            K = np.column_stack(all_jacobians)
            if use_LM:
                self.logger.info('gamma = {}'.format(LM_gamma))
                dbeta = np.linalg.inv((1+LM_gamma)*Sa_inv+K.T@Sy_inv@K)@(K.T@Sy_inv@(yy-yhat)-Sa_inv@(beta-beta0))
                beta_try = beta+dbeta
                c_i = (yy-yhat).T@Sy_inv@(yy-yhat)+(beta-beta0).T@Sa_inv@(beta-beta0)
                params.update_vectors(vector_name='value',vector=beta_try)
                obs_R2_try,_ = self.evaluate(params,**kwargs)
                yhat_try = obs_R2_try.ravel()
                c_in1 = (yy-yhat_try).T@Sy_inv@(yy-yhat_try)+(beta_try-beta0).T@Sa_inv@(beta_try-beta0)
                yhat_linear = yhat+K@dbeta
                c_in1_FC = (yy-yhat_linear).T@Sy_inv@(yy-yhat_linear)+(beta_try-beta0).T@Sa_inv@(beta_try-beta0)
                LM_R = (c_i-c_in1)/(c_i-c_in1_FC)
                self.logger.info('R = {:.3f}'.format(LM_R))
                if LM_R <=0.0001:
                    LM_gamma = LM_gamma*10
                    self.logger.warning('R = {:.3f} and is diverging. Abandon step and increase gamma by 10'.format(LM_R))
                    count_div += 1
                    self.logger.info('{} diverging steps'.format(count_div))
                    if count_div >= max_diverging_step:
                        self.logger.warning('too many diverging steps, abandon retrieval')
                        result.if_success = False
                        break
                    continue
                elif LM_R < 0.25:
                    LM_gamma = LM_gamma*10
                elif LM_R < 0.75:
                    pass
                else:
                    LM_gamma = LM_gamma/2
            else:
                dbeta = np.linalg.inv(Sa_inv+K.T@Sy_inv@K)@(K.T@Sy_inv@(yy-yhat)-Sa_inv@(beta-beta0))
            dsigma2 = dbeta.T@(K.T@Sy_inv@(yy-yhat)+Sa_inv@(beta-beta0))
            self.logger.info('dsigma2: {}'.format(dsigma2))
            self.logger.debug(' '.join('{:2E}'.format(b) for b in beta))
            beta = beta+dbeta
            if count == 0:
                result.y0 = obs_R2.ravel()
                result.wavelength = kwargs['wavelength']
                result.Sy = Sy
                result.Sa = Sa
                result.beta0 = beta0
            count = count+1
#        params.update_vectors(vector_name='value',vector=beta)
        result.nth = nth
        result.nw2 = nw2
        result.yy = yy
        result.yhat = yhat
        result.niter = count
        beta = beta-dbeta
        result.beta = beta
        result.Jprior = (beta-beta0).T@Sa_inv@(beta-beta0)
        result.max_iter = max_iter
        if result.niter == max_iter:
            result.if_success = False
        result.chi2 = np.sum(np.power(yy-yhat,2))/np.trace(Sy)
        result.rmse = np.sqrt(np.mean(np.power(yy-yhat,2)))
        Shat = np.linalg.inv(K.T@Sy_inv@K+Sa_inv)
        result.Shat = Shat
        AVK = Shat@K.T@Sy_inv@K
        result.AVK = AVK
        params.update_matrices(matrix_name='posterior_error_matrix',matrix=Shat)
        params.update_vectors(vector_name='posterior_error',vector=np.sqrt(np.diag(Shat)))
        params.update_matrices(matrix_name='averaging_kernel',matrix=AVK)
        params.update_vectors(vector_name='dofs',vector=np.diag(AVK))
        result.params = params
        return result

def F_sample_standard_atm(tangent_height):
    '''
    sample prior information at tangent height using US standard atmosphere
    '''
    from scipy.interpolate import interp1d
    z_grid = np.array([20,21,22,23,24,25,27.5,30,32.5,35,37.5,40,42.5,45,47.5,50,55,60,65,70,75,80,85,	90,95,100,105,110,115,120])
    p_grid = np.array([55.3,47.3,40.5,34.7,29.7,25.5,17.4,12,8.01,5.75,4.15,2.87114,2.06,1.49,1.09,0.798,0.425,0.219,0.109,0.0522,0.024,0.0105,0.00446,0.00184,0.00076,0.00032,0.000145,7.10E-05,4.01E-05,2.54E-05])
    p_grid = p_grid*100#hPa to Pa
    T_grid = np.array([216.7,217.6,218.6,219.6,220.6,221.6,224,226.5,230,236.5,242.9,250.4,257.3,264.2,270.6,270.7,260.8,247,233.3,219.6,208.4,198.6,188.9,186.9,188.4,195.1,208.8,240,300,360])
    f = interp1d(z_grid,p_grid,fill_value='extrapolate')
    p_profile = f(tangent_height)
    f = interp1d(z_grid,T_grid,fill_value='extrapolate')
    T_profile = f(tangent_height)
    return p_profile, T_profile

def F_msis_atm(tangent_height,latitude,longitude,time):
    '''
    atmospheric information from MSIS model
    '''
    import msise00
    # pooor data type handling
    latitude = latitude.astype(float)
    longitude = longitude.astype(float)
    ref_dt = dt.datetime(2000,1,1,0,0,0)
    p_profile = np.zeros_like(tangent_height)
    T_profile = np.zeros_like(tangent_height)
    nO2_profile = np.zeros_like(tangent_height)
    # avoid printing msise00 messages at INFO level
    clevel = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)
    for ith in range(len(tangent_height)):
        atmos = msise00.run(time=ref_dt+dt.timedelta(seconds=time[ith]),altkm=tangent_height[ith],glat=latitude[ith],glon=longitude[ith])
        T_profile[ith] = np.array(atmos['Tn'].squeeze())
        total_density = np.array(atmos['Total'].squeeze())#kg/m3
        p_profile[ith] = total_density*287.058*T_profile[ith]
        nO2_profile[ith] = np.array(atmos['O2'].squeeze())*1e-6#molec/cm3
    logging.getLogger().setLevel(clevel)
    return p_profile, T_profile, nO2_profile

def F_th_to_Hlayer(th):
    dth = np.nanmean(np.diff(th))
    Hlayer = th+dth/2
    return Hlayer
def F_th_to_Hlevel(th):
    dth = np.nanmean(np.diff(th))
    Hlevel = np.append(th,th[-1]+dth)
    return Hlevel
def F_T_prior_error(h,dT_up=30,dT_low=10,h_divide=50,lh=2.5):
    '''
    generate prior error of temperature as a function of h in km
    '''
    T_prior = dT_low+(dT_up-dT_low)/(1+np.exp(-(h-h_divide)/lh))
    return T_prior
def F_fit_profile(tangent_height,radiance,radiance_error,wavelength,
                  startWavelength=1240,endWavelength=1300,
                  minTH=None,maxTH=None,w1_step=None,einsteinA=None,
                  if_attenuation=True,n_nO2=None,nO2Scale_error=0.5,
                  max_iter=10,msis_pt=True,time=None,latitude=None,longitude=None,
                  dT_up=30,dT_low=10,h_divide=50,lh=2.5,
                  use_LM=True,converge_criterion_scale=1):
    '''
    call this function to fit a collection of sciamachy tangent heights
    '''
    if time is None:
        logging.warning('Time is needed to use MSIS!')
        msis_pt = False
    if latitude is None or longitude is None:
        logging.warning('lat/lon is needed to use MSIS!')
        msis_pt = False
    if endWavelength < 800:
        HW1E_prior = 0.3
        minTH = minTH or 50
        maxTH = maxTH or 120
        w1_step = w1_step or -0.0002
        einsteinA = einsteinA or 0.08693#10.1016/j.jqsrt.2010.05.011
    
    if startWavelength > 1100:
        HW1E_prior = 0.8
        minTH = minTH or 25
        maxTH = maxTH or 120
        w1_step = w1_step or -0.001
        einsteinA = einsteinA or 2.27e-4
    
    th_idx = np.argsort(tangent_height)
    tangent_height = np.sort(tangent_height)# TH has to go from low to high
    radiance = radiance[th_idx,:]
    radiance_error = radiance_error[th_idx,:]
    waveMask = (np.mean(wavelength,axis=0) >= startWavelength) & (np.mean(wavelength,axis=0) <= endWavelength) & (~np.isnan(np.mean(radiance,axis=0)))
    THMask = (~np.isnan(tangent_height)) & (tangent_height < maxTH) & (tangent_height > minTH)
    tangent_height = tangent_height[THMask]
    wavelength = wavelength[np.ix_(THMask,waveMask)]
    radiance = radiance[np.ix_(THMask,waveMask)]
    radiance_error = radiance_error[np.ix_(THMask,waveMask)]   
    dZ = np.abs(np.diff(tangent_height))
    dZ = np.append(dZ,dZ[-1])    
    nth = len(tangent_height)   
    L = np.zeros((nth,nth))
    Re=6371.
    for i in range(nth):       
        for j in range(i,nth):
            if j == nth-1:
                topTH = tangent_height[j]+np.abs(tangent_height[j]-tangent_height[j-1])
            else:
                topTH = tangent_height[j+1]
            L[i,j] = np.sqrt(np.power(topTH+Re,2)-np.power(tangent_height[i]+Re,2))-\
            np.sqrt(np.power(tangent_height[j]+Re,2)-np.power(tangent_height[i]+Re,2))
    L = L*1e5# km to cm
    So = np.diag(radiance_error[:,0])
    rg = 4*np.pi*np.linalg.inv(L.T@np.linalg.inv(So)@L)@L.T@np.linalg.inv(So)@radiance
    nO2s_profile = np.trapz(rg,wavelength)/einsteinA
    nO2s_profile[nO2s_profile < 0] = 0
    if msis_pt:
        try: 
            p_profile, T_profile_msis, nO2_profile = F_msis_atm(F_th_to_Hlayer(tangent_height),latitude,longitude,time)
            _, T_profile = F_sample_standard_atm(F_th_to_Hlayer(tangent_height))# T_profile is constant a priori from standar atmosphere
        except:
            logging.warning('MSIS doesn''t work!')
            logging.warning("MSIS error message:", sys.exc_info())
            msis_pt = False
    if msis_pt:
        if any(np.isnan(np.concatenate((p_profile,T_profile_msis,nO2_profile)))):
            logging.warning('MSIS contains nan values! Use standard atmosphere')
            msis_pt = False
    if not msis_pt:
        logging.info('Use standard atmosphere for p, T, nO2.')
        p_profile, T_profile = F_sample_standard_atm(tangent_height)
        nO2_profile = p_profile/T_profile/1.38065e-23*0.2095*1e-6# O2 number density in molecules/cm3, ideal gas law
    # this if statement makes msis temperature as prior
    if msis_pt:
        T_profile = T_profile_msis.copy()
        # del T_profile_msis
    # w1 is the high res wavelength grid. has to be descending
    w1 = arange_(endWavelength,startWavelength,-np.abs(w1_step))#-0.0005
    #T_profile_e = np.ones(T_profile.shape)*20
    #T_profile_e[tangent_height<50] = 2
    T_profile_e = F_T_prior_error(F_th_to_Hlayer(tangent_height),dT_up,dT_low,h_divide,lh)
    #nO2s_profile_e = np.ones(nO2s_profile.shape)*nO2s_profile
    #nO2s_profile_e[nO2s_profile_e<0.1*np.mean(nO2s_profile)] = 0.1*np.mean(nO2s_profile)
    nO2s_profile = np.ones_like(nO2s_profile)*np.nanmean(nO2s_profile)
    nO2s_profile_e = nO2s_profile*100# no effective prior constraint on nO2s profile
    
    n_nO2 = n_nO2 or len(T_profile)
    if n_nO2 > len(T_profile):
        # cap number of free O2 layers to the max
        n_nO2 = len(T_profile)
    if not if_attenuation:
        aOE = Forward_Model(func=F_airglow_forward_model_no_absorption,
                                  independent_vars=['w1','wavelength','L','p_profile','nu','einsteinA','nO2_profile'],
                                  param_names=['nO2s_profile','T_profile','HW1E','w_shift'])
        aOE.set_prior('nO2s_profile',prior=nO2s_profile,prior_error=nO2s_profile_e,p_profile=p_profile,correlation_scaleHeight=1)
        aOE.set_prior('T_profile',prior=T_profile,prior_error=T_profile_e,p_profile=p_profile,correlation_scaleHeight=1,vmin=50,vmax=500)
        aOE.set_prior('HW1E',prior=HW1E_prior,prior_error=HW1E_prior/2)
        aOE.set_prior('w_shift',prior=0.,prior_error=1.)
        result = aOE.retrieve(radiance,radiance_error,max_iter=max_iter,use_LM=use_LM,
                              w1=w1,wavelength=wavelength,
                              L=L,p_profile=p_profile,einsteinA=einsteinA,nO2_profile=nO2_profile,
                              converge_criterion_scale=converge_criterion_scale)
        result.tangent_height = tangent_height
        result.THMask = THMask
        result.dZ = dZ
        result.p_profile_middle = p_profile_middle
        result.p_profile = p_profile
        result.T_profile_prior = T_profile
        if 'T_profile_msis' in locals().keys():
            result.T_profile_msis = T_profile_msis
        result.nO2s_profile_prior = nO2s_profile
        result.nO2_profile= nO2_profile
        return result
    
    if n_nO2 == 0:
        aOE = Forward_Model(func=F_airglow_forward_model,
                                  independent_vars=['w1','wavelength','L','p_profile','nu','einsteinA','nO2_profile'],
                                  param_names=['nO2s_profile','T_profile','HW1E','w_shift'])
        aOE.set_prior('nO2s_profile',prior=nO2s_profile,prior_error=nO2s_profile_e,p_profile=p_profile,correlation_scaleHeight=1)
        aOE.set_prior('T_profile',prior=T_profile,prior_error=T_profile_e,p_profile=p_profile,correlation_scaleHeight=1,vmin=50,vmax=500)
        aOE.set_prior('HW1E',prior=HW1E_prior,prior_error=HW1E_prior/2)
        aOE.set_prior('w_shift',prior=0.,prior_error=1.)
        result = aOE.retrieve(radiance,radiance_error,max_iter=max_iter,use_LM=use_LM,
                              w1=w1,wavelength=wavelength,
                              L=L,p_profile=p_profile,einsteinA=einsteinA,nO2_profile=nO2_profile,
                              converge_criterion_scale=converge_criterion_scale)
    else:
        nO2Scale_profile = np.ones(n_nO2)
        nO2Scale_profile_e = np.ones(n_nO2)*nO2Scale_error
        aOE = Forward_Model(func=F_airglow_forward_model_nO2Scale,
                                  independent_vars=['w1','wavelength','L','p_profile','nu','T_profile_reference','einsteinA','nO2_profile'],
                                  param_names=['nO2s_profile','T_profile','HW1E','w_shift','nO2Scale_profile'])
        aOE.set_prior('nO2s_profile',prior=nO2s_profile,prior_error=nO2s_profile_e,p_profile=p_profile,correlation_scaleHeight=1,vmin=0)
        aOE.set_prior('T_profile',prior=T_profile,prior_error=T_profile_e,p_profile=p_profile,correlation_scaleHeight=1,vmin=50,vmax=500)
        aOE.set_prior('HW1E',prior=HW1E_prior,prior_error=HW1E_prior/2)
        aOE.set_prior('w_shift',prior=0.,prior_error=1.)
        aOE.set_prior('nO2Scale_profile',prior=nO2Scale_profile,prior_error=nO2Scale_profile_e,
                      p_profile=p_profile[0:len(nO2Scale_profile)],correlation_scaleHeight=1,vmin=0,vary=True)
        result = aOE.retrieve(radiance,radiance_error,max_iter=max_iter,use_LM=use_LM,
                              w1=w1,wavelength=wavelength,
                              L=L,p_profile=p_profile,einsteinA=einsteinA,nO2_profile=nO2_profile,
                              converge_criterion_scale=converge_criterion_scale)
    result.tangent_height = tangent_height
    result.THMask = THMask
    result.dZ = dZ
    result.p_profile = p_profile
    result.T_profile_prior = T_profile
    if 'T_profile_msis' in locals().keys():
        result.T_profile_msis = T_profile_msis
    result.nO2s_profile_prior = nO2s_profile
    result.nO2_profile = nO2_profile
    return result

class Level2_Reader(object):
    def __init__(self,filename):
        '''
        open file
        '''
        self.fid = Dataset(filename)
    
    def load_variable(self,data_fields=[],data_names=[]):
        '''
        load variables as attributes of the Level2_Reader object
        '''
        if len(data_fields) == 0:
            data_fields = ['longitude','latitude','tangent_height','layer_thickness','solar_zenith_angle','time',
                           'singlet_delta/temperature','singlet_delta/temperature_dofs','singlet_delta/temperature_error',
                           'singlet_delta/excited_O2','singlet_delta/excited_O2_dofs','singlet_delta/excited_O2_error',
                           'singlet_sigma/temperature','singlet_sigma/temperature_dofs','singlet_sigma/temperature_error',
                           'singlet_sigma/excited_O2','singlet_sigma/excited_O2_dofs','singlet_sigma/excited_O2_error']
            data_names = ['longitude','latitude','tangent_height','layer_thickness','solar_zenith_time','time',
                           'delta_temperature','delta_temperature_dofs','delta_temperature_error',
                           'delta_excited_O2','delta_excited_O2_dofs','delta_excited_O2_error',
                           'sigma_temperature','sigma_temperature_dofs','sigma_temperature_error',
                           'sigma_excited_O2','sigma_excited_O2_dofs','sigma_excited_O2_error']
        if len(data_names) != len(data_fields):
            data_names = [s.split('/')[-1] for s in data_fields]
        for (i,f) in enumerate(data_fields):
            setattr(self,data_names[i],self.fid[f][:].filled(np.nan))
            if f == 'time':
                time_data = np.array(self.fid['time'][:],dtype=np.float64)
                datetime_data = np.ndarray(shape=time_data.shape,dtype=np.object_)
                for iline in range(time_data.shape[0]):
                    for ift in range(time_data.shape[1]):
                        datetime_data[iline,ift] = dt.datetime(2000,1,1)+dt.timedelta(seconds=time_data[iline,ift])
                setattr(self,'datetime',datetime_data)
    
    def collocate_ACE(self,ace_filename,window_hour=2,window_km=500):
        '''
        collocate ACE-FTS sounding for validation
        '''
        ace_fid = Dataset(ace_filename)
        years = np.array(ace_fid['year'][:].squeeze(),dtype=np.int)
        months = np.array(ace_fid['month'][:].squeeze(),dtype=np.int)
        days = np.array(ace_fid['day'][:].squeeze(),dtype=np.int)
        hours = np.array(ace_fid['hour'][:].squeeze(),dtype=np.float)
        ace_datetime = pd.to_datetime([dt.datetime(years[i],months[i],days[i])+dt.timedelta(hours=hours[i])
            for i in range(len(ace_fid['year'][:]))])
        ace_seconds_since2000 = np.array((ace_datetime-dt.datetime(2000,1,1)).total_seconds())
        # remove most irrelevant data
        window_seconds = window_hour*3600
        time_mask = (ace_seconds_since2000 > self.time.min()-window_seconds) & (ace_seconds_since2000 < self.time.max()+window_seconds)
        ace_lon = ace_fid['longitude'][:].squeeze()[time_mask]
        ace_lat = ace_fid['latitude'][:].squeeze()[time_mask]
        ace_time = ace_seconds_since2000[time_mask]
        ace_temperature = ace_fid['temperature'][:][time_mask,]
        collocation_idx_list = []
        space_mask = np.zeros(len(ace_lat),dtype=np.bool)
        for i in range(len(ace_lon)):
            distance = np.zeros_like(self.latitude[...,0])
            for iline in range(self.latitude.shape[0]):
                for ift in range(self.latitude.shape[1]):
                    distance[iline,ift] = F_distance(self.latitude[iline,ift,0],self.longitude[iline,ift,0],ace_lat[i],ace_lon[i])
            tmp_mask = distance<window_km
            if np.sum(tmp_mask) > 0:
                collocation_idx_list.append(tmp_mask)
                space_mask[i] = True
        self.ace_lon = ace_lon[space_mask]
        self.ace_lat = ace_lat[space_mask]
        self.ace_time = ace_time[space_mask]
        self.ace_temperature = ace_temperature[space_mask,]
        self.ace_collocation_idx_list = collocation_idx_list
        self.ace_altitude = ace_fid['altitude'][:]
        self.ace_fid = ace_fid
        
    def close(self):
        self.fid.close()
        if hasattr(self,'ace_fid'):
            self.ace_fid.close()
    
class Level2_Saver(object):
    def __init__(self):
        pass
    def create(self,filename,longitude,latitude):
        self.filename = filename
        self.along_track_number = longitude.shape[0]
        self.across_track_number = longitude.shape[1]
        self.vertical_number = longitude.shape[2]
        
        self.ncid = Dataset(self.filename,'w',format='NETCDF4')
        self.ncid.createDimension('along_track',self.along_track_number)
        self.ncid.createDimension('across_track',self.across_track_number)
        self.ncid.createDimension('vertical',self.vertical_number)
#        self.ncid.createDimension('corners',4)
        self.lonc = self.ncid.createVariable('longitude',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
        self.lonc.comment = 'longitude at tangent height'
        self.lonc.long_name = 'longitude'
        self.lonc.standard_name = 'longitude'
        self.lonc.units = 'degrees_east'
        self.lonc.valid_min = -180.
        self.lonc.valid_max = 180.
        self.lonc._Storage = 'contiguous'
        
        self.latc = self.ncid.createVariable('latitude',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
        self.latc.comment = 'latitude at tangent height'
        self.latc.long_name = 'latgitude'
        self.latc.standard_name = 'latitude'
        self.latc.units = 'degrees_north'
        self.latc.valid_min = -90.
        self.latc.valid_max = 90.
        self.latc._Storage = 'contiguous'
        
        self.th = self.ncid.createVariable('tangent_height',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
        self.th.comment = 'tangent height'
        self.th.units = 'km'
        self.th._Storage = 'contiguous'
        
        self.dZ = self.ncid.createVariable('layer_thickness',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
        self.dZ.comment = 'tangent layer thickness'
        self.dZ.units = 'km'
        self.dZ._Storage = 'contiguous'
        
        self.sza = self.ncid.createVariable('solar_zenith_angle',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
        self.sza.comment = 'solar zenith angle'
        self.sza.units = 'degree'
        self.sza._Storage = 'contiguous'
        
        self.time = self.ncid.createVariable('time',np.float64,dimensions=('along_track','vertical'),fill_value=-1.0e+30)
        self.time.comment = 'start time of scan phase'
        self.time.units = 's since 2000-01-01'
        self.time._Storage = 'contiguous'
        
        self.ncid.convention = 'CF-1.6'
        self.ncid.Format = 'netCDF-4'
    
    def create_singlet_delta_group(self,group_name,if_save_nO2Scale=True):
        self.ncdelta = self.ncid.createGroup(group_name)
        
        self.d_nO2s = self.ncdelta.createVariable('excited_O2',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
        self.d_nO2s.comment = 'number density of O2 molecules at singlet delta state'
        self.d_nO2s.units = 'molec/cm3'
        self.d_nO2s._Storage = 'contiguous'
        
        self.d_nO2s_dofs = self.ncdelta.createVariable('excited_O2_dofs',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
        self.d_nO2s_dofs.comment = 'degrees of freedom for signal for number density of O2 molecules at singlet delta state'
        self.d_nO2s_dofs.units = ''
        self.d_nO2s_dofs._Storage = 'contiguous'
        
        self.d_nO2s_e = self.ncdelta.createVariable('excited_O2_error',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
        self.d_nO2s_e.comment = 'posterior uncertainty for number density of O2 molecules at singlet delta state'
        self.d_nO2s_e.units = 'molec/cm3'
        self.d_nO2s_e._Storage = 'contiguous'
        
        self.d_T = self.ncdelta.createVariable('temperature',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
        self.d_T.comment = 'temperature'
        self.d_T.units = 'K'
        self.d_T._Storage = 'contiguous'
        
        self.d_T_dofs = self.ncdelta.createVariable('temperature_dofs',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
        self.d_T_dofs.comment = 'degrees of freedom for signal for temperature'
        self.d_T_dofs.units = ''
        self.d_T_dofs._Storage = 'contiguous'
        
        self.d_T_e = self.ncdelta.createVariable('temperature_error',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
        self.d_T_e.comment = 'posterior uncertainty for temperature'
        self.d_T_e.units = 'K'
        self.d_T_e._Storage = 'contiguous'
        
        self.d_T_msis = self.ncdelta.createVariable('temperature_msis',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
        self.d_T_msis.comment = 'temperature from NRLMSISE-00 model'
        self.d_T_msis.units = 'K'
        self.d_T_msis._Storage = 'contiguous'
        
        self.d_nO2_msis = self.ncdelta.createVariable('O2_msis',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
        self.d_nO2_msis.comment = 'number density of O2 molecules'
        self.d_nO2_msis.units = 'molec/cm3'
        self.d_nO2_msis._Storage = 'contiguous'
        
        self.d_HW1E = self.ncdelta.createVariable('HW1E',np.float32,dimensions=('along_track','across_track'),fill_value=-1.0e+30)
        self.d_HW1E.comment = 'half width at 1/e of maximum of slit function'
        self.d_HW1E.units = 'nm'
        self.d_HW1E._Storage = 'contiguous'
        
        self.d_HW1E_dofs = self.ncdelta.createVariable('HW1E_dofs',np.float32,dimensions=('along_track','across_track'),fill_value=-1.0e+30)
        self.d_HW1E_dofs.comment = 'degrees of freedom for signal for half width at 1/e of maximum of slit function'
        self.d_HW1E_dofs.units = ''
        self.d_HW1E_dofs._Storage = 'contiguous'
        
        self.d_HW1E_e = self.ncdelta.createVariable('HW1E_error',np.float32,dimensions=('along_track','across_track'),fill_value=-1.0e+30)
        self.d_HW1E_e.comment = 'posterior error for half width at 1/e of maximum of slit function'
        self.d_HW1E_e.units = ''
        self.d_HW1E_e._Storage = 'contiguous'
        
        self.d_w_shift = self.ncdelta.createVariable('w_shift',np.float32,dimensions=('along_track','across_track'),fill_value=-1.0e+30)
        self.d_w_shift.comment = 'wavelength shift'
        self.d_w_shift.units = 'nm'
        self.d_w_shift._Storage = 'contiguous'
        
        self.d_w_shift_dofs = self.ncdelta.createVariable('w_shift_dofs',np.float32,dimensions=('along_track','across_track'),fill_value=-1.0e+30)
        self.d_w_shift_dofs.comment = 'degrees of freedom for signal for wavelength shift'
        self.d_w_shift_dofs.units = ''
        self.d_w_shift_dofs._Storage = 'contiguous'
        
        self.d_w_shift_e = self.ncdelta.createVariable('w_shift_error',np.float32,dimensions=('along_track','across_track'),fill_value=-1.0e+30)
        self.d_w_shift_e.comment = 'posterior error for wavelength shift'
        self.d_w_shift_e.units = ''
        self.d_w_shift_e._Storage = 'contiguous'
        
        self.d_chi2 = self.ncdelta.createVariable('chi2',np.float32,dimensions=('along_track','across_track'),fill_value=-1.0e+30)
        self.d_chi2.comment = 'goodness of fit indicated by the chi2 value'
        self.d_chi2.units = ''
        self.d_chi2._Storage = 'contiguous'
        
        self.d_rmse = self.ncdelta.createVariable('rmse',np.float32,dimensions=('along_track','across_track'),fill_value=-1.0e+30)
        self.d_rmse.comment = 'goodness of fit indicated by residual root mean square'
        self.d_rmse.units = 'same as radiance'
        self.d_rmse._Storage = 'contiguous'
        
        self.d_if_success = self.ncdelta.createVariable('if_success',np.int8,dimensions=('along_track','across_track'))
        self.d_if_success = 'if retrieve is successful'
        self.d_if_success = ''
        self.d_if_success = 'contiguous'
        
        self.d_Jprior = self.ncdelta.createVariable('Jprior',np.float32,dimensions=('along_track','across_track'),fill_value=-1.0e+30)
        self.d_Jprior.comment = 'distance between solution and prior normalized by prior error'
        self.d_Jprior.units = ''
        self.d_Jprior._Storage = 'contiguous'
        
        self.d_niter = self.ncdelta.createVariable('number_of_iterations',np.int8,dimensions=('along_track','across_track'))
        self.d_niter.comment = 'number of iterations'
        self.d_niter.units = ''
        self.d_niter._Storage = 'contiguous'
        
        if if_save_nO2Scale:
            self.d_nO2Scale = self.ncdelta.createVariable('O2_scaling',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
            self.d_nO2Scale.comment = 'scaling factor for number density of O2 molecules at ground state'
            self.d_nO2Scale.units = ''
            self.d_nO2Scale._Storage = 'contiguous'
            
            self.d_nO2Scale_dofs = self.ncdelta.createVariable('O2_scaling_dofs',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
            self.d_nO2Scale_dofs.comment = 'degrees of freedom for signal for scaling factor for number density of O2 molecules at ground state'
            self.d_nO2Scale_dofs.units = ''
            self.d_nO2Scale_dofs._Storage = 'contiguous'
            
            self.d_nO2Scale_e = self.ncdelta.createVariable('O2_scaling_error',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
            self.d_nO2Scale_e.comment = 'posterior uncertainty for scaling factor for number density of O2 molecules at ground state'
            self.d_nO2Scale_e.units = ''
            self.d_nO2Scale_e._Storage = 'contiguous'
    
    def create_singlet_sigma_group(self,group_name,if_save_nO2Scale=True):
        self.ncsigma = self.ncid.createGroup(group_name)
        
        self.s_nO2s = self.ncsigma.createVariable('excited_O2',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
        self.s_nO2s.comment = 'number density of O2 molecules at singlet sigma state'
        self.s_nO2s.units = 'molec/cm3'
        self.s_nO2s._Storage = 'contiguous'
        
        self.s_nO2s_dofs = self.ncsigma.createVariable('excited_O2_dofs',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
        self.s_nO2s_dofs.comment = 'degrees of freedom for signal for number density of O2 molecules at singlet sigma state'
        self.s_nO2s_dofs.units = ''
        self.s_nO2s_dofs._Storage = 'contiguous'
        
        self.s_nO2s_e = self.ncsigma.createVariable('excited_O2_error',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
        self.s_nO2s_e.comment = 'posterior uncertainty for number density of O2 molecules at singlet sigma state'
        self.s_nO2s_e.units = 'molec/cm3'
        self.s_nO2s_e._Storage = 'contiguous'
        
        self.s_T = self.ncsigma.createVariable('temperature',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
        self.s_T.comment = 'temperature'
        self.s_T.units = 'K'
        self.s_T._Storage = 'contiguous'
        
        self.s_T_dofs = self.ncsigma.createVariable('temperature_dofs',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
        self.s_T_dofs.comment = 'degrees of freedom for signal for temperature'
        self.s_T_dofs.units = ''
        self.s_T_dofs._Storage = 'contiguous'
        
        self.s_T_e = self.ncsigma.createVariable('temperature_error',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
        self.s_T_e.comment = 'posterior uncertainty for temperature'
        self.s_T_e.units = 'K'
        self.s_T_e._Storage = 'contiguous'
        
        self.s_T_msis = self.ncsigma.createVariable('temperature_msis',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
        self.s_T_msis.comment = 'temperature from NRLMSISE-00 model'
        self.s_T_msis.units = 'K'
        self.s_T_msis._Storage = 'contiguous'
        
        self.s_nO2_msis = self.ncsigma.createVariable('O2_msis',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
        self.s_nO2_msis.comment = 'number density of O2 molecules'
        self.s_nO2_msis.units = 'molec/cm3'
        self.s_nO2_msis._Storage = 'contiguous'
        
        self.s_HW1E = self.ncsigma.createVariable('HW1E',np.float32,dimensions=('along_track','across_track'),fill_value=-1.0e+30)
        self.s_HW1E.comment = 'half width at 1/e of maximum of slit function'
        self.s_HW1E.units = 'nm'
        self.s_HW1E._Storage = 'contiguous'
        
        self.s_HW1E_dofs = self.ncsigma.createVariable('HW1E_dofs',np.float32,dimensions=('along_track','across_track'),fill_value=-1.0e+30)
        self.s_HW1E_dofs.comment = 'degrees of freedom for signal for half width at 1/e of maximum of slit function'
        self.s_HW1E_dofs.units = ''
        self.s_HW1E_dofs._Storage = 'contiguous'
        
        self.s_HW1E_e = self.ncsigma.createVariable('HW1E_error',np.float32,dimensions=('along_track','across_track'),fill_value=-1.0e+30)
        self.s_HW1E_e.comment = 'posterior error for half width at 1/e of maximum of slit function'
        self.s_HW1E_e.units = ''
        self.s_HW1E_e._Storage = 'contiguous'
        
        self.s_w_shift = self.ncsigma.createVariable('w_shift',np.float32,dimensions=('along_track','across_track'),fill_value=-1.0e+30)
        self.s_w_shift.comment = 'wavelength shift'
        self.s_w_shift.units = 'nm'
        self.s_w_shift._Storage = 'contiguous'
        
        self.s_w_shift_dofs = self.ncsigma.createVariable('w_shift_dofs',np.float32,dimensions=('along_track','across_track'),fill_value=-1.0e+30)
        self.s_w_shift_dofs.comment = 'degrees of freedom for signal for wavelength shift'
        self.s_w_shift_dofs.units = ''
        self.s_w_shift_dofs._Storage = 'contiguous'
        
        self.s_w_shift_e = self.ncsigma.createVariable('w_shift_error',np.float32,dimensions=('along_track','across_track'),fill_value=-1.0e+30)
        self.s_w_shift_e.comment = 'posterior error for wavelength shift'
        self.s_w_shift_e.units = ''
        self.s_w_shift_e._Storage = 'contiguous'
        
        self.s_chi2 = self.ncsigma.createVariable('chi2',np.float32,dimensions=('along_track','across_track'),fill_value=-1.0e+30)
        self.s_chi2.comment = 'goodness of fit indicated by the chi2 value'
        self.s_chi2.units = ''
        self.s_chi2._Storage = 'contiguous'
        
        self.s_rmse = self.ncsigma.createVariable('rmse',np.float32,dimensions=('along_track','across_track'),fill_value=-1.0e+30)
        self.s_rmse.comment = 'goodness of fit indicated by residual root mean square'
        self.s_rmse.units = 'same as radiance'
        self.s_rmse._Storage = 'contiguous'
        
        self.s_if_success = self.ncsigma.createVariable('if_success',np.int8,dimensions=('along_track','across_track'))
        self.s_if_success = 'if retrieve is successful'
        self.s_if_success = ''
        self.s_if_success = 'contiguous'
        
        self.s_Jprior = self.ncsigma.createVariable('Jprior',np.float32,dimensions=('along_track','across_track'),fill_value=-1.0e+30)
        self.s_Jprior.comment = 'distance between solution and prior normalized by prior error'
        self.s_Jprior.units = ''
        self.s_Jprior._Storage = 'contiguous'
        
        self.s_niter = self.ncsigma.createVariable('number_of_iterations',np.int8,dimensions=('along_track','across_track'))
        self.s_niter.comment = 'number of iterations'
        self.s_niter.units = ''
        self.s_niter._Storage = 'contiguous'
        
        if if_save_nO2Scale:
            self.s_nO2Scale = self.ncsigma.createVariable('O2_scaling',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
            self.s_nO2Scale.comment = 'scaling factor for number density of O2 molecules at ground state'
            self.s_nO2Scale.units = ''
            self.s_nO2Scale._Storage = 'contiguous'
            
            self.s_nO2Scale_dofs = self.ncsigma.createVariable('O2_scaling_dofs',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
            self.s_nO2Scale_dofs.comment = 'degrees of freedom for signal for scaling factor for number density of O2 molecules at ground state'
            self.s_nO2Scale_dofs.units = ''
            self.s_nO2Scale_dofs._Storage = 'contiguous'
            
            self.s_nO2Scale_e = self.ncsigma.createVariable('O2_scaling_error',np.float32,dimensions=('along_track','across_track','vertical'),fill_value=-1.0e+30)
            self.s_nO2Scale_e.comment = 'posterior uncertainty for scaling factor for number density of O2 molecules at ground state'
            self.s_nO2Scale_e.units = ''
            self.s_nO2Scale_e._Storage = 'contiguous'    
    
    def check_dtype(self,ncfvar,npvar):
        ''' Check compatibility between data type of numpy array
            and netCDF variable
            ARGS:
                ncfvar: netCDF file variable
                npvar: numpy array (holding data to be saved in file )
        '''
        if (ncfvar.dtype.type == npvar.dtype.type):
            return True
        else:
            sys.exit("numpy array type {} is not compatible with netCDF file '{}' variable {} type. Abort write ouput!!!"
                     .format(npvar.dtype.type,ncfvar.name,ncfvar.dtype.type))

    def check_dim(self,ncfvar,npvar):
        ''' Check compatibility between data type of numpy array
            and netCDF variable
            ARGS:
                ncfvar (netCDF file variable)
                npvar (numpy variable (holding data to be saved in file )
        '''

        if (ncfvar.size == npvar.size and ncfvar.shape == npvar.shape):
            return True
        else:
            sys.exit("numpy array shape/size {}/{} is not compatible with netCDF file '{}' shape/size {}/{}. Abort write ouput!!!"
                     .format(npvar.shape,npvar.size,ncfvar.name,ncfvar.shape,ncfvar.size))
    
    def set_variable(self,ncfvar,npvar,if_mask_invalid=True):
        ''' Set values to netCDF variables
            ARGS:
                ncfvar: netCDF variable
                npvar: numpy array
        '''  
      
        # Check consistency of data types
        self.check_dtype(ncfvar,npvar)
        # Check consistency of data shape and size
        self.check_dim(ncfvar,npvar)
        # Set data values
        if if_mask_invalid:
            ncfvar[:] = np.ma.masked_invalid(npvar)
        else:
            ncfvar[:] = npvar
        
    def close(self):
        self.ncid.close()

class Level2_Regridder(object):
    def __init__(self,west=-180.,east=180.,south=-90.,north=90.,
                 grid_size=1,grid_sizex=None,grid_sizey=None,
                 lower_z=60.,upper_z=120.,dz=5,
                 start_year=2004,start_month=1,start_day=1,
                 end_year=2012,end_month=12,end_day=None):
        '''
        initialize properties
        '''
        if end_day is None:
            end_day = monthrange(end_year,end_month)[-1]
        self.logger = logging.getLogger(__name__)
        self.logger.info('creating an instance of Level2_Regridder')
        if grid_sizex is None:
            grid_sizex = grid_size
            grid_sizey = grid_size
        else:
            grid_size = None
        self.grid_size = grid_size
        self.grid_sizex = grid_sizex
        self.grid_sizey = grid_sizey
        self.start_datetime = dt.datetime(start_year,start_month,start_day,0,0,0)
        self.end_datetime = dt.datetime(end_year,end_month,end_day,23,59,59)
        self.xgrid = arange_(west+grid_sizex/2,east,grid_sizex)
        self.ygrid = arange_(south+grid_sizey/2,north,grid_sizey)
        self.xgridr = arange_(west,east,grid_sizex)
        self.ygridr = arange_(south,north,grid_sizey)
        self.zgrid = arange_(lower_z+dz/2,upper_z,dz)
        self.zgridr = arange_(lower_z,upper_z,dz)
        self.nlon = len(self.xgrid)
        self.nlat = len(self.ygrid)
        self.nz = len(self.zgrid)
        self.dz = dz
    
    def find_orbits(self,l2_dir):
        l2_list = glob.glob(os.path.join(l2_dir,'SCI*.nc'))
        l2_datetimes = np.array([dt.datetime.strptime(os.path.split(l2_path)[-1][4:19],'%Y%m%dT%H%M%S') for l2_path in l2_list])
        l2_orbits = np.array([np.int(os.path.split(l2_path)[-1][26:31]) for l2_path in l2_list])
        time_mask = np.array([(l2_dt >= self.start_datetime) & (l2_dt <= self.end_datetime) for l2_dt in l2_datetimes])
        self.l2_list = np.array(l2_list)[time_mask]
        self.l2_datetimes = l2_datetimes[time_mask]
        self.l2_orbits = l2_orbits[time_mask]
        self.l2_dir = l2_dir
        self.nl2 = len(self.l2_list)
        self.logger.info('{} orbits are found'.format(self.nl2))
        
    def drop_in_the_box(self,l2_dir=None,regrid_field='temperature',
                        max_delta_chi2=2,max_sigma_chi2=2,
                        min_delta_dofs=0.5,min_sigma_dofs=0.5):
        if not hasattr(self,'l2_list'):
            self.find_orbits(l2_dir)
        
        A = np.zeros((self.nlat,self.nlon,self.nz),dtype=np.float32)
        B = np.zeros((self.nlat,self.nlon,self.nz),dtype=np.float32)
        Dd = np.zeros((self.nlat,self.nlon,self.nz),dtype=np.int32)
        Ds = np.zeros((self.nlat,self.nlon,self.nz),dtype=np.int32)
        
        for fn in self.l2_list:
            s = Level2_Reader(fn)
            self.logger.info('loading variables from '+fn)
            s.load_variable(data_fields=['longitude','latitude','tangent_height',
                                         'singlet_delta/{}'.format(regrid_field),
                                         'singlet_delta/{}_dofs'.format(regrid_field),
                                         'singlet_delta/{}_error'.format(regrid_field),
                                         'singlet_delta/chi2',
                                         'singlet_sigma/{}'.format(regrid_field),
                                         'singlet_sigma/{}_dofs'.format(regrid_field),
                                         'singlet_sigma/{}_error'.format(regrid_field),
                                         'singlet_sigma/chi2'],
                            data_names=['lon','lat','z','d','ddofs','de','dchi2','s','sdofs','se','schi2'])
            for iline in range(s.lat.shape[0]):
                for ift in range(s.lat.shape[1]):
                    for ith in range(s.lat.shape[2]):
                        if (s.lat[iline,ift,ith] < self.ygridr.min()) or (s.lat[iline,ift,ith] > self.ygridr.max()) or\
                            (s.lon[iline,ift,ith] < self.xgridr.min()) or (s.lon[iline,ift,ith] > self.xgridr.max()) or\
                            (s.z[iline,ift,ith] < self.zgridr.min()) or (s.z[iline,ift,ith] > self.zgridr.max()):
                                continue
                        lat_idx = np.argmin(np.abs(self.ygrid-s.lat[iline,ift,ith]))
                        lon_idx = np.argmin(np.abs(self.xgrid-s.lon[iline,ift,ith]))
                        z_idx = np.argmin(np.abs(self.zgrid-s.z[iline,ift,ith]))
                        if (s.ddofs[iline,ift,ith] >= min_delta_dofs) and (s.dchi2[iline,ift] < max_delta_chi2) and ~np.isnan(s.d[iline,ift,ith]):
                            A[lat_idx,lon_idx,z_idx] = np.nansum([A[lat_idx,lon_idx,z_idx],s.d[iline,ift,ith]*s.de[iline,ift,ith]])
                            B[lat_idx,lon_idx,z_idx] = np.nansum([B[lat_idx,lon_idx,z_idx],s.de[iline,ift,ith]])
                            Dd[lat_idx,lon_idx,z_idx] = Dd[lat_idx,lon_idx,z_idx]+1
                        if (s.sdofs[iline,ift,ith] >= min_sigma_dofs) and (s.schi2[iline,ift] < max_sigma_chi2) and ~np.isnan(s.s[iline,ift,ith]):
                            A[lat_idx,lon_idx,z_idx] = np.nansum([A[lat_idx,lon_idx,z_idx],s.s[iline,ift,ith]*s.se[iline,ift,ith]])
                            B[lat_idx,lon_idx,z_idx] = np.nansum([B[lat_idx,lon_idx,z_idx],s.se[iline,ift,ith]])
                            Ds[lat_idx,lon_idx,z_idx] = Ds[lat_idx,lon_idx,z_idx]+1
            s.close()
            '''
            d_mask = (s.ddofs >= min_delta_dofs) & (np.repeat(s.dchi2[...,np.newaxis],s.z.shape[2],axis=2) < max_delta_chi2) & (~np.isnan(s.d))
            s_mask = (s.sdofs >= min_sigma_dofs) & (np.repeat(s.schi2[...,np.newaxis],s.z.shape[2],axis=2) < max_sigma_chi2) & (~np.isnan(s.s)) & (s.z >= 70)
            all_ones = np.ones(s.lat.shape,dtype=np.int32)
            for ilat in range(self.nlat):
                for ilon in range(self.nlon):
                    for iz in range(self.nz):
                        grid_mask = (s.lat >= self.ygridr[ilat]) & (s.lat <= self.ygridr[ilat+1]) &\
                        (s.lon >= self.xgridr[ilon]) & (s.lon <= self.xgridr[ilon+1]) &\
                        (s.z >= self.zgridr[iz]) & (s.z <= self.zgridr[iz+1]) 
                        grid_mask_d = grid_mask & d_mask
                        grid_mask_s = grid_mask & s_mask
                        A[ilat,ilon,iz] = A[ilat,ilon,iz] + np.nansum(s.d[grid_mask_d]*s.de[grid_mask_d]) + np.nansum(s.s[grid_mask_s]*s.se[grid_mask_s])
                        B[ilat,ilon,iz] = B[ilat,ilon,iz] + np.nansum(s.de[grid_mask_d]) + np.nansum(s.se[grid_mask_s])
                        Dd[ilat,ilon,iz] = Dd[ilat,ilon,iz] + np.sum(all_ones[grid_mask_d])
                        Ds[ilat,ilon,iz] = Ds[ilat,ilon,iz] + np.sum(all_ones[grid_mask_s])
            '''
        setattr(self,regrid_field,A/B)
        self.total_sample_weight = B
        self.num_sample_delta = Dd
        self.num_sample_sigma = Ds
    
    def save_nc(self,l3_dir,header='SCI_airglow_L3_'):
        l3_path = os.path.join(l3_dir,header+self.start_datetime.strftime('%Ym%m%d')+'-'+self.end_datetime.strftime('%Ym%m%d')+'.nc')
        nc = Dataset(l3_path,'w',format='NETCDF4')
        nc.createDimension('lat',self.nlat)
        nc.createDimension('lon',self.nlon)
        nc.createDimension('z',self.nz)
        
        latid = nc.createVariable('latitude',np.float32,dimensions=('lat'))
        latid.comment = 'latitude at grid center'
        latid.units = 'degrees north'
        latid[:] = self.ygrid
        
        lonid = nc.createVariable('longitude',np.float32,dimensions=('lon'))
        lonid.comment = 'longitude at grid center'
        lonid.units = 'degrees east'
        lonid[:] = self.xgrid
        
        zid = nc.createVariable('altitude',np.float32,dimensions=('z'))
        zid.comment = 'altitude at grid center'
        zid.units = 'km'
        zid[:] = self.zgrid
        
        if hasattr(self,'temperature'):
            tmp = nc.createVariable('temperature',np.float32,dimensions=('lat','lon','z'))
            tmp.comment = 'temperature'
            tmp.units = 'K'
            tmp[:] = np.ma.masked_invalid(self.temperature)
        
        b = nc.createVariable('total_sample_weight',np.float32,dimensions=('lat','lon','z'))
        b.comment = 'cumulative weight for regridding'
        b.units = 'K'
        b[:] = np.ma.masked_invalid(self.total_sample_weight)
        
        dd = nc.createVariable('num_sample_delta',np.int32,dimensions=('lat','lon','z'))
        dd.comment = 'number of valid singlet Delta observations'
        dd.units = ''
        dd[:] = np.ma.masked_invalid(self.num_sample_delta)
        
        ds = nc.createVariable('num_sample_sigma',np.int32,dimensions=('lat','lon','z'))
        ds.comment = 'number of valid singlet Sigma observations'
        ds.units = ''
        ds[:] = np.ma.masked_invalid(self.num_sample_sigma)
        
        nc.close()

# From https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput
def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.
​
    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.
    
    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]
    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]
    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)
    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))
    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)
    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)
    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)
    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]
    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)
    return rval
def get_angle(v1,v2):
    dot = np.sum(v1*v2)
    det = v1[0]*v2[1] - v1[1]*v2[0]
    ang = np.rad2deg(np.arctan2(det,dot))
    if(ang < 0.0):
        ang += 360.0
    return ang
def aggregate_corners(clon,clat,xtrk_agg,atrk_agg,bl_clockwise_order=True,snap_to_corners=True):
    # Dimensions
    imx = clon.shape[0] ; jmx = clon.shape[1]
    # Get the block aggregate 
    ilm = np.concatenate([np.arange(0,imx,xtrk_agg),[imx]]) ; imx_a = len(ilm)-1
    jlm = np.concatenate([np.arange(0,jmx,atrk_agg),[jmx]]) ; jmx_a = len(jlm)-1
    # Output corners
    clon_agg = np.zeros((imx_a,jmx_a,4))
    clat_agg = np.zeros((imx_a,jmx_a,4))
    # Loop over corners
    for i in range(imx_a):
        x0 = ilm[i] ; xf = ilm[i+1]-1
        for j in range(jmx_a):
            y0 = jlm[j] ; yf = jlm[j+1]-1
            
            # Use edges as inputs
            pts = np.zeros((16,2))
            pts[0:4,0]   = clon[x0,y0,:] ; pts[0:4,1]   = clat[x0,y0,:]
            pts[4:8,0]   = clon[x0,yf,:] ; pts[4:8,1]   = clat[x0,yf,:]
            pts[8:12,0]  = clon[xf,yf,:] ; pts[8:12,1]  = clat[xf,yf,:]
            pts[12:16,0] = clon[xf,y0,:] ; pts[12:16,1] = clat[xf,y0,:]
            
            rval = minimum_bounding_rectangle(pts)
            if(bl_clockwise_order):
                ang = np.zeros(4)
                for c in range(4):
                    # Compute center coordinate
                    lon = np.mean(pts[:,0])
                    lat = np.mean(pts[:,1])
                    # Compute angle from due south
                    v1 = np.array([rval[c,0]-lon,rval[c,1]-lat])
                    v2 = np.array([0,-1])
                    ang[c] = get_angle(v1,v2)
                # Sort angles clockwise
                ids = np.argsort(ang)
                rval = rval[ids,:]
            if(snap_to_corners):
                for c in range(4):
                    cdst = np.sqrt( np.power(pts[:,0] - rval[c,0],2) + \
                                    np.power(pts[:,1] - rval[c,1],2) )
                    ic = np.argmin(cdst)
                    rval[c,0] = pts[ic,0] ; rval[c,1] = pts[ic,1]
            
            # Store value
            clon_agg[i,j,:] = rval[:,0]
            clat_agg[i,j,:] = rval[:,1]
            
    return clon_agg,clat_agg
def F_wrapper_parallel_ft(args):
    try:
        outp = F_parallel_ft(*args)
        return outp
    except Exception as e:
        print(e)
        outp = Retrieval_Results()
        return outp

def F_parallel_ft(granule,ift,startWavelength=1240,endWavelength=1300,
                  minTH=35,maxTH=100,Re=6371.,
                  w1_step=-0.001):
    
    # extract info from each footprint of each granule
    wavelength = granule['wavelength'].copy()
    radiance = granule['radiance'][:,ift,:].squeeze().copy()
    radiance_error = granule['radiance_error'][:,ift,:].squeeze().copy()
    tangent_height = granule['tangent_height'][:,ift].copy()
    waveMask = (np.mean(wavelength,axis=0) >= startWavelength) & (np.mean(wavelength,axis=0) <= endWavelength) & (~np.isnan(np.mean(radiance,axis=0)))
    THMask = (~np.isnan(tangent_height)) & (tangent_height < maxTH) & (tangent_height > minTH)
    tangent_height = tangent_height[THMask]
    wavelength = wavelength[np.ix_(THMask,waveMask)]
    radiance = radiance[np.ix_(THMask,waveMask)]
    radiance_error = radiance_error[np.ix_(THMask,waveMask)]
    
    dZ = np.abs(np.diff(tangent_height))
    dZ = np.append(dZ,dZ[-1])
    
    nth = len(tangent_height)
    nw2 = radiance.shape[1]
    L = np.zeros((nth,nth))
    for i in range(nth):
        
        for j in range(i,nth):
            if j == nth-1:
                topTH = tangent_height[j]+np.abs(tangent_height[j]-tangent_height[j-1])
            else:
                topTH = tangent_height[j+1]
            L[i,j] = np.sqrt(np.power(topTH+Re,2)-np.power(tangent_height[i]+Re,2))-\
            np.sqrt(np.power(tangent_height[j]+Re,2)-np.power(tangent_height[i]+Re,2))
    L = L*1e5# km to cm
    So = np.diag(radiance_error[:,0])
    rg = 4*np.pi*np.linalg.inv(L.T@np.linalg.inv(So)@L)@L.T@np.linalg.inv(So)@radiance
    nO2s_profile = np.trapz(rg,wavelength)/2.27e-4
    nO2s_profile[nO2s_profile < 0] = 0
    p_profile, T_profile = F_sample_standard_atm(tangent_height)
    # w1 is the high res wavelength grid. has to be descending
    w1 = arange_(endWavelength,startWavelength,-np.abs(w1_step))#-0.0005
    T_profile_e = np.ones(T_profile.shape)*20
    T_profile_e[tangent_height<50] = 2
    nO2s_profile_e = np.ones(nO2s_profile.shape)*nO2s_profile
    nO2s_profile_e[nO2s_profile_e<0.1*np.mean(nO2s_profile)] = 0.1*np.mean(nO2s_profile)
    
    aOE = Forward_Model(func=F_airglow_forward_model,
                              independent_vars=['w1','wavelength','L','p_profile','nu'],
                              param_names=['nO2s_profile','T_profile','HW1E','w_shift'])
    p_profile_middle = p_profile+np.append(np.diff(p_profile),0.)/2
    aOE.set_prior('nO2s_profile',prior=nO2s_profile,prior_error=nO2s_profile_e,p_profile=p_profile_middle,correlation_scaleHeight=1)
    aOE.set_prior('T_profile',prior=T_profile,prior_error=T_profile_e,p_profile=p_profile_middle,correlation_scaleHeight=1,vmin=80,vmax=350)
    aOE.set_prior('HW1E',prior=0.9,prior_error=0.5)
    aOE.set_prior('w_shift',prior=0.,prior_error=1.)
    result = aOE.retrieve(radiance,radiance_error,max_iter=6,
                            w1=w1,wavelength=wavelength,
                            L=L,p_profile=p_profile_middle)
    result.tangent_height = tangent_height
    result.THMask = THMask
    result.dZ = dZ
    return result
