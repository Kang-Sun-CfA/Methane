# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 21:15:24 2020

@author: Kang Sun
"""

import datetime
import numpy as np
import os
import glob
import logging
import scipy.io as sio
from netCDF4 import Dataset
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

def datedev_py(matlab_datenum):
    """
    convert matlab datenum double to python datetime object
    """
    python_datetime = datetime.datetime.fromordinal(int(matlab_datenum)) + datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days = 366)
    return python_datetime

def F_ncread_selective(fn,varnames):
    """
    very basic netcdf reader, similar to F_ncread_selective.m
    created on 2019/08/13
    """
    #from netCDF4 import Dataset
    ncid = Dataset(fn,'r')
    outp = {}
    for varname in varnames:
        outp[varname] = ncid.variables[varname][:]
    ncid.close()
    return outp

def F_local2utc_hour(longitude):
    if longitude > 180:
        longitude = longitude-360
    return np.round(longitude/15)

class amdar(object):
    
    def __init__(self,
                 start_year=2018,start_month=8,start_day=1,start_hour=0,
                 end_year=2018,end_month=8,end_day=31,end_hour=23,
                 daily_start_local_hour=13,daily_end_local_hour=13,
                 airports_info_path=''):
        """
        initialize the amdar object
        start/end times: 
            beginning and end of analysing period
        daily_start/end_local_hour: 
            local hours of interest for all airports.
            the utc hour in amdar are converted to local solar time, rounded to 
            integer hour numbers. See F_local2utc_hour function
        airports_info_path:
            path to amdar airports basic information. Necessary fields are
            Name, Lat, Lon, and DEM
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info('creating an instance of amdar logger')
        if not airports_info_path:
            self.airports_info = pd.read_csv(airports_info_path)
        else:
            self.airports_info = []
            self.logger.info('airports information is not provided. amdar-related functions may not work')
        self.start_datetime = datetime.datetime(start_year,start_month,start_day,
                                           start_hour,0,0)
        self.end_datetime = datetime.datetime(end_year,end_month,end_day,
                                           end_hour,0,0)
        self.daily_start_local_hour = daily_start_local_hour
        self.daily_end_local_hour = daily_end_local_hour
        
    def F_load_amdar(self,amdar_dir,pblh_index=13,pbl_type_index=18):
        """
        load amdar data within specified time interval and at specified daily 
        local hours into memory.
        amdar_dir:
            directory containing amdar pblh data, which are grouped annually for 
            each airport. For example, SFO_PBLH_2019.mat
        pblh_index, pbl_type_index:
            in default pblh files from Zhang Yuanjie, the pblh is a 19x8760 matrix.
            rows 0-17 are different version of pblh calculation (using different parameters in the bulk Ri method)
            the last row is an integer denoting the type of pbl, i.e., 1-stable, 2-neutral, 3-convective
            the total column number 8760 is the number of hours in 365 days
        key output:
            amdar_dataframe, a pandas data frame containing amdar information
        """
        airports_info = self.airports_info
        start_datetime = self.start_datetime
        end_datetime = self.end_datetime
        daily_start_local_hour = self.daily_start_local_hour
        daily_end_local_hour = self.daily_end_local_hour
        
        airport_lon = np.array([])
        airport_lat = np.array([])
        airport_name = np.array([])
        amdar_datetime = np.array([])
        amdar_local_datetime = np.array([])
        amdar_pblh = np.array([])
        amdar_pbl_type = np.array([])
        for year in range(start_datetime.year,end_datetime.year+1):
            self.logger.info('loading amdar %04d'%year)
            # create an array of hours spanning over one year
            hours_in_this_year = np.array([datetime.datetime(year,1,1,0,0,0)+datetime.timedelta(hours=i) for i in range(9000)])
            # mask hours between specified time interval
            use_hours = (hours_in_this_year >= start_datetime) & (hours_in_this_year <= end_datetime)
            for iairport in range(len(airports_info['Name'])):
                airport = airports_info['Name'][iairport]
                airport_local2utc_hour = F_local2utc_hour(airports_info['Lon'][iairport])
                self.logger.info('loading airport '+airport+', %02d'%airport_local2utc_hour+' hours from UTC')
                zyj_file_name = os.path.join(amdar_dir,airport+'_PBLH_%04d'%year+'.mat')
                zyj_data = sio.loadmat(zyj_file_name)['PBLH']
                airport_use_hours = use_hours[0:zyj_data.shape[1]]
                zyj_pblh = zyj_data[pblh_index,airport_use_hours].squeeze()
                zyj_pbl_type = zyj_data[pbl_type_index,airport_use_hours].squeeze().astype(np.int16)
                zyj_datetime = hours_in_this_year[0:zyj_data.shape[1]][use_hours[0:zyj_data.shape[1]]]
                zyj_local_datetime = zyj_datetime+datetime.timedelta(hours=airport_local2utc_hour)
                if (daily_start_local_hour != 0) or (daily_end_local_hour != 23):
                    self.logger.debug('load only local hour from %02d'%daily_start_local_hour+' to %02d'%daily_end_local_hour)
                    daily_index = np.array([(dt.hour >= daily_start_local_hour) & 
                                            (dt.hour <= daily_end_local_hour)
                                            for dt in zyj_local_datetime])
                else:
                    daily_index = np.ones(zyj_pblh.shape,dtype=bool)
                amdar_datetime = np.concatenate((amdar_datetime,zyj_datetime[daily_index]))
                amdar_local_datetime = np.concatenate((amdar_local_datetime,zyj_local_datetime[daily_index]))
                amdar_pblh = np.concatenate((amdar_pblh,zyj_pblh[daily_index]))
                amdar_pbl_type = np.concatenate((amdar_pbl_type,zyj_pbl_type[daily_index]))
                airport_lon = np.concatenate((airport_lon,
                                              airports_info['Lon'][iairport]
                                              *np.ones(zyj_pblh[daily_index].shape)))
                airport_lat = np.concatenate((airport_lat,
                                              airports_info['Lat'][iairport]
                                              *np.ones(zyj_pblh[daily_index].shape)))
                airport_name = np.concatenate((airport_name,
                                              np.repeat(airport,len(zyj_pblh[daily_index]))))
        
        self.amdar_datetime = amdar_datetime
        self.amdar_local_datetime = amdar_local_datetime
        # convert python datetime object array to matlab datenum (easier to use)
        amdar_datenum = np.array([dt.toordinal()+dt.hour/24.+dt.minute/1440.+dt.second/86400.+366.
                                       for dt in amdar_datetime],dtype=np.float64)
        amdar_local_datenum = np.array([dt.toordinal()+dt.hour/24.+dt.minute/1440.+dt.second/86400.+366.
                                       for dt in amdar_local_datetime],dtype=np.float64)
        amdar_hour = np.array([dt.hour for dt in amdar_datetime],dtype=np.int16)
        amdar_local_hour = np.array([dt.hour for dt in amdar_local_datetime],dtype=np.int16)
        amdar_doy = np.array([dt.timetuple().tm_yday for dt in amdar_datetime],dtype=np.int16)
        dict_data = {'airport_name':airport_name,\
                     'airport_lon':airport_lon,\
                     'airport_lat':airport_lat,\
                     'amdar_datenum':amdar_datenum,\
                     'amdar_local_datenum':amdar_local_datenum,\
                     'amdar_doy':amdar_doy,\
                     'amdar_hour':amdar_hour,\
                     'amdar_local_hour':amdar_local_hour,\
                     'amdar_pblh':amdar_pblh,\
                     'amdar_pbl_type':amdar_pbl_type}
        self.amdar_dataframe = pd.DataFrame(dict_data)
    
    def F_load_era5_map(self,era5_dir,
                        era5_fields=['blh','sp','zust'],
                        fn_header='CONUS',
                        amdar_lag_hour=[0.5]):
        """
        load era5 fields as a map at amdar pblh observation local hours
        era5_dir:
            directory where era5 data are saved. See the era5 python script at
            https://github.com/Kang-Sun-CfA/Methane/blob/master/l2_met/era5.py
            for downloading
        era5_fields:
            variables from era5 see https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation#ERA5datadocumentation-Parameterlistings
            for era5 variables
        fn_header:
            file name header of downloaded era5 nc files
        amdar_lag_hour:
            zero lag should be by default 0.5. The lag of amdar true time from era5 time. For 
            example, the amdar pblh at 12 utc is the average of 12-13 utc, 
            whereas era5 pblh at 12 utc is the instantaneous value at 12 utc.
            So, to match 12 utc amdar, era5 should be sampled at 12.5 utc. May 
            add more lag hours to sample era5 at 11.5, 10.5 utc, etc.
        key outputs:
            era_map, a dictionary containing np arrays of maps at each local hour on each day
        """
        if len(amdar_lag_hour) > 1:
            self.logger.error('Multiple lag hours not supported!')
        start_datetime = self.start_datetime
        end_datetime = self.end_datetime
        amdar_local_hours = np.arange(self.daily_start_local_hour,self.daily_end_local_hour+1)
        nhour = len(amdar_local_hours)
        start_date = start_datetime.date()
        end_date = end_datetime.date()
        days = (end_date-start_date).days+1
        DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
        # initialize era maps
        era_map = {}
        for field in era5_fields:
            era_map[field] = []
        nc_out = {}
        # for each date, load era5 netcdf and sample at amdar local hours. add time lags if necessary
        for DATE in DATES:
            fn = os.path.join(era5_dir,DATE.strftime('Y%Y'),\
                              DATE.strftime('M%m'),\
                              DATE.strftime('D%d'),\
                              fn_header+'_2D_'+DATE.strftime('%Y%m%d')+'.nc')
            self.logger.info('loading/sampling ERA5 file '+fn)
            if not nc_out:
                nc_out = F_ncread_selective(fn,np.concatenate(
                        (era5_fields,['latitude','longitude','time'])))
                # rearrange data to lon, lat, time, sort the lat dimension
                nc_out['latitude'] = nc_out['latitude'][::-1]
                nlat = len(nc_out['latitude'])
                nlon = len(nc_out['longitude'])
                lon_mesh,lat_mesh = np.meshgrid(nc_out['longitude'],nc_out['latitude'])
                lon_mesh = lon_mesh
                lat_mesh = lat_mesh
            else:
                nc_out = F_ncread_selective(fn,np.concatenate(
                        (era5_fields,['latitude','longitude','time'])))
                # rearrange data to lon, lat, time, sort the lat dimension
                nc_out['latitude'] = nc_out['latitude'][::-1]
            # rearrange data to lon, lat, time, sort the lat dimension
            for field in era5_fields:
                nc_out[field] = nc_out[field].transpose((2,1,0))[:,::-1,:]
            era_utc_datenum = nc_out['time']/24.+693962.
            map_utc_datenum = np.zeros((nhour,nlon))
            for ilon in range(nlon):
                map_utc_datenum[:,ilon] = (amdar_local_hours+amdar_lag_hour[0]-\
                               np.round(nc_out['longitude'][ilon]/15))/24.\
                               +DATE.toordinal()+366.
            for ihour in range(nhour):
                utc_mesh = np.tile(map_utc_datenum[ihour,:].squeeze(),(nlat,1))
                for field in era5_fields:
                    f=RegularGridInterpolator((nc_out['longitude'],nc_out['latitude'],era_utc_datenum),
                                            nc_out[field],bounds_error=False,fill_value=np.nan)
                    era_map[field].append(f((lon_mesh,lat_mesh,utc_mesh)))
        for field in era5_fields:
            era_map[field] = np.array(era_map[field])
        self.era_map = era_map
        self.era_lon = nc_out['longitude']
        self.era_lat = nc_out['latitude']
       
    def F_predict_pblh(self, RFmodel_path, variables=[]):
        """
        author: Shiqi Tao
        estimate pblh using existing RF (random forest) model and predictors
        RFmodel_path:
            path to RF regressor object saved as pickle file
        variables:
            a list of predictor variables, each variable should be a flattened
            1d array. example:
            [[var05['blh'].flatten(),var_15['blh'].flatten(),var_35['blh'].flatten(),var05['zust'].flatten(),var05['ssr'].flatten(),var05['sshf'].flatten()]]
        """
        import pickle
        if len(variables) == 0:
            self.logger.error('No variable input')
            return
        predictors = np.concatenate(variables,axis=0).T
        RF_pblh = pickle.load(open(RFmodel_path, 'rb'))
        y_predict = RF_pblh.predict(predictors)
        #PBLH_estimated = y_predict.reshape(self.era_lat.shape[0],self.era_lon.shape[0])
        return y_predict
        
    def F_interp_merra2_amdar(self,sounding_lon,sounding_lat,sounding_datenum,\
                            merra2_dir,\
                            merra2_fields=['PBLTOP','PS'],\
                            fn_header='MERRA2_300.tavg1_2d_slv_Nx',
                            lag_hour=[0,-2,-4]):
        """
        sample a field from merra2 data in .nc format at coordinates sounding_lon, sounding_lat, and
        time sounding_datenum. based on F_interp_merra2 in popy. added lag_hour
        to faciliate random forest pblh model
        sounding_lon:
            longitude for interpolation
        sounding_lat:
            latitude for interpolation
        sounding_datenum:
            time for interpolation in matlab datenum double format, in !!!UTC!!!
        merra2_dir:
            directory where subset era5 data in .nc are saved
        merra2_fields:
            variables to interpolate from merra2, only 2d fields are supported
        fn_header:
            'MERRA2_300.tavg1_2d_slv_Nx' or 'MERRA2_400.tavg1_2d_slv_Nx'
        lag_hour:
            if sounding is 17 utc, lag hour 0 samples at 17 utc, lag hour
            -2 samples at 15 utc
        """
        start_datenum = np.amin(sounding_datenum)
        end_datenum = np.amax(sounding_datenum)
        start_date = datedev_py(start_datenum).date()
        
        end_date = datedev_py(end_datenum).date()
        days = (end_date-start_date).days+1
        DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
        merra2_data = {}
        iday = 0
        for DATE in DATES:
            merra_filedir = os.path.join(merra2_dir,DATE.strftime('Y%Y'),\
                                         DATE.strftime('M%m'),DATE.strftime('D%d'))
            merra_flist = glob.glob(merra_filedir+'/*.nc')
            if len(merra_flist) > 1:
                self.logger.warning('Careful! More than one nc file in MERRA daily folder!')
            elif len(merra_flist) == 0:
                self.logger.warning('No merra file')
                continue
            fn = merra_flist[0]
            if not merra2_data:
                nc_out = F_ncread_selective(fn,np.concatenate(
                        (merra2_fields,['lat','lon','time'])))
                merra2_data['lon'] = nc_out['lon']
                merra2_data['lat'] = nc_out['lat']
                # how many hours are there in each daily file? have to be the same 
                nhour = len(nc_out['time'])
                merra2_data['datenum'] = np.zeros((nhour*(days)),dtype=np.float64)
                # merra2 time is defined as minutes since 00:30:00 on that day
                merra2_data['datenum'][iday*nhour:((iday+1)*nhour)] = DATE.toordinal()+366.+(nc_out['time']+30)/1440
                for field in merra2_fields:
                    merra2_data[field] = np.zeros((len(merra2_data['lon']),len(merra2_data['lat']),nhour*(days)))
                    # was read in as 3-d array in time, lat, lon; transpose to lon, lat, time
                    merra2_data[field][...,iday*nhour:((iday+1)*nhour)] = nc_out[field].transpose((2,1,0))
            else:
                nc_out = F_ncread_selective(fn,np.concatenate(
                        (merra2_fields,['time'])))
                # merra2 time is defined as minutes since 00:30:00 on that day
                merra2_data['datenum'][iday*nhour:((iday+1)*nhour)] = DATE.toordinal()+366.+(nc_out['time']+30)/1440
                for field in merra2_fields:
                    # was read in as 3-d array in time, lat, lon; transpose to lon, lat, time
                    merra2_data[field][...,iday*nhour:((iday+1)*nhour)] = nc_out[field].transpose((2,1,0))
            # increment iday
            iday = iday+1
    
        sounding_interp = {}
        # interpolate
        for field in merra2_fields:
            f = \
            RegularGridInterpolator((merra2_data['lon'],merra2_data['lat'],merra2_data['datenum']),\
                                merra2_data[field],bounds_error=False,fill_value=np.nan)
            for lag in lag_hour:
                fieldname = field+'%s'%lag
                sounding_interp[fieldname] = f((sounding_lon,sounding_lat,sounding_datenum+lag/24))
        return sounding_interp
        
    def F_interp_era5_amdar(self,sounding_lon,sounding_lat,sounding_datenum,\
                            era5_dir,\
                            era5_fields=['blh','zust','ssr','sshf'],\
                            fn_header='CONUS',
                            lag_hour=[0,-2,-4]):
        """
        sample a field from era5 data in .nc format at coordinates sounding_lon, sounding_lat, and
        time sounding_datenum. based on F_interp_era5 in popy. added lag_hour
        to faciliate random forest pblh model
        sounding_lon:
            longitude for interpolation
        sounding_lat:
            latitude for interpolation
        sounding_datenum:
            time for interpolation in matlab datenum double format, in !!!UTC!!!
        era5_dir:
            directory where subset era5 data in .nc are saved
        era5_fields:
            variables to interpolate from era5, only 2d fields are supported
        fn_header:
            in general should denote domain location of era5 data
        lag_hour:
            if sounding is 17 utc, lag hour 0 samples era5 at 17 utc, lag hour
            -2 samples era5 at 15 utc
        """
        start_datenum = np.amin(sounding_datenum)
        end_datenum = np.amax(sounding_datenum)
        start_date = datedev_py(start_datenum).date()
        
        end_date = datedev_py(end_datenum).date()
        days = (end_date-start_date).days+1
        DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
        era5_data = {}
        iday = 0
        for DATE in DATES:
            fn = os.path.join(era5_dir,DATE.strftime('Y%Y'),\
                                       DATE.strftime('M%m'),\
                                       DATE.strftime('D%d'),\
                                       fn_header+'_2D_'+DATE.strftime('%Y%m%d')+'.nc')
            if not era5_data:
                nc_out = F_ncread_selective(fn,np.concatenate(
                        (era5_fields,['latitude','longitude','time'])))
                era5_data['lon'] = nc_out['longitude']
                era5_data['lat'] = nc_out['latitude'][::-1]
                # how many hours are there in each daily file? have to be the same 
                nhour = len(nc_out['time'])
                era5_data['datenum'] = np.zeros((nhour*(days)),dtype=np.float64)
                # era5 time is defined as 'hours since 1900-01-01 00:00:00.0'
                era5_data['datenum'][iday*nhour:((iday+1)*nhour)] = nc_out['time']/24.+693962.
                for field in era5_fields:
                    era5_data[field] = np.zeros((len(era5_data['lon']),len(era5_data['lat']),nhour*(days)))
                    if len(nc_out[field].shape) != 3:
                        self.logger.warning('Warning!!! Anomaly in the dimension of ERA5 fields.')
                        self.logger.warning('Tentatively taking only the first element of the second dimension')
                        nc_out[field] = nc_out[field][:,0,...].squeeze()
                    # was read in as 3-d array in time, lat, lon; transpose to lon, lat, time
                    era5_data[field][...,iday*nhour:((iday+1)*nhour)] = nc_out[field].transpose((2,1,0))[:,::-1,:]
            else:
                nc_out = F_ncread_selective(fn,np.concatenate((era5_fields,['time'])))
                # era5 time is defined as 'hours since 1900-01-01 00:00:00.0'
                era5_data['datenum'][iday*nhour:((iday+1)*nhour)] = nc_out['time']/24.+693962.
                for field in era5_fields:
                    # was read in as 3-d array in time, lat, lon; transpose to lon, lat, time
                    era5_data[field][...,iday*nhour:((iday+1)*nhour)] = nc_out[field].transpose((2,1,0))[:,::-1,:]
            # increment iday
            iday = iday+1
        
        sounding_interp = {}
        for field in era5_fields:
            f = RegularGridInterpolator((era5_data['lon'],era5_data['lat'],era5_data['datenum']),\
                                        era5_data[field],bounds_error=False,fill_value=np.nan)
            for lag in lag_hour:
                fieldname = field+'%s'%lag
                sounding_interp[fieldname] = f((sounding_lon,sounding_lat,sounding_datenum+lag/24))
        return sounding_interp
    
    def F_load_era5(self,era5_dir,
                        era5_fields=['blh','sp','zust','ishf','ie','p83.162','p84.162'],
                        fn_header='CONUS',
                        amdar_lag_hour=[0.5,-0.5]):
        """
        sample era5 fields at amdar pblh observation coordinates and times
        era5_dir:
            directory where era5 data are saved. See the era5 python script at
            https://github.com/Kang-Sun-CfA/Methane/blob/master/l2_met/era5.py
            for downloading
        era5_fields:
            variables to interpolate from era5 see https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation#ERA5datadocumentation-Parameterlistings
            for era5 variables
        fn_header:
            file name header of downloaded era5 nc files
        amdar_lag_hour:
            zero lag should be by default 0.5. The lag of amdar true time from era5 time. For 
            example, the amdar pblh at 12 utc is the average of 12-13 utc, 
            whereas era5 pblh at 12 utc is the instantaneous value at 12 utc.
            So, to match 12 utc amdar, era5 should be sampled at 12.5 utc. May 
            add more lag hours to sample era5 at 11.5, 10.5 utc, etc.
        key outputs:
            amdar_era5_dataframe, a pandas dataframe with sampled era5 data,
            in the same height as amdar_dataframe
        """
        start_datetime = self.start_datetime
        end_datetime = self.end_datetime
        amdar_dataframe = self.amdar_dataframe
        start_date = start_datetime.date()
        end_date = end_datetime.date()
        days = (end_date-start_date).days+1
        DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
        # initialize nan dictionary same height as amdar dataframe
        amdar_interp = {}
        for field in era5_fields:
            for lag in amdar_lag_hour:
                fieldname = field+'%s'%lag
                amdar_interp[fieldname] = np.full(amdar_dataframe['amdar_pblh'].shape,np.nan)
        # for each date, load era5 netcdf and sample at amdar coordinate and time
        # may need to combine dates later
        for DATE in DATES:
            fn = os.path.join(era5_dir,DATE.strftime('Y%Y'),\
                              DATE.strftime('M%m'),\
                              DATE.strftime('D%d'),\
                              fn_header+'_2D_'+DATE.strftime('%Y%m%d')+'.nc')
            self.logger.info('loading/sampling ERA5 file '+fn)
            nc_out = F_ncread_selective(fn,np.concatenate(
                    (era5_fields,['latitude','longitude','time'])))
            # rearrange data to lon, lat, time, sort the lat dimension
            nc_out['latitude'] = nc_out['latitude'][::-1]
            for field in era5_fields:
                nc_out[field] = nc_out[field].transpose((2,1,0))[:,::-1,:]
            era_datenum = nc_out['time']/24.+693962.
            # sample era5 fields at amdar lon/lat/time
            amdar_daily_filter = (np.array(amdar_dataframe['amdar_datenum']) >= np.min(era_datenum)) & (np.array(amdar_dataframe['amdar_datenum']) <= np.max(era_datenum))
            for field in era5_fields:
                f = RegularGridInterpolator((nc_out['longitude'],nc_out['latitude'],era_datenum),
                                            nc_out[field],bounds_error=False,fill_value=np.nan)
                for lag in amdar_lag_hour:
                    fieldname = field+'%s'%lag
                    amdar_interp[fieldname][amdar_daily_filter] = f((amdar_dataframe['airport_lon'][amdar_daily_filter],
                                amdar_dataframe['airport_lat'][amdar_daily_filter],
                                amdar_dataframe['amdar_datenum'][amdar_daily_filter]+lag/24))
        
        self.amdar_era5_dataframe = pd.DataFrame(amdar_interp)
        
    def F_load_geosfp(self,geos_dir,interp_fields=['PBLTOP','PS','TROPPT'],\
                      time_collection='tavg1',
                      fn_header='subset',
                      amdar_lag_hour=[0.5]):
        start_datetime = self.start_datetime
        end_datetime = self.end_datetime
        amdar_dataframe = self.amdar_dataframe
        start_date = start_datetime.date()
        end_date = end_datetime.date()
        days = (end_date-start_date).days+1
        DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
        # initialize nan dictionary same height as amdar dataframe
        amdar_interp = {}
        for field in interp_fields:
            for lag in amdar_lag_hour:
                fieldname = field+'%s'%lag
                amdar_interp[fieldname] = np.full(amdar_dataframe['amdar_pblh'].shape,np.nan)
        # for each date, load geosfp subset (.mat files) and sample at amdar coordinate and time
        # may need to combine dates later
        cwd = os.getcwd()
        for DATE in DATES:
            self.logger.info('loading/sampling GEOS-FP file on '+DATE.strftime('%Y%m%d'))
            geos_data = {}
            os.chdir(os.path.join(geos_dir,DATE.strftime('Y%Y'),\
                              DATE.strftime('M%m'),\
                              DATE.strftime('D%d')))
            flist = glob.glob(fn_header+'*.mat')
            nstep = len(flist)
            for istep in range(nstep):
                filename = flist[istep]
                file_datetime = datetime.datetime.strptime(filename,fn_header+'_%Y%m%d_%H%M.mat')
                if not geos_data:
                    mat_data = sio.loadmat(filename,variable_names=np.concatenate((['lat','lon'],interp_fields)))
                    geos_data['lon'] = mat_data['lon'].flatten()
                    geos_data['lat'] = mat_data['lat'].flatten()
                    geos_data['datenum'] = np.zeros((nstep),dtype=np.float64)
                    for fn in interp_fields:
                        geos_data[fn] = np.zeros((len(geos_data['lon']),len(geos_data['lat']),nstep))
                        # geos fp uses 9.9999999E14 as missing value
                        mat_data[fn][mat_data[fn]>9e14] = np.nan
                        geos_data[fn][...,istep] = mat_data[fn]
                else:
                    mat_data = sio.loadmat(filename,variable_names=interp_fields)
                    for fn in interp_fields:
                        geos_data[fn][...,istep] = mat_data[fn]
                geos_data['datenum'][istep] = (file_datetime.toordinal()\
                                        +file_datetime.hour/24.\
                                        +file_datetime.minute/1440.\
                                        +file_datetime.second/86400.+366.)
            # sort geos_data time
            datenum_index = np.argsort(geos_data['datenum'])
            geos_data['datenum'] = geos_data['datenum'][datenum_index]
            for fn in interp_fields:
                geos_data[fn] = geos_data[fn][...,datenum_index]
            # sample geosfp fields at amdar lon/lat/time
            amdar_daily_filter = (np.array(amdar_dataframe['amdar_datenum']) >= np.min(geos_data['datenum'])) & (np.array(amdar_dataframe['amdar_datenum']) <= np.max(geos_data['datenum']))
            for field in interp_fields:
                f = RegularGridInterpolator((geos_data['lon'],geos_data['lat'],geos_data['datenum']),
                                            geos_data[field],bounds_error=False,fill_value=np.nan)
                for lag in amdar_lag_hour:
                    fieldname = field+'%s'%lag
                    amdar_interp[fieldname][amdar_daily_filter] = f((amdar_dataframe['airport_lon'][amdar_daily_filter],
                                amdar_dataframe['airport_lat'][amdar_daily_filter],
                                amdar_dataframe['amdar_datenum'][amdar_daily_filter]+lag/24))
        os.chdir(cwd)
        self.amdar_geosfp_dataframe = pd.DataFrame(amdar_interp)
            