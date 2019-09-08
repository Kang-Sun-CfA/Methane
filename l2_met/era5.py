# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 11:04:00 2019

@author: kangsun
"""
import cdsapi
import os
import datetime
import numpy as np
import logging

class era5(object):
    
    def __init__(self,era5_dir,\
                 west=-180.,east=180.,south=-90.,north=90.):
        """
        initiate the era5 object
        era5_dir:
            root directory for handling era5 data
        west,east,north,south:
            boundaries to subset geos fp
        created on 2019/09/07
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info('creating an instance of era5 logger')
        if not os.path.exists(era5_dir):
            self.logger.warning('era5_dir '+era5_dir+' does not exist! creating one...')
            os.makedirs(era5_dir)
        self.era5_dir = era5_dir;
        if east < west:
            east = east+360
        self.west = west
        self.east = east
        self.south = south
        self.north = north
        # era5 starts at 00:00 everyday
        self.daily_start_time = datetime.time(hour=0,minute=0)
        # initialize the connection to cdsapi
        self.client = cdsapi.Client()
        
    def F_set_time_bound(self,start_year=2004,start_month=10,start_day=1,\
                 start_hour=0,start_minute=0,start_second=0,\
                 end_year=2017,end_month=12,end_day=31,\
                 end_hour=0,end_minute=0,end_second=0):
        """
        reset start and end time.
        also create era5 time stamps covering the time bounds
        created on 2019/09/07
        """
        self.start_python_datetime = datetime.datetime(start_year,start_month,start_day,\
                                                  start_hour,start_minute,start_second)
        self.end_python_datetime = datetime.datetime(end_year,end_month,end_day,\
                                                end_hour,end_minute,end_second)
        step_hour = 1 # we know era5 is hourly
        daily_start_time = self.daily_start_time
        # extend the start/end datetime to the closest step_hour intervals
        t_array0 = datetime.datetime.combine(datetime.date(start_year,start_month,start_day),\
        daily_start_time)-datetime.timedelta(hours=step_hour)
        t_array = np.array([t_array0+datetime.timedelta(hours=int(step_hour)*i) for i in range(int(24/step_hour+2))])
        tn_array = np.array([(self.start_python_datetime-dt).total_seconds() for dt in t_array])
        era5_start_datetime = t_array[tn_array >= 0.][-1]
        
        t_array0 = datetime.datetime.combine(datetime.date(end_year,end_month,end_day),\
        daily_start_time)-datetime.timedelta(hours=step_hour)
        t_array = np.array([t_array0+datetime.timedelta(hours=int(step_hour)*i) for i in range(int(24/step_hour+2))])
        tn_array = np.array([(self.end_python_datetime-dt).total_seconds() for dt in t_array])
        era5_end_datetime = t_array[tn_array <= 0.][0]
        
        nstep = (era5_end_datetime-era5_start_datetime).total_seconds()/3600/step_hour+1
        self.era5_start_datetime = era5_start_datetime
        self.era5_end_datetime = era5_end_datetime
        self.nstep = int(nstep)
        self.logger.info('specified time from '+\
                         self.start_python_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')+
                         ' to '+self.end_python_datetime.strftime('%Y-%m-%dT%H:%M:%SZ'))
        self.logger.info('extended time from '+\
                         self.era5_start_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')+
                         ' to '+self.era5_end_datetime.strftime('%Y-%m-%dT%H:%M:%SZ'))
        self.logger.info('there will be %d'%nstep+' era5 time steps')
    
    def F_download_era5(self,file_collection_names=['reanalysis-era5-single-levels'],\
                        file_collection_fields=[['boundary_layer_height',\
                                                 'surface_pressure',\
                                                 '10m_u_component_of_wind',\
                                                 '10m_v_component_of_wind',
                                                 '100m_u_component_of_wind',\
                                                 '100m_v_component_of_wind',\
                                                 '2m_temperature',\
                                                 'skin_temperature']],\
                        download_start_hour=0.,download_end_hour=23.,\
                        fn_header='CONUS'):
        """
        download era5 data from climate data store (cds)
        file_collection_names:
            a list of era5 collection names, such as ['reanalysis-era5-single-levels','reanalysis-era5-pressure-levels']
            see https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation
            for a complete list
        download_start_hour:
            only hours larger than that will be downloaded (include that hour)
        download_end_hour:
            only hours smaller than that will be downloaded (include that hour)
        fn_header:
            header of the downloaded netcdf file
        created on 2019/09/07
        """
        cwd = os.getcwd()
        days = (self.era5_end_datetime-self.era5_start_datetime).days+1
        dates = [self.era5_start_datetime + datetime.timedelta(days=d) for d in range(days)]
        hour_str = ['%02d'%h+':00' for h in range(download_start_hour,download_end_hour+1)]
        cds_dict = {}
        cds_dict['product_type'] = 'reanalysis'
        cds_dict['grid'] = [0.25,0.25]
        cds_dict['time'] = hour_str
        cds_dict['area'] = '%.1f'%self.north+'/'+'%.1f'%self.west+'/'+'%.1f'%self.south+'/'+'%.1f'%self.east
        cds_dict['format'] = 'netcdf'
        for icollection in range(len(file_collection_names)):
            file_collection_name = file_collection_names[icollection]
            file_collection_field = file_collection_fields[icollection]
            if file_collection_name == 'reanalysis-era5-single-levels':
                collection_header = '_2D'
            elif file_collection_name == 'reanalysis-era5-pressure-levels':
                collection_header = '_3D'
                cds_dict['pressure_level'] = ['825','850','875','900','925','950','975','1000']
            else:
                collection_header = ''
            for date in dates:
                cds_dict['year'] = date.strftime("%Y")
                cds_dict['month'] = date.strftime("%m")
                cds_dict['day'] = date.strftime("%d")
                cds_dict['variable'] = file_collection_field
                download_dir = os.path.join(self.era5_dir,date.strftime('Y%Y'),\
                                   date.strftime('M%m'),\
                                   date.strftime('D%d'))
                if not os.path.exists(download_dir):
                    os.makedirs(download_dir)
                os.chdir(download_dir)
                nc_fn = fn_header+collection_header+'_'+date.strftime('%Y%m%d')+'.nc'
                self.client.retrieve(file_collection_name,cds_dict,nc_fn)
            if file_collection_name == 'reanalysis-era5-pressure-levels':
                cds_dict.pop('pressure_level')
        os.chdir(cwd)

# the following snippet is an example
from calendar import monthrange
# path to save era5 data
era5_dir = '/mnt/Data2/ERA5'

# spatial bounds
west = -135.
east = -60.
south = 20.
north = 55.
# date bounds
start_year = 2018
end_year = 2018
start_month = 8
end_month = 8
# start/end hour for daily downloading, may be tricky
# for conus
download_start_hour=16
download_end_hour=22

file_collection_names = ['reanalysis-era5-single-levels']
file_collection_fields=[['boundary_layer_height',\
                         'surface_pressure',\
                         '10m_u_component_of_wind',\
                         '10m_v_component_of_wind',
                         '100m_u_component_of_wind',\
                         '100m_v_component_of_wind',\
                         '2m_temperature',\
                         'skin_temperature']]
e = era5(era5_dir=era5_dir,west=west,east=east,south=south,north=north)
for year in range(start_year,end_year+1):
    for month in range(1,13):
        if year == start_year and month < start_month:
            continue
        elif year == end_year and month > end_month:
            continue
        # set temporal extent
        e.F_set_time_bound(start_year=year,start_month=month,start_day=1,\
                           start_hour=0,start_minute=0,start_second=0,\
                           end_year=year,end_month=month,end_day=monthrange(year,month)[-1],\
                           end_hour=23,end_minute=0,end_second=0)
        # download
        e.F_download_era5(file_collection_names=file_collection_names,
                          file_collection_fields=file_collection_fields,
                          download_start_hour=download_start_hour,
                          download_end_hour=download_end_hour,
                          fn_header='CONUS')
