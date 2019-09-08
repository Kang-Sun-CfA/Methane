#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 14:50:49 2019

@author: kangsun

python class handling GEOS FP
"""
import datetime
import numpy as np
import os
import logging
import scipy.io as sio
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator, interp1d

# dry air gas constant
R_d = 287.058
# reference gravity
g0 = 9.8
# earth average radius in m
R_earth = 6371e3

def F_variable_g(H,g0=9.8,R_earth=6371e3):
    """
    calculate g as a function of location. only altitude is considered now
    H:
        height above sea level, in m
    g0:
        standard gravity in m/s2
    R_earth:
        earth average radius in m
    """
    variable_g = g0*(R_earth/(R_earth+H))**2
    return variable_g

def F_read_geos_nc4(file_name,varnames):
    """
    read geos fp nc4 files, reverse the ndarray order as lon, lat, pressure, time
    """
    ncid = Dataset(file_name,'r')
    ncid.set_auto_mask(False)
    outp = {}
    for varname in varnames:
        data = ncid.variables[varname][:]
        outp[varname] = np.transpose(np.squeeze(data))
    ncid = None
    return outp

def F_compute_dust_aod( dst_mcol ):
    """
    compute_dust_aod function from GCNR_Chem.py by C. Chan Miller
    dst_mcol:
        sub mass column at each layer of each dust category, shape=(nlon,nlat,nlayer,ndst)
        where ndst is number of dust categories (5 for geos fp)
    """
    reff_bin       = np.array([ 0.151, 0.253, 0.402, 0.818, 1.491, 2.417, 3.721])*1e-6 # m
    aerdens_bin    = np.array([2500.0,2500.0,2500.0,2500.0,2650.0,2650.0,2650.0])
    QExt_500nm_bin = np.array([1.5736,3.6240,3.6424,2.6226,2.3682,2.2699,2.1247])
    #dst_idx        = np.array([     0,     0,     0,     0,     1,     2,     3])# this is CCM's version for GCNR
    dst_idx        = np.array([     0,     0,     0,     1,    2,     3,     4])
    dst_scl        = np.array([  0.06,  0.12,  0.24,  0.58,  1.00,  1.00,  1.00])
    
    tot_aod = np.zeros( dst_mcol.shape[0:3] )
    for n in range(0,dst_idx.shape[0]):
        tot_aod = tot_aod + np.pi * QExt_500nm_bin[n] * 0.75 /reff_bin[n] * \
                  dst_mcol[:,:,:,dst_idx[n]]*dst_scl[n] / aerdens_bin[n]
    
    return tot_aod

def F_compute_aod( aername, aer_mcol, rh ):
    """
    compute_aod function from GCNR_Chem.py by C. Chan Miller
    aername:
        choosen from 'SU', 'BC', 'OC', 'SF', 'SC'
    aer_mcol:
        sub mass column of each layer, in kg/m2, calculated by DELP/g*aerosol mass mixing ratio in kg/kg
    """
    # Relative Humidity grid
    rh_grid = [0.0,50.0,70.0,80.0,90.0,95.,99.0]
    
    # Dry aerosol density (kg/m3)
    aerdens = {}
    aerdens['SU'] = 1700.0
    aerdens['BC'] = 1800.0
    aerdens['OC'] = 1800.0
    aerdens['SF'] = 2200.0
    aerdens['SC'] = 2200.0
    
    # Extinction at 550nm
    QExt_550nm = {}
    QExt_550nm['SU'] = np.array([0.9028,0.9931,1.0630,1.1382,1.3021,1.5186,2.0735])
    QExt_550nm['BC'] = np.array([0.3755,0.3755,0.3755,0.3148,0.2682,0.2518,0.2236])
    QExt_550nm['OC'] = np.array([1.0059,0.9462,0.9463,0.9552,0.9903,1.0531,1.2725])
    QExt_550nm['SF'] = np.array([0.8986,1.3819,1.5943,1.7894,2.1810,2.6124,3.0997])
    QExt_550nm['SC'] = np.array([2.6154,2.4399,2.3946,2.3608,2.3058,2.2596,2.1751])
    
    # Effective radius [m]
    reff = {}
    reff['SU'] = np.array([ 0.121, 0.149, 0.162, 0.174, 0.198, 0.227, 0.304])*1e-6 
    reff['BC'] = np.array([ 0.035, 0.035, 0.035, 0.042, 0.049, 0.052, 0.066])*1e-6
    reff['OC'] = np.array([ 0.127, 0.139, 0.144, 0.149, 0.159, 0.171, 0.203])*1e-6
    reff['SF'] = np.array([ 0.129, 0.207, 0.233, 0.256, 0.306, 0.372, 0.613])*1e-6
    reff['SC'] = np.array([ 0.952, 1.534, 1.725, 1.899, 2.274, 2.780, 4.673])*1e-6
    
    # Interpolate to observed RH
    Qext_prof = np.interp(rh, rh_grid, QExt_550nm[aername]) 
    reff_prof = np.interp(rh, rh_grid, reff[aername]      )
    
    # Compute AOD
    aod = np.pi * np.power(reff_prof,2) * Qext_prof * \
          aer_mcol / aerdens[aername]         \
         * 0.75 /np.power(reff[aername][0],3) 
    
    return aod

class geos(object):
    
    def __init__(self,geos_dir,geos_constants_path='',\
                 west=-180.,east=180.,south=-90.,north=90.,\
                 time_collection='inst3'):
        """
        initiate the geos object
        geos_dir:
            root directory for handling geos data
        geos_constants_path:
            absolute path to the .mat file containing geos fp constants, lat, lon, surface elevation
        west,east,north,south:
            boundaries to subset geos fp
        time_collection:
            choose from inst3, tavg1, tavg3
        created on 2019/05/12
        updated on 2019/05/29 to be compatible with tavg collections
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info('creating an instance of geos logger')
        if not os.path.exists(geos_dir):
            self.logger.warning('geos_dir '+geos_dir+' does not exist! creating one...')
            os.makedirs(geos_dir)
        self.geos_dir = geos_dir;
        if east < west:
            east = east+360
        self.west = west
        self.east = east
        self.south = south
        self.north = north
        constants = sio.loadmat(geos_constants_path)
        lat0 = constants['lat']
        lon0 = constants['lon']
        tmplon = lon0-west
        tmplon[tmplon<0] = tmplon[tmplon<0]+360
        HS0 = constants['HS']
        int_lat = np.squeeze((lat0 >= south) & (lat0 <= north))
        int_lon = np.squeeze((tmplon >= 0) & (tmplon <= east-west))
        self.lat = np.squeeze(lat0[int_lat])
        self.lon = np.squeeze(tmplon[int_lon]+west)
        self.int_lon = int_lon
        self.int_lat = int_lat
        self.HS = HS0[np.ix_(int_lon,int_lat)]
        self.nlat = len(self.lat)
        self.nlon = len(self.lon)
        if time_collection == 'inst3':
            step_hour = 3
            daily_start_time = datetime.time(hour=0,minute=0)
        elif time_collection == 'tavg1':
            step_hour = 1
            daily_start_time = datetime.time(hour=0,minute=30)
        elif time_collection == 'tavg3':
            step_hour = 3
            daily_start_time = datetime.time(hour=1,minute=30)
        self.step_hour = step_hour
        self.daily_start_time = daily_start_time
        self.nlayer = 72
        self.nlayer0 = 72
        self.nlat0 = 721
        self.nlon0 = 1152
        # model top pressure in Pa
        self.ptop = 1
            
    def F_set_time_bound(self,start_year=1995,start_month=1,start_day=1,\
                 start_hour=0,start_minute=0,start_second=0,\
                 end_year=2025,end_month=12,end_day=31,\
                 end_hour=0,end_minute=0,end_second=0):
        """
        reset start and end time.
        also create geos time stamps covering the time bounds
        created on 2019/05/12
        updated on 2019/05/29 to be compatible with tavg collections
        """
        self.start_python_datetime = datetime.datetime(start_year,start_month,start_day,\
                                                  start_hour,start_minute,start_second)
        self.end_python_datetime = datetime.datetime(end_year,end_month,end_day,\
                                                end_hour,end_minute,end_second)
        step_hour = self.step_hour
        daily_start_time = self.daily_start_time
        # extend the start/end datetime to the closest step_hour intervals
        t_array0 = datetime.datetime.combine(datetime.date(start_year,start_month,start_day),\
        daily_start_time)-datetime.timedelta(hours=step_hour)
        t_array = np.array([t_array0+datetime.timedelta(hours=int(step_hour)*i) for i in range(int(24/step_hour+2))])
        tn_array = np.array([(self.start_python_datetime-dt).total_seconds() for dt in t_array])
        geos_start_datetime = t_array[tn_array >= 0.][-1]
        
        t_array0 = datetime.datetime.combine(datetime.date(end_year,end_month,end_day),\
        daily_start_time)-datetime.timedelta(hours=step_hour)
        t_array = np.array([t_array0+datetime.timedelta(hours=int(step_hour)*i) for i in range(int(24/step_hour+2))])
        tn_array = np.array([(self.end_python_datetime-dt).total_seconds() for dt in t_array])
        geos_end_datetime = t_array[tn_array <= 0.][0]
        
#        geos_start_hour = start_hour-start_hour%step_hour
#        geos_start_datetime = datetime.datetime(year=start_year,month=start_month,day=start_day,hour=geos_start_hour)
#        if end_hour > 24-step_hour or (end_hour == 24-step_hour and (end_minute > 0 or end_second > 0)):
#            geos_end_hour = 0
#            geos_end_datetime = datetime.datetime(year=end_year,month=end_month,day=end_day,hour=geos_end_hour) +datetime.timedelta(days=1)
#        elif end_hour%step_hour == 0 and end_minute == 0 and end_second == 0:
#            geos_end_hour = end_hour
#            geos_end_datetime = datetime.datetime(year=end_year,month=end_month,day=end_day,hour=geos_end_hour)
#        else:
#            geos_end_hour = (step_hour-(end_hour+1)%step_hour)%step_hour+end_hour+1
#            geos_end_datetime = datetime.datetime(year=end_year,month=end_month,day=end_day,hour=geos_end_hour)
        nstep = (geos_end_datetime-geos_start_datetime).total_seconds()/3600/step_hour+1
        self.geos_start_datetime = geos_start_datetime
        self.geos_end_datetime = geos_end_datetime
        self.nstep = int(nstep)
        self.logger.info('specified time from '+\
                         self.start_python_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')+
                         ' to '+self.end_python_datetime.strftime('%Y-%m-%dT%H:%M:%SZ'))
        self.logger.info('extended time from '+\
                         self.geos_start_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')+
                         ' to '+self.geos_end_datetime.strftime('%Y-%m-%dT%H:%M:%SZ'))
        self.logger.info('there will be %d'%nstep+' geos time steps')
        
        
    def F_download_geos(self,file_collection_names=['inst3_2d_asm_Nx'],\
                        download_start_hour=-3.,download_end_hour=27.):
        """
        download geos fp data from https://portal.nccs.nasa.gov/datashare/gmao_ops/pub/fp/das/
        use wget. may add/switch to request to be compatible with PC
        file_collection_names:
            a list of geos fp collection names, such as ['inst3_3d_asm_Nv','inst3_3d_chm_Nv','inst3_2d_asm_Nx']
            see https://gmao.gsfc.nasa.gov/GMAO_products/documents/GEOS_5_FP_File_Specification_ON4v1_2.pdf
            for a complete list
        download_start_hour:
            only hours larger than that will be downloaded
        download_end_hour:
            only hours smaller than that will be downloaded
        created on 2019/05/12
        updated on 2019/05/29 to be compatible with tavg collections
        """
        geos_url = ' https://portal.nccs.nasa.gov/datashare/gmao_ops/pub/fp/das/'
        cwd = os.getcwd()
        os.chdir(self.geos_dir)
        nstep = self.nstep
        for istep in range(nstep):
            file_datetime = self.geos_start_datetime+datetime.timedelta(hours=self.step_hour*istep)
            file_hour = file_datetime.hour+file_datetime.minute/60.+file_datetime.second/3600.
            if file_hour < download_start_hour or file_hour > download_end_hour:
                continue
            for file_collection_name in file_collection_names:
                fn = 'GEOS.fp.asm.'+file_collection_name+'.'+file_datetime.strftime("%Y%m%d_%H%M")+'.V01.nc4'
                runstr = 'wget -r -np -nH --cut-dirs=5 '+geos_url+\
                    'Y'+file_datetime.strftime("%Y")+'/M'+file_datetime.strftime("%m")+'/D'+file_datetime.strftime("%d")+'/'+fn
                os.system(runstr)
#        start_date = self.start_python_datetime.date()
#        end_date = self.end_python_datetime.date()
#        days = (end_date-start_date).days+1
#        DATES = [start_date + datetime.timedelta(days=d) for d in range(days)]
#        for DATE in DATES:
#            for file_collection_name in file_collection_names:
#                for ihour in range(download_start_hour,download_end_hour,3):
#                    fn = 'GEOS.fp.asm.'+file_collection_name+'.'+DATE.strftime("%Y%m%d")+\
#                    '_'+"{:0>2d}".format(ihour)+'00.V01.nc4'
#                    runstr = 'wget -r -np -nH --cut-dirs=5 '+geos_url+\
#                    'Y'+DATE.strftime("%Y")+'/M'+DATE.strftime("%m")+'/D'+DATE.strftime("%d")+'/'+fn
#                    os.system(runstr)
        os.chdir(cwd)
    
    def F_load_geos(self,file_collection_names=['inst3_2d_asm_Nx','inst3_3d_asm_Nv'], \
                    file_collection_fields=[['PS','U10M','V10M'],['PL','H','T','QV']]):
        """
        load geos files into memory, stack the files if there are more than 
        one time steps for future spatiotemporal interpolation
        file_collection_names:
            a list of geos fp collection names, such as ['inst3_3d_asm_Nv','inst3_3d_chm_Nv','inst3_2d_asm_Nx']
            see https://gmao.gsfc.nasa.gov/GMAO_products/documents/GEOS_5_FP_File_Specification_ON4v1_2.pdf
            for a complete list
        file_collection_fields:
            a list of variable lists, has to match file_collection_names
        updated on 2019/05/15 to include aod calculation
        """
        # fields to lump into aerosol categories
        SU_fields = ['SO4','NH4A','NO3AN1','NO3AN2','NO3AN3']
        DU_fields = ['DU001','DU002','DU003','DU004','DU005']
        SC_fields = ['SS003','SS004','SS005'] # I've no idea about sea salt
        SF_fields = ['SS001','SS002']
        OC_fields = ['OCPHILIC','OCPHOBIC']
        BC_fields = ['BCPHILIC','BCPHOBIC']
        geos_data = {}
        
        # make sure RH and DELP are there if one wants aerosols
        new_file_collection_fields = file_collection_fields.copy()
        for i in range(len(file_collection_names)):
            file_collection_name = file_collection_names[i]
            file_collection_field = file_collection_fields[i]
            
            if file_collection_name == 'inst3_3d_aer_Nv' and 'RH' not in file_collection_field:
                new_file_collection_fields[i] = np.concatenate((file_collection_fields[i],['RH']),0)
                self.logger.warning('RH has to be included in aer! add now')
            if file_collection_name == 'inst3_3d_aer_Nv' and 'DELP' not in file_collection_field:
                new_file_collection_fields[i] = np.concatenate((file_collection_fields[i],['DELP']),0)
                self.logger.warning('DELP has to be included in aer! add now')
        
        file_collection_fields = new_file_collection_fields
        # allocate memory for geos_data        
        for i in range(len(file_collection_names)):
            file_collection_name = file_collection_names[i]
            file_collection_field = file_collection_fields[i]
            
            for fn in file_collection_field:
                if '3d' in file_collection_name:
                    geos_data[fn] = np.zeros((self.nlon,self.nlat,self.nlayer,self.nstep))
                else:
                    geos_data[fn] = np.zeros((self.nlon,self.nlat,self.nstep))
                # calculate lapse rate if all necessary data are available
                if file_collection_name == 'inst3_3d_asm_Nv' and \
                'H' in file_collection_field and 'T' in file_collection_field:
                    geos_data['lapse_rate'] = np.zeros((self.nlon,self.nlat,self.nstep))
                # allocating for AOD calculations, dust
                if file_collection_name == 'inst3_3d_aer_Nv' and \
                set(file_collection_field).intersection(DU_fields):
                    geos_data['ODU'] = np.zeros((self.nlon,self.nlat,self.nlayer,self.nstep))
                # sulfate
                if file_collection_name == 'inst3_3d_aer_Nv' and \
                set(file_collection_field).intersection(SU_fields):
                    geos_data['OSU'] = np.zeros((self.nlon,self.nlat,self.nlayer,self.nstep))
                # organic carbon
                if file_collection_name == 'inst3_3d_aer_Nv' and \
                set(file_collection_field).intersection(OC_fields):
                    geos_data['OOC'] = np.zeros((self.nlon,self.nlat,self.nlayer,self.nstep))
                # black carbon
                if file_collection_name == 'inst3_3d_aer_Nv' and \
                set(file_collection_field).intersection(BC_fields):
                    geos_data['OBC'] = np.zeros((self.nlon,self.nlat,self.nlayer,self.nstep))
                # fine sea salt
                if file_collection_name == 'inst3_3d_aer_Nv' and \
                set(file_collection_field).intersection(SF_fields):
                    geos_data['OSF'] = np.zeros((self.nlon,self.nlat,self.nlayer,self.nstep))
                # coarse sea salt
                if file_collection_name == 'inst3_3d_aer_Nv' and \
                set(file_collection_field).intersection(SC_fields):
                    geos_data['OSC'] = np.zeros((self.nlon,self.nlat,self.nlayer,self.nstep))
        
        self.matlab_datenum = np.zeros((self.nstep),dtype=np.float64)
        self.tai93 = np.zeros((self.nstep),dtype=np.float64)
        self.file_collection_names = file_collection_names
        self.file_collection_fields = file_collection_fields
        
        for istep in range(self.nstep):
            file_datetime = self.geos_start_datetime+datetime.timedelta(hours=self.step_hour*istep)
            file_dir = os.path.join(self.geos_dir,file_datetime.strftime('Y%Y'),file_datetime.strftime('M%m'),file_datetime.strftime('D%d'))
            file_datenum = (file_datetime.toordinal()\
                                +file_datetime.hour/24.\
                                +file_datetime.minute/1440.\
                                +file_datetime.second/86400.+366.)
            self.matlab_datenum[istep] = file_datenum
            
            for i in range(len(file_collection_names)):
                file_path = os.path.join(file_dir,'GEOS.fp.asm.'+file_collection_names[i]+file_datetime.strftime('.%Y%m%d_%H%M')+'.V01.nc4')
                self.logger.info('loading '+file_path)
                if i == 0:
                    self.tai93[istep] = F_read_geos_nc4(file_path,['TAITIME'])['TAITIME']
                file_collection_field= file_collection_fields[i]
                outp = F_read_geos_nc4(file_path,file_collection_field)
                
                for fn in file_collection_field:
                    geos_data[fn][...,istep] = outp[fn][np.ix_(self.int_lon,self.int_lat)]
                
                # calculate lapse rate if all necessary data are available
                if file_collection_names[i] == 'inst3_3d_asm_Nv' and \
                'H' in file_collection_field and 'T' in file_collection_field:
                    self.logger.info('calculating lapse rate')
                    geos_data['lapse_rate'][...,istep] = -np.squeeze((geos_data['T'][:,:,-1,istep]-geos_data['T'][:,:,-2,istep])\
                    /(geos_data['H'][:,:,-1,istep]-geos_data['H'][:,:,-2,istep]))
                
                # calculate sulfate aod if all necessary data are available
                loaded_aer_fields = set(file_collection_field).intersection(SU_fields)
                if file_collection_names[i] == 'inst3_3d_aer_Nv' and loaded_aer_fields:
                    self.logger.info('calculating sulfate aod')
                    # calculate mass column
                    aer_mcol = np.zeros((self.nlon,self.nlat,self.nlayer))
                    for aer_field in loaded_aer_fields:
                        aer_mcol = aer_mcol+geos_data['DELP'][...,istep]/g0*geos_data[aer_field][...,istep]
                    # Compute AOD
                    geos_data['OSU'][...,istep] = F_compute_aod( 'SU', aer_mcol, 100*geos_data['RH'][...,istep])
                
                # calculate organic carbon aod if all necessary data are available
                loaded_aer_fields = set(file_collection_field).intersection(OC_fields)
                if file_collection_names[i] == 'inst3_3d_aer_Nv' and loaded_aer_fields:
                    self.logger.info('calculating organic carbon aod')
                    # calculate mass column
                    aer_mcoli = geos_data['DELP'][...,istep]/g0*geos_data['OCPHILIC'][...,istep]
                    aer_mcolo = geos_data['DELP'][...,istep]/g0*geos_data['OCPHOBIC'][...,istep]
                    # Compute AOD
                    geos_data['OOC'][...,istep] = F_compute_aod( 'OC', aer_mcoli, 100*geos_data['RH'][...,istep])\
                    + F_compute_aod( 'OC', aer_mcolo, 0*geos_data['RH'][...,istep])
                
                # calculate BC aod if all necessary data are available
                loaded_aer_fields = set(file_collection_field).intersection(BC_fields)
                if file_collection_names[i] == 'inst3_3d_aer_Nv' and loaded_aer_fields:
                    self.logger.info('calculating black carbon aod')
                    # calculate mass column
                    aer_mcoli = geos_data['DELP'][...,istep]/g0*geos_data['BCPHILIC'][...,istep]
                    aer_mcolo = geos_data['DELP'][...,istep]/g0*geos_data['BCPHOBIC'][...,istep]
                    # Compute AOD
                    geos_data['OBC'][...,istep] = F_compute_aod( 'BC', aer_mcoli, 100*geos_data['RH'][...,istep])\
                    + F_compute_aod( 'BC', aer_mcolo, 0*geos_data['RH'][...,istep])
                
                # calculate fine sea salt aod if all necessary data are available
                loaded_aer_fields = set(file_collection_field).intersection(SF_fields)
                if file_collection_names[i] == 'inst3_3d_aer_Nv' and loaded_aer_fields:
                    self.logger.info('calculating fine sea salt aod')
                    # calculate mass column
                    aer_mcol = np.zeros((self.nlon,self.nlat,self.nlayer))
                    for aer_field in loaded_aer_fields:
                        aer_mcol = aer_mcol+geos_data['DELP'][...,istep]/g0*geos_data[aer_field][...,istep]
                    # Compute AOD
                    geos_data['OSF'][...,istep] = F_compute_aod( 'SF', aer_mcol, 100*geos_data['RH'][...,istep])
                
                # calculate coarse sea salt aod if all necessary data are available
                loaded_aer_fields = set(file_collection_field).intersection(SC_fields)
                if file_collection_names[i] == 'inst3_3d_aer_Nv' and loaded_aer_fields:
                    self.logger.info('calculating coarse sea salt aod')
                    # calculate mass column
                    aer_mcol = np.zeros((self.nlon,self.nlat,self.nlayer))
                    for aer_field in loaded_aer_fields:
                        aer_mcol = aer_mcol+geos_data['DELP'][...,istep]/g0*geos_data[aer_field][...,istep]
                    # Compute AOD
                    geos_data['OSC'][...,istep] = F_compute_aod( 'SC', aer_mcol, 100*geos_data['RH'][...,istep])
                
                # calculate dust aod if all necessary data are available
                loaded_aer_fields = set(file_collection_field).intersection(DU_fields)
                if file_collection_names[i] == 'inst3_3d_aer_Nv' and loaded_aer_fields:
                    self.logger.info('calculating dust aod')
                    # calculate mass column
                    ndst = len(DU_fields)
                    dst_mcol = np.zeros((self.nlon,self.nlat,self.nlayer,ndst))
                    for idst in range(ndst):
                        dst_mcol[...,idst] = geos_data['DELP'][...,istep]/g0*geos_data[DU_fields[idst]][...,istep]
                    # Compute AOD
                    geos_data['ODU'][...,istep] = F_compute_dust_aod(dst_mcol)
                
        idx = np.argsort(self.lon)
        self.lon = np.sort(self.lon)
        self.geos_data =  {k:v[idx,] for (k,v) in geos_data.items()}
    
    def F_save_geos_data2mat(self,if_delete_nc=False,fn_header='subset'):
        """
        save geos_data loaded by F_load_geos to mat file
        if_delete_nc:
            true if delete geos netcdf files
        fn_header:
            header string of the subsetted mat files
        created on 2019/05/25
        updated on 2019/06/20 to include file name header
        """
        if self.nstep != 1:
            self.logger.error('nstep = '+'%d'%self.nstep+', this function only works for single time step (start=end)!')
            return
        from scipy.io import savemat
        file_datetime = self.geos_start_datetime
        file_dir = os.path.join(self.geos_dir,file_datetime.strftime('Y%Y'),\
                                   file_datetime.strftime('M%m'),\
                                   file_datetime.strftime('D%d'))
        mat_fn = os.path.join(file_dir,fn_header+'_'+file_datetime.strftime('%Y%m%d_%H%M')+'.mat')
        save_dict = self.geos_data
        save_dict = {k:np.squeeze(v) for (k,v) in save_dict.items()}
        save_dict['lon'] = self.lon.flatten()
        save_dict['lat'] = self.lat.flatten()
        # save in single precision to save space
        save_dict = {k:np.float32(v) for (k,v) in save_dict.items()}
        savemat(mat_fn,save_dict)
        if not if_delete_nc:
            return
        for i in range(len(self.file_collection_names)):
            file_path = os.path.join(file_dir,'GEOS.fp.asm.'+\
                                     self.file_collection_names[i]+\
                                     file_datetime.strftime('.%Y%m%d_%H%M')+'.V01.nc4')
            self.logger.info('deleting '+file_path)
            os.remove(file_path)
        
    def F_interp_geos(self,sounding_lon,sounding_lat,sounding_tai93=None,sounding_datenum=None,\
                      sounding_dem=None,interp_var=['PS'],Ap=None,Bp=None):
        """
        resample from geos_data at sounding locations and time
        souding_lon:
            longitude of level 2 pixels
        souding_lat:
            latitude of level 2 pixels
        souding_tai93:
            seconds after 1993-1-1 at level 2 sounding time, preferred way for timing as leap seconds are included
        souding_datenum:
            matlab datenum of level 2 pixels.
        souding_dem:
            surface elevation at sounding location, default to geos fp dem
        interp_var:
            variables to be resampled from geos
        created on 2019/05/13
        """
        if sounding_tai93 is None and sounding_datenum is None:
            self.logger.error('at least one of tai93 and datenum have to be provided!')
        elif sounding_tai93 is None:
            self.logger.info('using matlab datenum as time')
            sounding_time = sounding_datenum
            geos_time = self.matlab_datenum
        else:
            self.logger.info('using tai93 as time')
            sounding_time = sounding_tai93
            geos_time = self.tai93
        pts_2d = (sounding_lon,sounding_lat)
        my_interpolating_function = RegularGridInterpolator((self.lon,self.lat),self.HS)
        sounding_HS = my_interpolating_function(pts_2d)
        if sounding_dem is None:
            self.logger.warning('DEM at sounding locations are not provided! Using GEOS FP HS as sounding DEM...')
            self.logger.warning('no hypsometric adjust will be made')
            sounding_dem = sounding_HS
            do_hypometric = False
        else:# hyposmetric correction
            do_hypometric = True
        sounding_data = {}
        sounding_data['sounding_dem'] = sounding_dem
        sounding_data['sounding_HS'] = sounding_HS
        pts_3d = (sounding_lon,sounding_lat,sounding_time)
        data3d = []
        for var in interp_var:
            if var not in self.geos_data.keys():
                self.logger.warning(var+' is not available in your loaded geos data!')
            # sample surface pressure
            elif var == 'PS':
                self.logger.info('interpolating surface pressure')
                my_interpolating_function = \
                RegularGridInterpolator((self.lon,self.lat,geos_time),self.geos_data['PS'])
                sounding_PS = my_interpolating_function(pts_3d)
                if do_hypometric:
                    self.logger.info('making hypsometric correction...')
                    # get geos lowest layer temperature
                    my_interpolating_function = \
                    RegularGridInterpolator((self.lon,self.lat,geos_time),np.squeeze(self.geos_data['T'][:,:,-1,:]))
                    sounding_T0 = my_interpolating_function(pts_3d)
                    # get geos lowest layer lapse rate
                    my_interpolating_function = \
                    RegularGridInterpolator((self.lon,self.lat,geos_time),self.geos_data['lapse_rate'])
                    sounding_lapse_rate = my_interpolating_function(pts_3d)
                    # get spatially variable g due to elevation
                    variable_g = F_variable_g(sounding_HS,g0=g0,R_earth=R_earth)
                    sounding_psurf = sounding_PS*\
                    (sounding_T0/(sounding_T0-sounding_lapse_rate*(sounding_dem-sounding_HS)))\
                    **(-variable_g/R_d/sounding_lapse_rate)
                else:
                    sounding_psurf = sounding_PS
                sounding_data['sounding_psurf'] = sounding_psurf
                sounding_data['sounding_PS'] = sounding_PS
            # sample 2d fields other than PS, need 3d interpolation (with time)
            elif self.geos_data[var].shape == (self.nlon,self.nlat,self.nstep):
                self.logger.info('interpolating '+var)
                my_interpolating_function = \
                RegularGridInterpolator((self.lon,self.lat,geos_time),self.geos_data[var])
                sounding_data['sounding_'+var] = my_interpolating_function(pts_3d)
            # sample 3d fields, need 4d interpolation (with time)
            elif self.geos_data[var].shape == (self.nlon,self.nlat,self.nlayer,self.nstep):
                if 'lev_lev' not in locals():
                    # use model layer index as the vertical coordinate
                    lev = np.arange(self.nlayer)
                    lon_lev = np.repeat(sounding_lon[...,np.newaxis],self.nlayer,axis=sounding_lon.ndim)#np.tile(sounding_lon,(self.nlayer,1))
                    lat_lev = np.repeat(sounding_lat[...,np.newaxis],self.nlayer,axis=sounding_lat.ndim)#np.tile(sounding_lat,(self.nlayer,1))
                    time_lev = np.repeat(sounding_time[...,np.newaxis],self.nlayer,axis=sounding_time.ndim)#np.tile(sounding_time,(self.nlayer,1))
                    # I don't know to create it one-step, why not a for loop
                    lev_lev = np.ones(lon_lev.shape)
                    for ilayer in range(self.nlayer):
                        lev_lev[...,ilayer] = lev_lev[...,ilayer]*lev[ilayer]
                                        
                self.logger.info('interpolating '+var)
                my_interpolating_function = \
                RegularGridInterpolator((self.lon,self.lat,lev,geos_time),self.geos_data[var])
                sounding_data['sounding_'+var] = my_interpolating_function((lon_lev,lat_lev,lev_lev,time_lev))
                data3d = np.concatenate((data3d,['sounding_'+var]))
        self.data3d = data3d
        s = slice(None)
        if (Ap is not None) and (Bp is not None):
            self.logger.info('Interpreting 3D fields to custom eta grids')
            self.logger.info('nlayer = %d'%(len(Ap)-1))
            nlevel1 = len(Ap)
            nlayer1 = len(Ap)-1
            npixel = len(np.ravel(sounding_data['sounding_psurf'],order='F'))
            pixel_shape = sounding_data['sounding_psurf'].shape
            
            eta_data = {}
            eta_data['pedge'] = np.zeros((pixel_shape+(nlevel1,)))
            eta_data['Tedge'] = np.zeros((pixel_shape+(nlevel1,)))
            eta_data['pmid'] = np.zeros((pixel_shape+(nlayer1,)))
            for d3d in data3d:
                if d3d == 'sounding_DELP':
                    self.logger.warning('DELP cannot be interpolated directly. skipping for now...')
                    continue
                eta_data[d3d] = np.zeros((pixel_shape+(nlayer1,)))
            for ip in range(npixel):
                pixel_idx = np.unravel_index(ip,pixel_shape,order='F')
                pedge = sounding_data['sounding_psurf'][pixel_idx]*Bp+Ap
                pmid = (pedge[0:-1]+pedge[1:])/2
                full_idx = pixel_idx+(s,)
                eta_data['pedge'][full_idx] = pedge
                eta_data['pmid'][full_idx] = pmid
                PL = sounding_data['sounding_PL'][full_idx]
                for d3d in data3d:
                    if d3d == 'sounding_T':
                        T = sounding_data['sounding_T'][full_idx]
                        f_interp1d = interp1d(PL,T,fill_value="extrapolate")
                        eta_data['Tedge'][full_idx] = f_interp1d(pedge)
                    elif d3d == 'sounding_DELP':
                        continue
                    full_prof = sounding_data[d3d][full_idx]
                    f_interp1d = interp1d(PL,full_prof,fill_value="extrapolate")
                    eta_data[d3d][full_idx] = f_interp1d(pmid)
                
        
            
        self.eta_data = eta_data
        self.sounding_data = sounding_data
#    def F_regularize_geos3d(self):
#        