# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 21:22:34 2021

@author: kangsun
"""
import numpy as np
import pandas as pd
import os, glob, sys
import logging
from collections import OrderedDict
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from scipy.signal import peak_widths
from netCDF4 import Dataset
import datetime as dt
from astropy.convolution import convolve_fft

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

def F_center_of_mass(xx,yy):
    mask = (~np.isnan(xx)) & (~np.isnan(yy))
    xx = xx[mask]
    yy = yy[mask]
    return np.trapz(xx*yy,xx)/np.trapz(yy,xx)

def F_peak_width(isrfx,isrfy,percent=0.5):
    if all(np.isnan(isrfy)):
        return np.nan
    # peaks,_ = find_peaks(isrfy,np.nanmax(isrfy)*0.5)
    peaks = np.array([np.argmax(isrfy)])
    result = peak_widths(isrfy,peaks,rel_height=percent)
    dx = np.abs(np.nanmean(np.diff(isrfx)))
    return (result[0]*dx).squeeze()

def F_center2edge(lon,lat):
    '''
    function to shut up complain of pcolormesh like 
    MatplotlibDeprecationWarning: shading='flat' when X and Y have the same dimensions as C is deprecated since 3.3.
    create grid edges from grid centers
    '''
    res=np.mean(np.diff(lon))
    lonr = np.append(lon-res/2,lon[-1]+res/2)
    res=np.mean(np.diff(lat))
    latr = np.append(lat-res/2,lat[-1]+res/2)
    return lonr,latr

def F_stray_light_input(strayLightKernelPath,rowExtent=400,colExtent=400,
                        rowCenterMask=5,colCenterMask=7):
    """
    options for stray light correction
    strayLightKernelPath:
        path to the stray light deconvolution kernel
    rowExtent:
        max row difference away from center (0)
    colExtent:
        max column difference away from center (0)
    rowCenterMask:
        +/-rowCenterMask rows of the kernel will be masked to be 0
    colCenterMask:
        +/-colCenterMask columns of the kernel will be masked to be 0
    """
    d = loadmat(strayLightKernelPath)
    rowFilter = (d['rowAllGrid'].squeeze() >= -np.abs(rowExtent)) & \
    (d['rowAllGrid'].squeeze() <= np.abs(rowExtent))
    colFilter = (d['colAllGrid'].squeeze() >= -np.abs(colExtent)) & \
    (d['colAllGrid'].squeeze() <= np.abs(colExtent))
    reducedRow = d['rowAllGrid'].squeeze()[rowFilter]
    reducedCol = d['colAllGrid'].squeeze()[colFilter]
    strayLightKernel = d['medianStrayLight'][np.ix_(rowFilter,colFilter)]
    strayLightKernel[strayLightKernel<0] = 0.
    strayLightKernel = strayLightKernel/np.nansum(strayLightKernel)
    centerRowFilter = (reducedRow >= -np.abs(rowCenterMask)) & \
    (reducedRow <= np.abs(rowCenterMask))
    centerColFilter = (reducedCol >= -np.abs(colCenterMask)) & \
    (reducedCol <= np.abs(colCenterMask))
    strayLightKernel[np.ix_(centerRowFilter,centerColFilter)] = 0
    return strayLightKernel

class Merged_Frame(dict):
    ''' 
    merged exposures with a range of integration times
    '''
    def __init__(self,ISSF_Exposure_list,limits=None,normalize_peak=True):
        self.logger = logging.getLogger(__name__)
        if limits is None:
            limits = np.ones(ISSF_Exposure_list.shape)*5000
            limits[-1] = 20000
        # make sure integration time goes from high to low
        ISSF_Exposure_list = np.array(ISSF_Exposure_list)[np.argsort([expo['int_time'] for expo in ISSF_Exposure_list])][::-1]
        for (i,(expo,limit)) in enumerate(zip(ISSF_Exposure_list,limits)):
            tmp_data = np.ma.masked_where(expo['data']>limit, expo['data'])/expo['int_time']
            if i == 0:
                tmp_data0 = tmp_data
                data = tmp_data
            elif i == 1:
                data = np.ma.where(np.ma.getmask(tmp_data0),tmp_data,tmp_data0)
            else:
                data = np.ma.where(np.ma.getmask(data),tmp_data,data)
        data = data.filled(np.nan)
        #data[data < 0] = np.nan
        if normalize_peak:
            data = data/np.nanmax(data)
        self.add('data',data)
        max_idx=np.unravel_index(np.nanargmax(self['data']),self['data'].shape)
        max_col_idx = max_idx[1]
        max_row_idx = max_idx[0]
        self.add('max_col_idx',max_col_idx)
        self.add('max_row_idx',max_row_idx)
        self.add('rows_1based',expo['rows_1based'])
        self.add('cols_1based',expo['cols_1based'])
        self.add('wavelength', expo['wavelength'])
        self.get_spatial_response()
    def add(self,key,value):
        self.__setitem__(key,value)
    def get_spectral_response(self,spectral_extent=20,spatial_extent=100,
                              normalize_peak=True):
        col_mask = (self['cols_1based'] >= self['max_col_idx']-spectral_extent) &\
            (self['cols_1based'] <= self['max_col_idx']+spectral_extent)
        row_mask = (self['rows_1based'] >= self['max_row_idx']-spatial_extent) &\
            (self['rows_1based'] <= self['max_row_idx']+spatial_extent)
        spectral_response = np.nansum(self['data'][np.ix_(row_mask,col_mask)],axis=0)
        x = self['cols_1based'][col_mask][~np.isnan(spectral_response)]
        y = spectral_response[~np.isnan(spectral_response)]
        spectral_response = spectral_response/np.trapz(y,x)
        sp = np.full(len(self['cols_1based']),np.nan)
        sp[col_mask] = spectral_response
        if normalize_peak:
            spectral_response = spectral_response/np.nanmax(spectral_response)
        peak_col_1based = np.trapz(x*y,x)/np.trapz(y,x)
        self.add('spectral_response',sp)
        self.add('peak_col_1based',peak_col_1based)
        
    def plot_spectral_response(self,existing_ax=None,scale='log',extent=200):
        if existing_ax is None:
            self.logger.info('axes not supplied, creating one')
            fig,ax = plt.subplots(1,1,figsize=(10,5))
        else:
            fig = None
            ax = existing_ax
        figout = {}
        figout['fig'] = fig
        figout['ax'] = ax
        
        if scale == 'log':
            y = self['spectral_response'].copy()
            y[y<0]=np.nan
            figout['plot'] = ax.plot(self['cols_1based'],y)
            ax.set_yscale('log')
        else:
            figout['plot'] = ax.plot(self['cols_1based'],self['spectral_response'])
        ax.set_xlim((np.max([1,self['max_col_idx']-extent]),np.min([self['max_col_idx']+extent,2048])));
    def get_spatial_response(self,spectral_extent=100,normalize_peak=True):
        col_mask = (self['cols_1based'] >= self['max_col_idx']-spectral_extent) &\
            (self['cols_1based'] <= self['max_col_idx']+spectral_extent)
        spatial_response = np.nansum(self['data'][:,col_mask],axis=1)
        # x = self['rows_1based'][~np.isnan(spatial_response)]
        # y = spatial_response[~np.isnan(spatial_response)]
        # spatial_response = spatial_response/np.trapz(y,x)
        if normalize_peak:
            spatial_response = spatial_response/np.nanmax(spatial_response)
        # peak_row_1based = np.trapz(x*y,x)/np.trapz(y,x)
        self.add('spatial_response',spatial_response)
        # self.add('peak_row_1based',peak_row_1based)
    def plot_spatial_response(self,existing_ax=None,scale='log',extent=200):
        if existing_ax is None:
            self.logger.info('axes not supplied, creating one')
            fig,ax = plt.subplots(1,1,figsize=(10,5))
        else:
            fig = None
            ax = existing_ax
        figout = {}
        figout['fig'] = fig
        figout['ax'] = ax
        
        if scale == 'log':
            y = self['spatial_response'].copy()
            y[y<0]=np.nan
            figout['plot'] = ax.plot(self['rows_1based'],y)
            ax.set_yscale('log')
        else:
            figout['plot'] = ax.plot(self['rows_1based'],self['spatial_response'])
        ax.set_xlim((np.max([1,self['max_row_idx']-extent]),np.min([self['max_row_idx']+extent,2048])));
    def plot(self,existing_ax=None,scale='log',**kwargs):
        if existing_ax is None:
            self.logger.info('axes not supplied, creating one')
            fig,ax = plt.subplots(1,1,figsize=(10,5))
        else:
            fig = None
            ax = existing_ax
        figout = {}
        figout['fig'] = fig
        figout['ax'] = ax
        if scale == 'log':
            from matplotlib.colors import LogNorm
            if 'vmin' in kwargs:
                inputNorm = LogNorm(vmin=kwargs['vmin'],vmax=kwargs['vmax'])
                kwargs.pop('vmin');
                kwargs.pop('vmax');
            else:
                inputNorm = LogNorm()
            figout['pc'] = ax.pcolormesh(*F_center2edge(self['cols_1based'],self['rows_1based']),
                                         self['data'],norm=inputNorm,
                                         **kwargs)
        else:
            figout['pc'] = ax.pcolormesh(*F_center2edge(self['cols_1based'],self['rows_1based']),
                                         self['data'],**kwargs)
        return figout

class ISSF_Exposure(dict):
    """Exposure of a single laser wavelength
    
    Represents a single frame of the 2D detector array. It is a customized class based on dictionary
    
    Items:
        nrow: number of rows of the frame.
        ncol: number of columns of the frame.
        rows_1based: an integer array starting from 1 as the index of rows.
        cols_1based: an integer array starting from 1 as the index of columns.
        wavelength: float number indicating the laser wavelength in nm for this exposure.
        power: laser power in linear unit (usually mW) for this exposure, remember to convert from dB.
        int_time: integration time for the exposure, ms and s are OK but need to be consistent.
        if_power_normalized: boolean to indicate if power has been normalized (True) to avoid duplication.
        if_time_normalized: boolean to indicate if int_time has been normalized (True) to avoid duplication.
        data: 2D float array with shape nrow by ncol for the detector data.
        
    Methods:
        __init__: initializes instance and key attributes.
        add: adds attributes.
        flip_columns: flips columns of data. We want the column index to increase with wavelength.
        flip_rows: flips rows. Useful for MethaneAIR.
        running_average_rows: 
            replaces columns near the exposure x all rows with running average. Only use
            it for ISRF data, not straylight data.
        normalize_power: divides data by power.
        normalize_time: divides data by int_time.
        plot: plots the data as 2D pcolormesh.
        remove_straylight: 
            removes straylight from the data array. More work is needed for MethaneSAT straylight correction.
        
    """
    def __init__(self,wavelength,data,nrow=None,ncol=None,power=None,int_time=None):
        self.logger = logging.getLogger(__name__)
        # self.logger.info('creating an ISSF_Exposure instance')
        if nrow is None:
            nrow = data.shape[0]
        else:
            if nrow != data.shape[0]:
                self.logger.error('row dimension inconsistent')
        self.add('rows_1based',np.arange(1,nrow+1,dtype=np.int))
        self.add('nrow',int(nrow))
        if ncol is None:
            ncol = data.shape[1]
        else:
            if ncol!= data.shape[1]:
                self.logger.error('column dimension inconsistent')
        self.add('cols_1based',np.arange(1,ncol+1,dtype=np.int))
        self.add('ncol',int(ncol))
        self.add('wavelength', wavelength)
        self.add('data', data)
        self.add('power',power)
        self.add('if_power_normalized',False)
        self.add('int_time',int_time)
        self.add('if_time_normalized',False)
    def add(self,key,value):
        self.__setitem__(key,value)
    def flip_columns(self):
        self['data'] = self['data'][:,::-1]
    def flip_rows(self):
        self['data'] = self['data'][::-1,:]
    def running_average_rows(self,window_size,col_extent=20):
        '''
        using astropy convolution function to running-average across rows. assuming boxcar kernel
        '''
        if col_extent is None:
            self['data'] = convolve_fft(self['data'],
                                        np.ones((window_size,1)),
                                        normalize_kernel=True)
        else:
            max_col = self['cols_1based'][np.argmax(np.nansum(self['data'],axis=0))]
            self.logger.info('Running averaging is limited to {}+/-{}'.format(max_col,col_extent))
            col_mask = (self['cols_1based'] >= max_col-col_extent) & (self['cols_1based'] <= max_col+col_extent)
            self['data'][:,col_mask] = convolve_fft(self['data'][:,col_mask],
                                                    np.ones((window_size,1)),
                                                    normalize_kernel=True)
    def normalize_power(self):
        if self['if_power_normalized'] == True:
            self.logger.warning('already normalized by power!')
            return
        if self['power'] is None:
            self.logger.error('you need to provide a laser power value to normalize power!')
            return
        self['data'] = self['data']/self['power']
        self['if_power_normalized']=True
    def normalize_time(self):
        if self['if_time_normalized'] == True:
            self.logger.warning('already normalized by integration time!')
            return
        if self['int_time'] is None:
            self.logger.error('you need to provide an integration time to normalize time!')
            return
        self['data'] = self['data']/self['int_time']
        self['if_time_normalized']=True
    def plot(self,existing_ax=None,scale='log',**kwargs):
        if existing_ax is None:
            self.logger.info('axes not supplied, creating one')
            fig,ax = plt.subplots(1,1,figsize=(10,5))
        else:
            fig = None
            ax = existing_ax
        figout = {}
        figout['fig'] = fig
        figout['ax'] = ax
        if scale == 'log':
            from matplotlib.colors import LogNorm
            if 'vmin' in kwargs:
                inputNorm = LogNorm(vmin=kwargs['vmin'],vmax=kwargs['vmax'])
                kwargs.pop('vmin');
                kwargs.pop('vmax');
            else:
                inputNorm = LogNorm()
            figout['pc'] = ax.pcolormesh(*F_center2edge(self['cols_1based'],self['rows_1based']),
                                         self['data'],norm=inputNorm,
                                         **kwargs)
        else:
            figout['pc'] = ax.pcolormesh(*F_center2edge(self['cols_1based'],self['rows_1based']),
                                         self['data'],**kwargs)
        return figout
    def remove_straylight(self,K_far,sum_K_far=None,n_iter=3):
        if K_far is None:
            self.logger.warning('kernel not provided, return')
            return
        
        if sum_K_far is None:
            sum_K_far = np.nansum(K_far)
        new_data = self['data'].copy()
        for i_iter in range(n_iter):
            new_data = (self['data']-convolve_fft(new_data, K_far, normalize_kernel=False))/(1-sum_K_far)
            self.logger.info('{} iteration done'.format(i_iter))
        self['data'] = new_data

class Central_Wavelength(list):
    '''
    a list of multiple laser wavelength steps around a central wavelength
    each element should be an ISSF_Exposure object
    '''
    def __init__(self,central_wavelength,
                 delta_wavelength_range=None,
                 instrum=None):
        self.logger = logging.getLogger(__name__)
        self.central_wavelength = central_wavelength
        self.delta_wavelength_range = delta_wavelength_range
        if instrum is None:
            self.logger.warning('instrum should be provided as either MethaneAIR or MethaneSAT, assuming MethaneAIR')
            instrum = 'MethaneAIR'
        if instrum.lower() == 'methaneair':
            instrum = 'MethaneAIR'
            nrow = 1280
            ncol = 1024
        elif instrum.lower() == 'methanesat':
            instrum = 'MethaneSAT'
            nrow = 2048
            ncol = 2048
        else:
            self.logger.error('what are you talking about?!')
        self.instrum = instrum
        self.nrow = nrow
        self.ncol = ncol
    
    def read_jf_nc(self,nc_fn,micro_window=None,micro_step=1):
        '''read nc files from Jonathan Franklin
        micro_window:
            window size in nm to subset the wavelength micro steps
        micro_step:
            if thinning the micro steps
        '''
        self.logger.info('loading {}'.format(nc_fn))
        nc = Dataset(nc_fn,'r')
        w = nc['Wavelength'][:]
        if self.central_wavelength < w.min() or self.central_wavelength > w.max():
            self.logger.error('central wavelength at {}, not compatible with file wavelength {}-{}'.format(self.central_wavelength,w.min(),w.max()))
        if micro_window is not None:
            wmask = (w>=self.central_wavelength-micro_window/2)&(w<=self.central_wavelength+micro_window/2)
        else:
            wmask = np.ones(len(w),dtype=bool)
        stepmask = np.zeros(len(w),dtype=bool)
        stepmask[::micro_step] = True
        wmask = wmask & stepmask
        if np.sum(wmask) < len(w):
            self.logger.warning('{} wavelengths will be reduced to {}'.format(len(w),np.sum(wmask)))
        for idx in np.arange(len(w))[wmask]:
            expo = ISSF_Exposure(wavelength=w[idx],data=nc['Data'][...,idx],
                                 nrow=2048,ncol=2048,
                                 int_time=None)
            expo.flip_columns()
            self.logger.info('loading {:.3f} nm'.format(w[idx]))
            self.append(expo)
        
        nc.close()
    def read_MethaneSAT(self,info_csv_path,
                        data_dir,which_band,
                        if_remove_straylight=False,
                        K_far=None,
                        straylight_kernel_path=None,
                        straylight_n_iter=3):
        if self.instrum != 'MethaneSAT':
            self.logger.error('this works only for MethaneSAT!')
        from scipy.io import loadmat
        df = pd.read_csv(info_csv_path,sep=',')
        bright_idx = np.where((df['LASER_WL']>=self.central_wavelength+self.delta_wavelength_range[0]) &
                           (df['LASER_WL']<=self.central_wavelength+self.delta_wavelength_range[1]))[0]
        dark_idx = np.where((df['Img_Source']=='Background') & (df.index<=bright_idx[0]))[-1]
        df_bright = df.iloc[bright_idx]
        df_dark = df.iloc[dark_idx]
        bright_fn = np.array([os.path.join(data_dir,df_row.Name.split('.')[0]+'.mat') for (i,df_row) in df_bright.iterrows()])
        df_bright.insert(0,'mat_path',bright_fn.copy(),True)
        dark_fn = np.array([os.path.join(data_dir,df_row.Name.split('.')[0]+'.mat') for (i,df_row) in df_dark.iterrows()])
        df_dark.insert(0,'mat_path',dark_fn.copy(),True)
        # df_dark._set_item('mat_path',dark_fn.copy())
        self.df_dark = df_dark
        self.df_bright = df_bright
        self.logger.info('loading dark '+df_dark.iloc[0].mat_path)
        dark = loadmat(df_dark.iloc[0].mat_path,squeeze_me=True)['data']
        for (irow,df_row) in df_bright.iterrows():
            
            if not os.path.exists(df_row.mat_path):
                self.logger.warning('{} does not exist!'.format(df_row.mat_path))
                continue
            self.logger.info('loading bright '+df_row.mat_path)
            self.logger.info('wavelength = {} nm'.format(df_row.LASER_WL))
            bright_data = loadmat(df_row.mat_path,squeeze_me=True)['data']
            # remove dark
            bright_data -= dark
            bright = ISSF_Exposure(wavelength=df_row.LASER_WL, data=bright_data)
            bright.flip_columns()
            self.append(bright)
    def read_MethaneAIR(self,data_dir,which_band,
                        integration_time=None,
                        BPM=None,BPM_path=None,
                        if_rad_cal=False,
                        rad_cal_path=None,
                        rad_cal_coef=None,
                        if_remove_straylight=False,
                        K_far=None,
                        straylight_kernel_path=None,
                        straylight_n_iter=3):
        if self.instrum != 'MethaneAIR':
            self.logger.error('this works only for MethaneAIR!')
        central_wavelength = self.central_wavelength
        delta_wavelength_range = self.delta_wavelength_range
        nrow = self.nrow
        ncol = self.ncol
        if BPM is None:
            if BPM_path is None:
                self.logger.warning('no BPM provided, assuming no bad pixels')
                BPM = np.zeros((nrow,ncol),dtype=np.bool)
            else:
                self.logger.info('loading BPM from '+BPM_path)
                BPM = np.genfromtxt(BPM_path,delimiter=',',dtype=np.bool)
        if if_rad_cal and rad_cal_coef is None:
            self.logger.info('load rad cal coefficient from '+rad_cal_path)
            rad_cal_coef = loadmat(rad_cal_path)['coef']
        if if_remove_straylight and K_far is None:
            self.logger.info('load straylight from '+straylight_kernel_path)
            K_far = F_stray_light_input(straylight_kernel_path)
        path = os.path.join(data_dir,'{:.0f}'.format(central_wavelength))
        self.path = path
        csv_fn = glob.glob(os.path.join(path,'*.csv'))
        if len(csv_fn) != 1:
            self.logger.error('only one csv info file should be in '+self.path)
            sys.exit()
        df = pd.read_csv(csv_fn[0],skiprows=4,skipinitialspace=True)
        df_dark = df.loc[df['Dark']==1]
        dark = np.zeros((nrow,ncol))
        for i in range(df_dark.shape[0]):
            if i > 0:continue#only the first dark works, don't know why
            dark_fn = os.path.join(path,df_dark.iloc[i].SeqName.split('.')[0]+'.mat')
            if not os.path.exists(dark_fn):
                continue
            self.logger.info('loading '+dark_fn)
            data = loadmat(dark_fn,squeeze_me=True)['data']
            dark += data
        dark[BPM] = np.nan
        df_bright = df.loc[df['Dark']==0]
        for i in range(df_bright.shape[0]):
            bright_fn = os.path.join(path,df_bright.iloc[i].SeqName.split('.')[0]+'.mat')
            delta_wavelength = df_bright.iloc[i].Wavelength-central_wavelength
            # self.logger.info('delta wavelength = {}'.format(delta_wavelength))
            if not os.path.exists(bright_fn) or delta_wavelength < delta_wavelength_range[0]\
                or delta_wavelength > delta_wavelength_range[1]:
                continue
            self.logger.info('loading '+bright_fn)
            self.logger.info('wavelength = {}'.format(df_bright.iloc[i].Wavelength))
            d = loadmat(bright_fn,squeeze_me=True,struct_as_record=False)
            if integration_time is None:
                integration_time = np.nanmax(d['meta'].intTime)
            data = d['data']
            data[BPM] = np.nan
            data -= dark
            if if_rad_cal:
                self.logger.info('apply radiometric calibration')
                # DN per s
                data = data/integration_time
                new_data = np.zeros_like(data)
                for ipoly in range(rad_cal_coef.shape[-1]):
                    new_data = new_data+rad_cal_coef[...,ipoly].squeeze()*np.power(data,ipoly+1)
                data = new_data
                self.logger.info('rad cal done')
            bright = ISSF_Exposure(wavelength=df_bright.iloc[i].Wavelength,
                                   data=data)
            if if_remove_straylight:
                self.logger.info('remove straylight')
                bright.remove_straylight(K_far,n_iter=straylight_n_iter)
                self.logger.info('straylight done')
            bright.flip_columns()
            self.append(bright)
    
    def reduce(self,use_rows_1based=None,pix_ext=10):
        '''
        trim 3-D data (wavelength steps, rows, columns) by columns. the trimmed
        array roughly extends +/-pix_ext from the ISSF peak.
        also specify rows to include by use_rows_1based
        '''
        peak_cols = np.array([np.argmax(np.nanmean(a['data'],axis=0)) for a in self])+1
        start_col = np.max([1,peak_cols.min()-pix_ext])
        end_col = np.min([self.ncol,peak_cols.max()+pix_ext])
        use_cols_1based = np.arange(start_col,end_col+1,dtype=np.int)
        issf_reduced_data = np.array([a['data']
                                      [np.ix_(np.searchsorted(a['rows_1based'],use_rows_1based),\
                                              np.searchsorted(a['cols_1based'],use_cols_1based))] for a in self])
        wavelengths = np.array([a['wavelength'] for a in self])
        self.use_cols_1based = use_cols_1based
        self.issf_reduced_data = issf_reduced_data
        self.wavelengths = wavelengths
    
    def ISSF_to_ISRF(self,use_rows_1based=None,
                     pix_ext=10,
                     ncores=None,
                     dw_grid=None,
                     tol=None,
                     max_iter=20,
                     savgol_window_length=None,
                     savgol_polyorder=None,
                     fit_pix_ext=8):
        if self.instrum == 'MethaneAIR':
            if tol is None:tol = 0.002
            if savgol_window_length is None: savgol_window_length = 81
            if savgol_polyorder is None: savgol_polyorder = 3
        if self.instrum == 'MethaneSAT':
            if tol is None:tol = 0.002
            if savgol_window_length is None: savgol_window_length = 251
            if savgol_polyorder is None: savgol_polyorder = 5
        if use_rows_1based is None:
            self.logger.info('use all rows by default')
            use_rows_1based = self[0]['rows_1based']
        if dw_grid is None:
            dw_grid = arange_(-1.5,1.5,0.0005)
        peak_cols = np.array([np.argmax(np.nanmean(a['data'],axis=0)) for a in self])+1
        start_col = np.max([1,peak_cols.min()-pix_ext])
        end_col = np.min([self.ncol,peak_cols.max()+pix_ext])
        use_cols_1based = np.arange(start_col,end_col+1,dtype=np.int)
        issf_reduced_data = np.array([a['data']
                                      [np.ix_(np.searchsorted(a['rows_1based'],use_rows_1based),\
                                              np.searchsorted(a['cols_1based'],use_cols_1based))] for a in self])
        wavelengths = np.array([a['wavelength'] for a in self])
        self.use_cols_1based = use_cols_1based
        self.issf_reduced_data = issf_reduced_data
        self.wavelengths = wavelengths
        central_wavelength = self.central_wavelength
        # tol=0.002
        # max_iter=20
        # savgol_window_length=251
        # savgol_polyorder=5
        # fit_pix_ext=8
        if ncores == 0:
            self.logger.info('use serial')
            ISRF_all_rows = np.array(
                [F_ISRF_per_row_wrapper((row,issf_reduced_data[:,i,:],use_cols_1based,
                         wavelengths,central_wavelength,dw_grid,
                         tol,max_iter,savgol_window_length,
                         savgol_polyorder,fit_pix_ext)) for (i,row) in enumerate(use_rows_1based)])
            return ISRF_all_rows
        import multiprocessing
        ncores_max = multiprocessing.cpu_count()
        if ncores is None:          
            ncores = int( np.ceil(ncores_max/2) )
            self.logger.info('no cpu number specified, use half, {}'.format(ncores))
        elif ncores > ncores_max:
            self.logger.warning('You asked for more cores than you have! Use max number %d'%ncores_max)
            ncores = ncores_max
        with multiprocessing.Pool(ncores) as pp:
            ISRF_all_rows = np.array(
                pp.map(F_ISRF_per_row_wrapper,
                       ((row,issf_reduced_data[:,i,:],use_cols_1based,
                         wavelengths,central_wavelength,dw_grid,
                         tol,max_iter,savgol_window_length,
                         savgol_polyorder,fit_pix_ext) for (i,row) in enumerate(use_rows_1based))
                ))
        return ISRF_all_rows
        
class Single_ISRF(OrderedDict):
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def merge(self,isrf,min_scale=8000):
        '''
        merge with another isrf at the same row/wavelength. 
        needed for methanesat where multiple fields fill the the slit
        isrf:
            another Single_ISRF object
        min_scale:
            if one isrf has scale lower than it, the other will be used without weighting
        '''
        if 'row' not in self.keys():
            return isrf
        if 'row' not in isrf.keys():
            return self
        if self['row'] != isrf['row']:
            self.logger.error('row inconsistent!')
            return
        if not all(np.isclose(self['dw_grid'],isrf['dw_grid'])):
            self.logger.warning('dw grid inconsistent, use the first one, good luck')
        self.tighten(remove_iterations=False)
        isrf.tighten(remove_iterations=False)
        new_isrf = Single_ISRF()
        new_isrf['row'] = self['row']
        new_isrf['dw_grid'] = self['dw_grid']
        w1 = np.nanmean(self['scales_{}'.format(self['niter'])])
        w2 = np.nanmean(isrf['scales_{}'.format(isrf['niter'])])
        if w1 < min_scale and w2 >= min_scale:
            self.logger.debug('second isrf receives full weight')
            w1 = 0.;w2 = 1.
        elif w1 >= min_scale and w2 < min_scale:
            self.logger.debug('first isrf receives full weight')
            w1 = 1.;w2 = 0.
        elif w1 < min_scale and w2 < min_scale:
            self.logger.warning('both isrfs are lower than min_scale of {}!!!'.format(min_scale))
            
        for f in ['ISRF','center_pix_final','scales_final','pp_final','pp_inv_final','centers_of_mass_final']:
            new_isrf[f] = (w1*self[f]+w2*isrf[f])/(w1+w2)
        
        if np.ptp(isrf['ISSFx']) < np.ptp(self['ISSFx']):
            issfx_narrower = isrf['ISSFx']
            issfx_wider = self['ISSFx']
            issfys_narrower = isrf['ISSFys']
            issfys_wider = self['ISSFys']
            weight_narrower = w2
            weight_wider = w1
        else:
            issfx_narrower = self['ISSFx']
            issfx_wider = isrf['ISSFx']
            issfys_narrower = self['ISSFys']
            issfys_wider = isrf['ISSFys']
            weight_narrower = w1
            weight_wider = w2
        # use the shorter issfx range, interpolate the wider to short and weight-average
        new_isrf['ISSFx'] = issfx_narrower
        interp_func = interp1d(issfx_wider,issfys_wider,bounds_error=False,fill_value=np.nan)
        issfys_wider_interp = interp_func(issfx_narrower)
        new_isrf['ISSFys'] = np.nansum(np.column_stack((issfys_narrower*weight_narrower,issfys_wider_interp*weight_wider)),axis=1)/(weight_narrower+weight_wider)
        xmask = (new_isrf['ISSFx'] > np.nanmax([np.nanmin(issfx_narrower),np.nanmin(issfx_narrower)]))&(new_isrf['ISSFx'] <np.nanmin([np.nanmax(issfx_narrower),np.nanmax(issfx_narrower)]))
        new_isrf['ISSFx'] = new_isrf['ISSFx'][xmask]
        new_isrf['ISSFys'] = new_isrf['ISSFys'][xmask]
        new_isrf['niter'] = np.max([self['niter'],isrf['niter']])
        # make sure the new isrf integrates to 1
        new_isrf['ISRF'] = new_isrf['ISRF']/np.trapz(new_isrf['ISRF'],new_isrf['dw_grid'])
        return new_isrf
    
    def smooth_ISRF_by_iterative_savgol(self,window_length=81, polyorder=3, 
                     logResidualThreshold=[0.5,0.25,0.1,0.05,0.01], nIteration=None, tail=0.28):
        '''
        from iterative_savgol function in combine_MethaneAIR_specCal.py
        '''
        if nIteration == None:
            nIteration = len(logResidualThreshold)
        y_isrf_filtered = self['ISRF'].copy()
        y_isrf = self['ISRF'].copy()
        self['ISRF_before_iterative_savgol'] = self['ISRF'].copy()
        isrf_dw = self['dw_grid']
        isrf_savgol=savgol_filter(y_isrf,window_length=window_length, polyorder=polyorder)
        for i in range(nIteration):
            log_resids = np.abs(np.log10(isrf_savgol)-np.log10(y_isrf))
            if i < len(logResidualThreshold):
                threshold = logResidualThreshold[i]
            else:
                threshold = logResidualThreshold[-1]
            loc = np.where((log_resids > threshold) & (np.abs(isrf_dw)>=tail))
            y_isrf_filtered[loc] = isrf_savgol[loc]
            isrf_savgol=savgol_filter(y_isrf_filtered,window_length=window_length, polyorder=polyorder) 
        self['ISRF'] = isrf_savgol
    
    def tighten(self,remove_iterations=True):
        '''
        get rid of diagnostic fields, leaving only the final ISRF
        get only the final values for center_pix, pp, pp_inv, scales, and centers_of_mass
        '''
        if 'isrf_{}'.format(self['niter']) in self.keys():
            self['ISRF'] = np.float32(self['isrf_{}'.format(self['niter'])].copy())
            self['ISSFy'] = np.float32(self['issfy_{}'.format(self['niter'])].copy())
            self['ISSFys'] = np.float32(self['issfys_{}'.format(self['niter'])].copy())
            self['ISSFx'] = np.float32(self['issfx_{}'.format(self['niter'])].copy())
            if remove_iterations:
                for i in range(self['niter']+1):
                    self.pop('issfx_{}'.format(i))
                    self.pop('issfy_{}'.format(i))
                    self.pop('issfys_{}'.format(i))
                    self.pop('isrf_{}'.format(i))
        if 'center_pix_{}'.format(self['niter']) in self.keys():
            for f in ['center_pix', 'pp', 'pp_inv', 'scales', 'centers_of_mass']:
                self[f+'_final'] = self['{}_{}'.format(f,self['niter'])]
                if remove_iterations:
                    for i in range(self['niter']+1):
                        self.pop('{}_{}'.format(f,i))
    
    def save_mat(self,fn):
        '''
        save Single_ISRF dictionary to .mat file
        '''
        savemat(fn,{k:v for (k,v) in self.items()})
    
    def read_mat(self,fn,fields=None):
        '''
        read the mat file saved by Single_ISRF.save_mat
        '''
        d = loadmat(fn,squeeze_me=True)
        for (k,v) in d.items():
            if k not in ['__header__', '__version__', '__globals__']:
                if fields != None:
                    if k in fields:
                        self.__setitem__(k, v)
                else:
                    self.__setitem__(k, v)
        return self
    
    def center_pix(self):
        if 'niter' not in self.keys():
            return np.nan
        if 'center_pix_final' in self.keys():
            return self['center_pix_final']
        else:
            return self['center_pix_{}'.format(self['niter'])]
    
    def tuning_rate(self):
        if 'niter' not in self.keys():
            return np.nan
        if 'pp_inv_final' in self.keys():
            return self['pp_inv_final']
        else:
            return self['pp_inv_{}'.format(self['niter'])][0]
    
    def peak_width(self,percent=0.5,field=None):
        if field is None:
            if 'ISRF_restretched' in self.keys():
                field = 'ISRF_restretched'
            else:
                field = 'ISRF'
        if field not in self.keys():
            return np.nan
        
        isrfy = self[field]
        isrfx = self['dw_grid']
        peaks = np.array([np.argmax(isrfy)])
        result = peak_widths(isrfy,peaks,rel_height=percent)
        dx = np.abs(np.nanmean(np.diff(isrfx)))
        return (result[0]*dx).squeeze()
    
    def plot_COM(self,iterations=None,existing_ax=None,kwargs={}):
        if existing_ax is None:
            self.logger.info('axes not supplied, creating one')
            fig,ax = plt.subplots(1,1,figsize=(10,5))
        else:
            fig = None
            ax = existing_ax
        figout = {}
        figout['fig'] = fig
        figout['ax'] = ax
        if iterations is None:
            iterations = range(self['niter']+1)
        leg = []
        for i in iterations:
            if i > self['niter']:
                self.logger.warning('iteration {} does not exist!'.format(i))
                continue
            ax.plot(self['wavelengths'],self['centers_of_mass_{}'.format(i)],**kwargs)
            leg.append(i)
        ax.set_xlabel('Wavelength [nm]');
        ax.set_ylabel('Center of mass [spectral pixel]');
        figout['leg'] = ax.legend(['Iteration {}'.format(l) for l in leg])
        return figout
    
    def plot_scale(self,iterations=None,existing_ax=None,kwargs={}):
        if existing_ax is None:
            self.logger.info('axes not supplied, creating one')
            fig,ax = plt.subplots(1,1,figsize=(10,5))
        else:
            fig = None
            ax = existing_ax
        figout = {}
        figout['fig'] = fig
        figout['ax'] = ax
        if iterations is None:
            iterations = range(self['niter']+1)
        leg = []
        for i in iterations:
            if i > self['niter']:
                self.logger.warning('iteration {} does not exist!'.format(i))
                continue
            ax.plot(self['wavelengths'],self['scales_{}'.format(i)],**kwargs)
            leg.append(i)
        ax.set_xlabel('Wavelength [nm]');
        ax.set_ylabel('Scale');
        figout['leg'] = ax.legend(['Iteration {}'.format(l) for l in leg])
        return figout
    
    def plot_ISSF(self,iteration=None,existing_ax=None):
        if existing_ax is None:
            self.logger.info('axes not supplied, creating one')
            fig,ax = plt.subplots(1,1,figsize=(10,5))
        else:
            fig = None
            ax = existing_ax
        if iteration is None:
            iteration = self['niter']
        figout = {}
        figout['fig'] = fig
        figout['ax'] = ax
        if 'issfx_{}'.format(iteration) in self.keys():
            figout['issf'] = ax.plot(self['issfx_{}'.format(iteration)],
                                     self['issfy_{}'.format(iteration)],'.')
            figout['issfs'] = ax.plot(self['issfx_{}'.format(iteration)],
                                      self['issfys_{}'.format(iteration)],'-')
        else:
            figout['issf'] = ax.plot(self['ISSFx'],
                                     self['ISSFy'],'.')
            figout['issfs'] = ax.plot(self['ISSFx'],
                                      self['ISSFys'],'-')
        ax.set_xlabel('Spectral pixel');
        ax.set_ylabel('Normalized signal');
        return figout
    
    def plot_ISRF(self,iteration=None,existing_ax=None,kwargs={}):
        if existing_ax is None:
            self.logger.info('axes not supplied, creating one')
            fig,ax = plt.subplots(1,1,figsize=(10,5))
        else:
            fig = None
            ax = existing_ax
        if iteration is None:
            iteration = self['niter']
        figout = {}
        figout['fig'] = fig
        figout['ax'] = ax
        if 'isrf_{}'.format(iteration) in self.keys():
            figout['isrf'] = ax.plot(self['dw_grid'],
                                     self['isrf_{}'.format(iteration)],**kwargs)
        else:
            figout['isrf'] = ax.plot(self['dw_grid'],
                                     self['ISRF'],**kwargs)
        ax.set_xlabel('Wavelength');
        ax.set_ylabel('Normalized signal');
        return figout
    
    def plot_diagnostic(self):
        fig = plt.figure(figsize=(12,6),constrained_layout=True)
        gs = fig.add_gridspec(3,2)
        ax1 = fig.add_subplot(gs[:,0])
        self.plot_ISSF(existing_ax=ax1);
        ax1.text(0.025,0.8,'w = {:.2f} nm;\ncol = {:.5f};\nrow = {};\niteration = {} (max {});\nslope={:.5f} nm/pix'.format(
            self['central_wavelength'],self['center_pix_{}'.format(self['niter'])],self['row'],self['niter'],self['max_iter'],self['pp_inv_{}'.format(self['niter'])][0]),
            transform=ax1.transAxes)
        ax1.set_xlim((-5,5))
        ax1.set_ylim((0,0.4))
        ax1.legend(['ISSF in final iteration','Smoothed ISSF in final iteration'],loc='upper right')
        
        ax2 = fig.add_subplot(gs[0,1])
        self.plot_ISSF(existing_ax=ax2);
        ax2.set_yscale('log')
        
        ax3 = fig.add_subplot(gs[1,1])
        plotx = self['ISSFx']
        ploty = self['ISSFy']-self['ISSFys']
        ax3.plot(plotx,ploty,'.')
        ax3.legend(['Residual RMS={:.5f}'.format(np.std(ploty[np.abs(plotx)<3]))])
        ax3.set_xlim((-5,5))
        ax3.set_xlabel('Spectral pixel')
        ax3.set_ylabel('Residual')
        
        ax4 = fig.add_subplot(gs[2,1])
        ax4.plot(self['dw_grid'],self['ISRF'])
        if 'ISRF_before_iterative_savgol' in self.keys():
            ax4.plot(self['dw_grid'],self['ISRF_before_iterative_savgol'])
            ax4.legend(['After smoothing','Before smoothing'])
        ax4.set_xlabel('Wavelength [nm]')
        ax4.set_ylabel(r'Normalized ISRF [nm$^{-1}$]')
        ax4.set_xlim((-0.3,0.3))
        fwhm = F_peak_width(self['dw_grid'],self['ISRF'],percent=0.5)
        ax4.text(0.025,0.7,'FWHM = {:.5f} nm;\nsampling = {:.2f} pix'.format(fwhm,fwhm/self['pp_inv_{}'.format(self['niter'])][0]),transform=ax4.transAxes)
        figout = {}
        figout['fig'] = fig
        figout['ax1'] = ax1
        figout['ax2'] = ax2
        figout['ax3'] = ax3
        figout['ax4'] = ax4
        return figout

class Multiple_ISRFs():
    """an class representing multiple ISRF in a row-central wavelength framework
    
    Items:
        shape: (number of rows, number of central wavelengths), should equal to self.data.shape
        rows_1based: an integer array of rows
        row_mask: keep as None
        central_wavelengths: an array of central wavelengths
        data: 2D numpy object array with shape len(rows_1based) by len(central_wavelengths). each element is a Single_ISRF object
        dw_grid: wavelength grid common for all elements in data
        
    Methods:
        __init__: initializes instance and key attributes.
        load_Single_ISRF: assemble Multiple_ISRFs from a list of Single_ISRF or a list of file names
        merge_by_row: merge two Multiple_ISRFs objects with same wavelengths but different rows into one
        merge_by_wavelength: merge by wavelength
        
    """
    def __init__(self,central_wavelengths=None,rows_1based=None,nc_fn=None,
                 dtype=Single_ISRF,instrum=None,which_band=None,row_mask=None):
        self.logger = logging.getLogger(__name__)
        if central_wavelengths is None or rows_1based is None:
            if nc_fn is None:
                self.logger.error('you have to provide nc_fn in this case')
            self.logger.info('reading data from {}'.format(nc_fn))
            self = self.read_nc(nc_fn)
        else:
            self.central_wavelengths = np.array(central_wavelengths)
            self.rows_1based = np.array(rows_1based)
            self.shape = (len(self.rows_1based),len(self.central_wavelengths))
            self.logger.info('creating an instance of Multiple_ISRFs with {} row(s) and {} wavelength(s)'.format(self.shape[0],self.shape[1]))
            self.data = np.full(self.shape,Single_ISRF(),dtype=Single_ISRF)
            self.instrum = instrum
            self.which_band = which_band
        if row_mask is None:
            self.row_mask = np.zeros(self.rows_1based.shape,np.bool)
        else:
            self.row_mask = row_mask
    
    def load_Single_ISRF(self,isrf_list=None,isrf_flist=None):
        if isrf_list is None:
            self.logger.info('isrf object list not given, read from isrf mat file list')
            if isrf_flist is None:
                self.logger.error('isrf_list and isrf_flist cannot be both None!')
            isrf_list = np.array([Single_ISRF().read_mat(fn) for fn in isrf_flist])
        # assume dw_grid of all Single_ISRF object in isrf_list is the same
        self.dw_grid = isrf_list[0]['dw_grid']
        isrf_rows = np.array([isrf['row'] for isrf in isrf_list])
        isrf_cws = np.array([isrf['central_wavelength'] for isrf in isrf_list])
        if self.instrum is None:
            if np.max(isrf_rows) < 1500:
                self.logger.info('This appears to be MethaneAIR')
                self.instrum = 'MethaneAIR'
            if np.max(isrf_rows) > 1500:
                self.logger.info('This appears to be MethaneSAT')
                self.instrum = 'MethaneSAT'
        if self.which_band is None:
            if np.max(isrf_cws) < 1500:
                self.logger.info('This appears to be O2 band')
                self.which_band = 'O2'
            if np.max(isrf_cws) > 1500:
                self.logger.info('This appears to be CH4 band')
                self.which_band = 'CH4'
        for (ir,row) in enumerate(self.rows_1based):
            if self.row_mask[ir]:
                self.logger.info('row {} is masked and won''t be dealt with'.format(row))
                continue
            for (ic,cw) in enumerate(self.central_wavelengths):
                idx = np.where((isrf_rows == row) & (isrf_cws == cw))[0]
                if len(idx) != 1:
                    self.logger.warning('found {} single ISRF at row {}, central wavelength {}'.format(len(idx),row,cw))
                if len(idx) == 1:
                    self.data[ir][ic] = isrf_list[idx[0]]
    
    def merge_by_wavelength(self,misrf):
        if self.shape[0]*self.shape[1] == 0:
            return misrf
        if misrf.shape[0]*misrf.shape[1] == 0:
            return self
        if not all(np.isclose(self.rows_1based,misrf.rows_1based)):
            self.logger.error('wavelength dimension has to be consistent!')
            return
        union_wvls = np.union1d(self.central_wavelengths,misrf.central_wavelengths)
        new_misrf = Multiple_ISRFs(central_wavelengths=union_wvls,
                                   rows_1based=self.rows_1based,
                                   instrum=self.instrum,which_band=self.which_band)
        
        if hasattr(self,'dw_grid'):
            new_misrf.dw_grid = self.dw_grid
        elif hasattr(misrf,'dw_grid'):
            new_misrf.dw_grid = misrf.dw_grid
        else:
            self.logger.error('both misrf objects seem to be empty!')
            return
        # rows only in the first object, not in the second
        mask_12 = np.isin(new_misrf.central_wavelengths,np.setdiff1d(self.central_wavelengths,misrf.central_wavelengths))
        mask_1 = np.isin(self.central_wavelengths,np.setdiff1d(self.central_wavelengths,misrf.central_wavelengths))
        new_misrf.data[:,mask_12] = self.data[:,mask_1]
        # rows only in the second object, not in the first
        mask_21 = np.isin(new_misrf.central_wavelengths,np.setdiff1d(misrf.central_wavelengths,self.central_wavelengths))
        mask_2 = np.isin(misrf.central_wavelengths,np.setdiff1d(misrf.central_wavelengths,self.central_wavelengths))
        new_misrf.data[:,mask_21] = misrf.data[:,mask_2]
        # rows in both
        intersect_wvls = np.intersect1d(self.central_wavelengths,misrf.central_wavelengths)
        mask_12_inter = np.isin(new_misrf.central_wavelengths,intersect_wvls)
        mask_1_inter = np.isin(self.central_wavelengths,intersect_wvls)
        mask_2_inter = np.isin(misrf.central_wavelengths,intersect_wvls)
        self.logger.info('merged misrf has {} wvls, {} from 1, {} from 2, {} merged from both'.format(len(new_misrf.central_wavelengths),np.sum(mask_1),np.sum(mask_2),np.sum(mask_12_inter)))
        if np.sum(mask_12_inter) > 0:
            self.logging.error('this should not happen!')
            return
        return new_misrf    
        
    def merge_by_row(self,misrf,**kwargs):
        if self.shape[0]*self.shape[1] == 0:
            return misrf
        if misrf.shape[0]*misrf.shape[1] == 0:
            return self
        if self.shape[1] != misrf.shape[1]:
            self.logger.error('wavelength dimension has to be consistent!')
            return
        
        union_rows = np.union1d(self.rows_1based,misrf.rows_1based)
        new_misrf = Multiple_ISRFs(central_wavelengths=self.central_wavelengths,
                                   rows_1based=union_rows,
                                   instrum=self.instrum,which_band=self.which_band)
        if hasattr(self,'dw_grid'):
            new_misrf.dw_grid = self.dw_grid
        elif hasattr(misrf,'dw_grid'):
            new_misrf.dw_grid = misrf.dw_grid
        else:
            self.logger.error('both misrf objects seem to be empty!')
            return
        # rows only in the first object, not in the second
        mask_12 = np.isin(new_misrf.rows_1based,np.setdiff1d(self.rows_1based,misrf.rows_1based))
        mask_1 = np.isin(self.rows_1based,np.setdiff1d(self.rows_1based,misrf.rows_1based))
        new_misrf.data[mask_12,] = self.data[mask_1,]
        # rows only in the second object, not in the first
        mask_21 = np.isin(new_misrf.rows_1based,np.setdiff1d(misrf.rows_1based,self.rows_1based))
        mask_2 = np.isin(misrf.rows_1based,np.setdiff1d(misrf.rows_1based,self.rows_1based))
        new_misrf.data[mask_21,] = misrf.data[mask_2,]
        # rows in both
        intersect_rows = np.intersect1d(self.rows_1based,misrf.rows_1based)
        mask_12_inter = np.isin(new_misrf.rows_1based,intersect_rows)
        mask_1_inter = np.isin(self.rows_1based,intersect_rows)
        mask_2_inter = np.isin(misrf.rows_1based,intersect_rows)
        self.logger.info('merged misrf has {} rows, {} from 1, {} from 2, {} merged from both'.format(len(new_misrf.rows_1based),np.sum(mask_1),np.sum(mask_2),np.sum(mask_12_inter)))
        for iwvl in range(self.shape[1]):
            new_misrf.data[mask_12_inter,iwvl] = [isrf1.merge(isrf2,**kwargs)
                                            for (isrf1,isrf2) in zip(
                                            self.data[mask_1_inter,iwvl],
                                            misrf.data[mask_2_inter,iwvl])]
        return new_misrf    
        
    def peak_widths(self,percent=0.5,use_isrf_data=True):
        '''
        find peak width for each Single_ISRF
        '''
        if use_isrf_data and hasattr(self,'isrf_data'):
            return np.array([F_peak_width(self.dw_grid,isrfy,percent) for isrf1 in self.isrf_data for isrfy in isrf1]).reshape(self.shape)
        else:
            self.logger.info('peak widths are from raw isrf profiles')
            return np.array([isrf.peak_width(percent=percent) for row in self.data for isrf in row]).reshape(self.shape)
    
    def register_wavelength_msat(self,use_central_wavelengths=None,n_wavcal_poly=None):
        ''' 
        wavcal for msat
        '''
        center_pix = np.full(self.shape,np.nan)
        center_pix_smooth = np.full(self.shape,np.nan)
        for iw in range(self.data.shape[1]):
            for irow in range(self.data.shape[0]):
                if 'niter' not in self.data[irow,iw].keys():
                    self.logger.warning('ISRF at row {}, central wavelength {} seems to be empty'.format(self.rows_1based[irow],self.central_wavelengths[iw]))
                    continue
                elif 'center_pix_final' in self.data[irow,iw].keys():
                    center_pix[irow,iw] = self.data[irow,iw]['center_pix_final']
                else:
                    center_pix[irow,iw] = self.data[irow,iw]['center_pix_{}'.format(self.data[irow,iw]['niter'])]
        mwindow=51
        center_pix_smooth = median_filter(center_pix,size=(mwindow,1))
        cp1 = center_pix.copy()
        cp1[np.abs(center_pix-center_pix_smooth)>0.03] = center_pix_smooth[np.abs(center_pix-center_pix_smooth)>0.03]
        center_pix_smooth = median_filter(cp1,size=(mwindow,1))
        wavcal_poly = np.full((self.shape[0],n_wavcal_poly+1),np.nan,dtype=np.float)
        for irow in range(self.shape[0]):
            if self.row_mask[irow]:
                continue
            xdata = center_pix_smooth[irow,:]
            if np.sum(np.isnan(xdata)) > 2:
                self.logger.info('Footprint %d appears to be empty'%irow)
                continue
            ydata = self.central_wavelengths
            wavcal_poly[irow,:] = np.flip(np.polyfit(xdata[~np.isnan(xdata)],ydata[~np.isnan(xdata)],n_wavcal_poly)) 
        self.center_pix_smooth = center_pix_smooth
        self.center_pix = center_pix
        self.wavcal_poly = wavcal_poly
        self.n_wavcal_poly = n_wavcal_poly
    def register_wavelength(self,use_central_wavelengths=None,n_wavcal_poly=None):
        '''
        wavelength calibration
        '''
        if n_wavcal_poly is None:
            if self.instrum.lower() == 'methanesat':
                n_wavcal_poly =3
            elif self.instrum.lower() == 'methaneair':
                n_wavcal_poly =1
                
        if use_central_wavelengths is None:
            use_central_wavelengths = self.central_wavelengths
        if self.instrum.lower() == 'methanesat':
            self.register_wavelength_msat(use_central_wavelengths,n_wavcal_poly)
            return
        data = self.data[:,np.searchsorted(self.central_wavelengths,use_central_wavelengths)]
        center_pix = np.full(data.shape,np.nan)
        center_pix_median = np.full(data.shape,np.nan)
        for iw in range(data.shape[1]):
            for irow in range(data.shape[0]):
                if 'niter' not in data[irow,iw].keys():
                    self.logger.warning('ISRF at row {}, central wavelength {} seems to be empty'.format(self.rows_1based[irow],self.central_wavelengths[iw]))
                    continue
                elif 'center_pix_final' in data[irow,iw].keys():
                    center_pix[irow,iw] = data[irow,iw]['center_pix_final']
                    center_pix_median[irow,iw] = data[irow,iw]['center_pix_final']
                else:
                    center_pix[irow,iw] = data[irow,iw]['center_pix_{}'.format(data[irow,iw]['niter'])]
                    center_pix_median[irow,iw] = data[irow,iw]['center_pix_{}'.format(data[irow,iw]['niter'])]
            center_pix_median[:,iw] = center_pix[:,iw]-np.nanmedian(center_pix[:,iw]) 
        center_pix_median2 = np.nanmedian(center_pix_median,axis=1)
        center_pix_smooth = np.full(data.shape,np.nan)       
        for iw in range(data.shape[1]):
            ydata = center_pix[:,iw].copy()
            xdata = center_pix_median2.copy()
            mask = (~np.isnan(xdata)) & (~np.isnan(ydata))
            xdata = xdata[mask];ydata = ydata[mask]
            pp = np.polyfit(xdata,ydata,1)
            yfit = np.polyval(pp,xdata)
            residual = ydata-yfit
            rmse = np.sqrt(np.nansum(np.power(residual,2))/(len(xdata)-2))
            mask = np.abs(residual) <= 2.5*rmse
            xdata = xdata[mask];ydata = ydata[mask]
            pp = np.polyfit(xdata,ydata,1)
            yhat = np.polyval(pp,center_pix_median2)
            center_pix_smooth[:,iw] = yhat
        wavcal_poly = np.full((self.shape[0],n_wavcal_poly+1),np.nan,dtype=np.float)
        for irow in range(self.shape[0]):
            if self.row_mask[irow]:
                continue
            xdata = center_pix_smooth[irow,:]
            if np.sum(np.isnan(xdata)) > 2:
                self.logger.info('Footprint %d appears to be empty'%irow)
                continue
            ydata = self.central_wavelengths
            wavcal_poly[irow,:] = np.flip(np.polyfit(xdata,ydata,n_wavcal_poly)) 
        self.center_pix_smooth = center_pix_smooth
        self.center_pix_median = center_pix_median
        self.center_pix_median2 = center_pix_median2
        self.wavcal_poly = wavcal_poly
        self.n_wavcal_poly = n_wavcal_poly
    
    def plot_center_pix(self):
        ''' 
        plot center pix location at center wavelengths for all rows
        '''
        fig,axs = plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
        figout = {}
        figout['fig'] = fig
        figout['axs'] = axs
        ax = axs[0]
        ax.plot(self.rows_1based,self.center_pix)
        ax.legend(['{} nm'.format(c) for c in self.central_wavelengths])
        ax.set_xlabel('Spatial pixels')
        ax.set_ylabel('Spectral pixels')
        ax = axs[1]
        ax.plot(self.rows_1based,np.array([c-np.nanmedian(c) for c in self.center_pix.T]).T)
        ax.set_xlabel('Spatial pixels')
        ax.set_ylabel('Relative spectral pixels')
        return figout
    def plot_dispersion(self,plot_rows=[500,700],ax=None):
        '''
        plot nm per pix at given rows
        '''
        if ax is None:
            self.logger.info('axes not supplied, creating one')
            fig,ax = plt.subplots(1,1,figsize=(7,3.5),constrained_layout=True)
        else:
            fig = None
        figout = {}
        figout['fig'] = fig
        figout['ax'] = ax
        row_indices = np.nonzero(np.isin(self.rows_1based,plot_rows))[0]
        bright_clist = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE',
                    '#AA3377', '#BBBBBB', '#000000']
        ll = []
        for (i,row_index) in enumerate(row_indices):
            if self.row_mask[row_index]:
                continue
            if np.sum(np.isnan(self.wavcal_poly[row_index,:])) > 0:
                self.logger.info('row {} does not have a valid wavcal'.format(row_index))
                continue
            dispersion = []
            local_tr = []
            for (iw,central_wavelength) in enumerate(self.central_wavelengths):
                nm_per_pix = 0.
                for ipoly in range(1,self.n_wavcal_poly+1):# loop over 1, 2, 3, 4 if n_wavcal_poly=4
                    nm_per_pix = nm_per_pix+self.wavcal_poly[row_index,ipoly]*ipoly*\
                    np.power(self.data[row_index,iw].center_pix(),ipoly-1)
                dispersion.append(nm_per_pix)
                local_tr.append(self.data[row_index,iw]['pp_inv_final'][0])
            tmp=ax.plot(self.central_wavelengths,dispersion,
            self.central_wavelengths,local_tr,'o',color=bright_clist[i])
            ll.append(tmp[0])
        ax.legend(ll,['Row {}'.format(r) for r in plot_rows])
        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel('nm per pix')
        return figout
    def plot_wavcal(self,plot_rows=[500,700],ax1=None,ax2=None):
        '''
        plot wavelength calibration curve
        '''
        if ax1 is None:
            self.logger.info('axes not supplied, creating one')
            fig,axs = plt.subplots(2,1,figsize=(7,5),sharex=True,constrained_layout=True)
            ax1 = axs[0];ax2 = axs[1]
        else:
            fig = None
        figout = {}
        figout['fig'] = fig
        figout['ax1'] = ax1
        figout['ax2'] = ax2
        row_indices = np.nonzero(np.isin(self.rows_1based,plot_rows))[0]
        bright_clist = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE',
                    '#AA3377', '#BBBBBB', '#000000']
        for (i,row_index) in enumerate(row_indices):
            xdata = self.center_pix_smooth[row_index,]
            ydata = self.central_wavelengths
            yhat = np.polyval(np.flip(self.wavcal_poly[row_index,:]),xdata)
            ax1.plot(xdata,ydata,marker='o',linestyle='none',color=bright_clist[i])
            ax1.plot(xdata,yhat,color=bright_clist[i])
            ax2.plot(xdata,ydata-yhat,marker='o',linestyle='--',color=bright_clist[i])
        ax2.legend(['Row {}'.format(r) for r in plot_rows])
        ax1.set_ylabel('Wavelength [nm]')
        ax2.set_ylabel('Relative wavelength [nm]')
        ax2.set_xlabel('Spectral pixel')
        ax1.grid();ax2.grid()
        return figout
    def restretch_ISRF(self):
        ''' 
        update the horizontal axis of ISRF using full-column wavcal
        '''
        for (irow,row) in enumerate(self.rows_1based):
            if self.row_mask[irow]:
                continue
            if np.sum(np.isnan(self.wavcal_poly[irow,:])) > 0:
                self.logger.info('row {} does not have a valid wavcal'.format(row))
                continue
            for (iw,central_wavelength) in enumerate(self.central_wavelengths):
                nm_per_pix = 0.
                for ipoly in range(1,self.n_wavcal_poly+1):# loop over 1, 2, 3, 4 if n_wavcal_poly=4
                    nm_per_pix = nm_per_pix+self.wavcal_poly[irow,ipoly]*ipoly*\
                    np.power(self.data[irow,iw].center_pix(),ipoly-1)
                self.data[irow,iw]['nm_per_pix'] = nm_per_pix
                if 'ISSFx' not in self.data[irow,iw].keys():
                    self.logger.warning('row {}, wavelength {} appears empty'.format(row,central_wavelength))
                    continue
                interp_func = interp1d(-self.data[irow,iw]['ISSFx']*nm_per_pix,self.data[irow,iw]['ISSFys'],
                bounds_error=False,fill_value=0)
                dw_grid = self.data[irow,iw]['dw_grid']
                isrfy = interp_func(dw_grid)
                isrfy[np.isnan(isrfy) | (isrfy < 0)] = 0
                self.data[irow,iw]['ISRF_restretched'] = isrfy/np.trapz(isrfy,dw_grid)
    
    def apply_median_filter(self,median_filter_size=(5,3,1),outlier_threshold=3):
        '''
        apply median filter to remove outlier isrf
        '''
        dummy_isrf = self.dw_grid*np.nan
        isrf_data = np.array([d['ISRF_restretched'] if 'ISRF_restretched' in d.keys() else dummy_isrf \
        for d in self.data.ravel()]).reshape(self.shape+(-1,))
        isrf_data_smooth = median_filter(isrf_data,size=median_filter_size)
        rms = np.sqrt(np.sum(np.power(isrf_data_smooth-isrf_data,2),axis=2)/(np.count_nonzero(~np.isnan(isrf_data), axis=2)-1))
        median_rms = np.nanmedian(rms,axis=1)
        outlier_criterion = rms-median_rms[:,np.newaxis]
        outlier_mask = np.abs(outlier_criterion)>outlier_threshold*np.nanstd(outlier_criterion)
        self.logger.info('{} outlier ISRFs will be replaced by median filtered values'.format(np.sum(outlier_mask)))
        isrf_data[outlier_mask,] = isrf_data_smooth[outlier_mask,]
        self.isrf_data = isrf_data
        self.outlier_criterion = outlier_criterion
    
    def fill_gaps(self,nrows_to_average=30):
        '''
        Gap filling of isrf and wavcal coefficients using the closest 
        nrows_to_average valid (non-nan) rows at the same central wavelength
        '''
        filled_mask = np.isnan(np.sum(self.isrf_data,axis=2))
        for irow in range(filled_mask.shape[0]):
            for icol in range(filled_mask.shape[1]):
                if not filled_mask[irow,icol]:
                    continue
                available_rows = self.rows_1based[~filled_mask[:,icol]]
                rows_to_average_0based = (available_rows[np.argsort(np.abs(available_rows-self.rows_1based[irow]))[0:nrows_to_average]]-1).astype(int)
                self.isrf_data[irow,icol,:] = np.mean(self.isrf_data[rows_to_average_0based,icol,:],axis=0)
                self.isrf_data[irow,icol,:] = self.isrf_data[irow,icol,:]/np.trapz(self.isrf_data[irow,icol,:],self.dw_grid)
        
        ismissing_wavcal = np.isnan(self.wavcal_poly[:,0])
        for irow in range(len(ismissing_wavcal)):
            if not ismissing_wavcal[irow]:
                continue
            available_rows = self.rows_1based[~ismissing_wavcal]
            rows_to_average_0based = (available_rows[np.argsort(np.abs(available_rows-self.rows_1based[irow]))[0:nrows_to_average]]-1).astype(int)
            self.wavcal_poly[irow,:] = np.mean(self.wavcal_poly[rows_to_average_0based,:],axis=0)
        
        
        
    def read_nc(self,fn):
        '''
        read a nc file and populate the object
        '''
        nc = Dataset(fn,'r')
        self.dw_grid = nc['delta_wavelength'][:].filled(np.nan)
        self.central_wavelengths = nc['central_wavelength'][:].filled(np.nan)
        self.isrf_data = nc['isrf'][:].filled(np.nan)
        self.rows_1based = np.arange(1,self.isrf_data.shape[0]+1)
        self.shape = (self.isrf_data.shape[0],self.isrf_data.shape[1])
        self.wavcal_poly = nc['pix2nm_polynomial'][:].filled(np.nan)
        self.n_wavcal_poly = self.wavcal_poly.shape[1]
        self.instrum = nc.instrument
        self.which_band = nc.band
        if 'filled_mask' in nc.variables.keys():
            self.filled_mask = nc['filled_mask'][:]
        nc.close()
        return self
        
    def save_nc(self,fn,saving_time=None):
        '''
        save data to netcdf
        '''
        if not hasattr(self,'isrf_data'):
            self.apply_median_filter()
        if saving_time is None:
            saving_time = dt.datetime.now()
        nc = Dataset(fn,'w')
        ncattr_dict = {'description':'MethaneAIR/MethaneSAT ISRF data (https://doi.org/10.5194/amt-14-3737-2021)',
                       'institution':'University at Buffalo',
                       'contact':'Kang Sun, kangsun@buffalo.edu',
                       'history':'Created '+saving_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                       'instrument':self.instrum,
                       'band':self.which_band}
        nc.setncatts(ncattr_dict)
        nc.createDimension('delta_wavelength',len(self.dw_grid))
        nc.createDimension('central_wavelength',self.shape[1])
        nc.createDimension('ground_pixel',self.shape[0])
        nc.createDimension('polynomial',self.n_wavcal_poly+1)
        
        var_isrf_dw = nc.createVariable('delta_wavelength',np.float,('delta_wavelength',))
        var_isrf_dw.units = 'nm'
        var_isrf_dw.long_name = 'wavelength grid of ISRF'
        var_isrf_dw[:] = self.dw_grid
        
        var_isrf_w = nc.createVariable('central_wavelength',np.float,('central_wavelength',))
        var_isrf_w.units = 'nm'
        var_isrf_w.long_name = 'wavelength grid where ISRF was measured'
        var_isrf_w[:] = self.central_wavelengths
        
        var_isrf = nc.createVariable('isrf',np.float,('ground_pixel','central_wavelength','delta_wavelength'))
        var_isrf.units = 'nm^-1'
        var_isrf.long_name = 'ISRF'
        var_isrf[:] = self.isrf_data
        
        if hasattr(self,'filled_mask'):
            var_mask = nc.createVariable('filled_mask',np.int,('ground_pixel','central_wavelength'))
            var_mask.units = 'T/F'
            var_mask.long_name = '1-filled,0-not filled'
            var_mask[:] = self.filled_mask
        
        var_wavcal = nc.createVariable('pix2nm_polynomial',np.float,('ground_pixel','polynomial'))
        var_wavcal.long_name = 'wavelength calibration coefficients, starting from intercept'
        var_wavcal[:] = self.wavcal_poly
        
        nc.close()
        
def F_ISRF_per_row_wrapper(args):
    try:
        out = F_ISRF_per_row(*args)
    except Exception as e:
        logging.warning('error occurred at row {}:'.format(args[0]))
        logging.warning(e)
        out = {}
    return out

def F_ISRF_per_row(row,issf_row_data,
                   use_cols_1based,
                   wavelengths,
                   central_wavelength,
                   dw_grid,
                   tol=0.002,max_iter=20,
                   savgol_window_length=251,
                   savgol_polyorder=5,
                   fit_pix_ext=8,
                   tighten_output=True,
                   if_iter_savgol=False):
    def F_shift_scale(inp,shift,scale):
        xdata = inp['issfx']+shift
        ydata = inp['issfy']*scale
        f = interp1d(xdata, ydata,fill_value='extrapolate')
        return f(inp['xx'])
    nstep = issf_row_data.shape[0]
    ncol0 = issf_row_data.shape[1]
    issf_results = Single_ISRF()
    for iround in range(max_iter):
        centers_of_mass = np.full(nstep,np.nan)
        scales = np.full(nstep,np.nan)
        for istep in range(nstep):
            yy = np.float64(issf_row_data[istep,:])
            xx = np.float64(use_cols_1based.copy())
            mask = (~np.isnan(yy)) & (~np.isnan(xx))
            yy = yy[mask];xx = xx[mask]
            if iround == 0:
                centers_of_mass[istep] = np.trapz(xx*yy,xx)/np.trapz(yy,xx)
                scales[istep] = np.trapz(yy,xx)/np.nanmedian(np.abs(np.diff(xx)))
            else:
                mask = (xx > issf_results['center_pix_{}'.format(iround-1)]-(fit_pix_ext-1))\
                &(xx<issf_results['center_pix_{}'.format(iround-1)]+(fit_pix_ext-1))
                xx = xx[mask];yy = yy[mask]
                initial_guess = [issf_results['centers_of_mass_{}'.format(0)][istep],
                                 issf_results['scales_{}'.format(0)][istep]*1.1]
                inp = {}
                inp['xx'] = xx
                mask = (issf_results['issfx_{}'.format(iround-1)] > -fit_pix_ext)\
                &(issf_results['issfx_{}'.format(iround-1)]<+fit_pix_ext)
                inp['issfx'] = issf_results['issfx_{}'.format(iround-1)][mask]
                inp['issfy'] = issf_results['issfys_{}'.format(iround-1)][mask]
                popt,_ = curve_fit(F_shift_scale,inp,yy,p0=initial_guess)
                centers_of_mass[istep] = popt[0]
                scales[istep] = popt[1]
        mask = (~np.isnan(centers_of_mass))
        pp = np.polyfit(wavelengths[mask],centers_of_mass[mask],1)
        center_pix = np.polyval(pp,central_wavelength)
        pix_mat = np.full((nstep,ncol0),np.nan)
        for istep in range(nstep):
            pix_mat[istep,:] = use_cols_1based-(centers_of_mass[istep]-center_pix)
        xx = pix_mat.ravel()
        yy = np.array([issf_row_data[istep,:]/scale for (istep,scale) in enumerate(scales)]).ravel()
        xind = np.argsort(xx)
        xx = xx[xind];yy = yy[xind]
        mask = (~np.isnan(yy)) & (~np.isnan(xx)) & (yy > 0)
        xx = xx[mask];yy = yy[mask]
        interp_func = interp1d(xx,yy,fill_value='extrapolate')
        xx = np.linspace(np.min(xx),np.max(xx),2*len(xx))
        yy = interp_func(xx)
        if if_iter_savgol:
            yy_smooth = iterative_savgol(yy,xx,window_length=savgol_window_length,
                                  polyorder=savgol_polyorder)
        else:
            yy_smooth = savgol_filter(yy,window_length=savgol_window_length,
                                  polyorder=savgol_polyorder)
        issf_results['centers_of_mass_{}'.format(iround)] = centers_of_mass
        issf_results['scales_{}'.format(iround)] = scales
        issf_results['center_pix_{}'.format(iround)] = center_pix
        issf_results['issfx_{}'.format(iround)] = xx-center_pix
        issf_results['issfy_{}'.format(iround)] = yy
        issf_results['issfys_{}'.format(iround)] = yy_smooth
        issf_results['pp_{}'.format(iround)] = pp
        # pp maps wavelengths to spectral pixels; pp_inv maps spectral pixels to wavelengths
        pp_inv = np.array([1/pp[0],-pp[1]/pp[0]])
        issf_results['pp_inv_{}'.format(iround)] = pp_inv
        isrf_x = -np.polyval(pp_inv,xx)+central_wavelength
        interp_func = interp1d(isrf_x,yy_smooth,fill_value='extrapolate')
        # dw_grid = arange_(-1.5,1.5,0.0005)
        isrf_y= interp_func(dw_grid)
        isrf_y[(dw_grid<np.nanmin(isrf_x))|(dw_grid>np.nanmax(isrf_x))] = 0.
        issf_results['isrf_{}'.format(iround)] = isrf_y/np.trapz(isrf_y,dw_grid)
        if iround >0:
            max_savgol_diff = np.max(np.abs(issf_results['isrf_{}'.format(iround)][np.abs(dw_grid*pp[0])<3]
                                            -issf_results['isrf_{}'.format(iround-1)][np.abs(dw_grid*pp[0])<3]))
            # logging.info('row {}, smoothed ISRF differs from previous iter by {:.5f}'.format(row,max_savgol_diff))
            if max_savgol_diff < tol:
                logging.info('row {} converges at iter {}'.format(row,iround))
                break
    issf_results['niter'] = iround
    issf_results['max_iter'] = max_iter
    issf_results['wavelengths'] = wavelengths
    issf_results['central_wavelength'] = central_wavelength
    issf_results['dw_grid'] = dw_grid
    issf_results['row'] = row
    if tighten_output:
        issf_results.tighten()
    return issf_results

def iterative_savgol(y_isrf, x_isrf, window_length=81, polyorder=3, 
                     logResidualThreshold=[0.5,0.25,0.1,0.05,0.01], nIteration=None, tail=2.8):
    if nIteration == None:
        nIteration = len(logResidualThreshold)
    isrf_savgol=savgol_filter(y_isrf,window_length=window_length, polyorder=polyorder)
    y_isrf_filtered = y_isrf.copy()
    for i in range(nIteration):
        log_resids = np.abs(np.log10(isrf_savgol)-np.log10(y_isrf))
        if i < len(logResidualThreshold):
            threshold = logResidualThreshold[i]
        else:
            threshold = logResidualThreshold[-1]
        loc = np.where((log_resids > threshold) & (np.abs(x_isrf)>=tail))
        y_isrf_filtered[loc] = isrf_savgol[loc]
        isrf_savgol=savgol_filter(y_isrf_filtered,window_length=window_length, polyorder=polyorder)         
    return isrf_savgol