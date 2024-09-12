#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:42:34 2024

@author: kangsun
"""

from netCDF4 import Dataset
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
from scipy.interpolate import RegularGridInterpolator, interp1d
from astropy.convolution import convolve_fft
import sys,os,glob
from collections import OrderedDict
from scipy.constants import N_A, R

PLANCK_CONSTANT = 6.62607004e-34
BOLTZMANN_CONSTANT = 1.38064852e-23
LIGHT_SPEED = 2.99792458e8

GOSAT_AP = np.array([1.        , 0.92307692, 0.84615385, 0.76923077, 0.69230769,
       0.61538462, 0.53846154, 0.46153846, 0.38461538, 0.30769231,
       0.23076923, 0.15384615, 0.07692308, 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ])
GOSAT_BP = np.array([1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
       1. , 0.5, 0. , 0. , 0. , 0. , 0. ])
GOSAT_CP = np.array([ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. , 40. , 80. , 50. , 10. ,  1. ,  0.1])

GOSAT_P = np.array([
    1.01325000e+03, 9.50692308e+02, 8.88134615e+02, 8.25576923e+02,
    7.63019231e+02, 7.00461538e+02, 6.37903846e+02, 5.75346154e+02,
    5.12788462e+02, 4.50230769e+02, 3.87673077e+02, 3.25115385e+02,
    2.62557692e+02, 2.00000000e+02, 1.40000000e+02, 8.00000000e+01,
    5.00000000e+01, 1.00000000e+01, 1.00000000e+00, 1.00000000e-01
])

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

def BB(T_K,w_nm,radiance_unit='photons/s/cm2/sr/nm',
       c=LIGHT_SPEED,h=PLANCK_CONSTANT,kB=BOLTZMANN_CONSTANT,
       do_dBdT=False):
    '''
    planck function with different output units
    '''
    rus = radiance_unit.split(' ')
    if len(rus) == 1:
        factor = 1.
        unit = rus[0]
    elif len(rus) == 2:
        factor = float(rus[0])
        unit = rus[1]
    
    w = w_nm*1e-9# wavelength in m
    T = T_K
    a = 1./(np.exp(h*c/w/kB/T)-1)
    if do_dBdT:
        dadT = np.power(np.exp(h*c/w/kB/T)-1,-2)*np.exp(h*c/w/kB/T)*h*c/w/kB/T**2
    if unit=='W/m2/sr/nm':
        r = 2*h*np.power(c,2)*np.power(w,-5)*a*1e-9
        if do_dBdT:
            drdT = 2*h*np.power(c,2)*np.power(w,-5)*dadT*1e-9
    elif unit=='photons/s/cm2/sr/nm':
        r = 2*c*np.power(w,-4)*a*1e-13
        if do_dBdT:
            drdT = 2*c*np.power(w,-4)*dadT*1e-13
    elif unit=='photons/s/cm2/sr/cm-1':
        r = 2*c*np.power(w,-4)*a*1e-13*np.power(w_nm,2)*1e-7
        if do_dBdT:
            drdT = 2*c*np.power(w,-4)*dadT*1e-13*np.power(w_nm,2)*1e-7
    if do_dBdT:
        return r/factor,drdT/factor
    else:
        return r/factor

def F_interp_absco(absco_P,absco_T,absco_B,absco_w,absco_sigma,
                   Pq,Tq,Bq,wq,do_dsigmadT=False):
    ''' 
    absco_* should be the same format as in the absco table saved by splat
    P should be in Pa; T in K; B in volume/volume
    '''
    # absco_P has to be ascending
    nearest_P_index = np.argmin(np.abs(absco_P-Pq))
    nearest_P = absco_P[nearest_P_index]
    if nearest_P >= Pq:
        next_P_index = nearest_P_index - 1
    else:
        next_P_index = nearest_P_index + 1
    next_P = absco_P[next_P_index]
    P_mask = np.isin(absco_P,[nearest_P,next_P])
    aP = absco_P[P_mask]
    aT = absco_T[P_mask,]
    asigma = absco_sigma[P_mask,]
    T_mask = np.zeros(aT.shape,dtype=bool)
    for ilayer,iT in enumerate(aT):
        T_mask[ilayer,] = np.isin(iT,iT[np.argsort(np.abs(iT-Tq))[:2]])
    T_grid = aT[0,T_mask[0,]]
    sigma_grid = np.zeros((len(aP),len(T_grid),*asigma.shape[2:]))
    for ilayer in range(len(aP)):
        sigma_grid[ilayer,...] = asigma[ilayer,T_mask[ilayer,],...]
    func = RegularGridInterpolator((aP,T_grid,absco_B,absco_w), sigma_grid)
    sigma = func((Pq,Tq,Bq,wq))
    if not do_dsigmadT:
        return sigma
    else:
        dsigmadT_grid = (sigma_grid[:,1,:,:]-sigma_grid[:,0,:,:])\
            /(T_grid[1]-T_grid[0])
        func_d = RegularGridInterpolator((aP,absco_B,absco_w), dsigmadT_grid)    
        dsigmadT = func_d((Pq,Bq,wq))
        return sigma, dsigmadT

def F_get_dry_air_density(P_Pa,T_K,B_H2OVMR):
    '''calculate dry air density in molec/cm3 given P, T, water vapor dry air vmr'''
    return P_Pa/(1+B_H2OVMR)/T_K/1.38064852e-23*1e-6

def F_gravity(z,g0=9.80991,Re=6378.1):
    return g0*np.square(Re/(Re+z))

def F_level2layer(profiles):
    '''
    generate altitude. level to layer for pressure, temperature, and altitude. assuming layer P
    averged from levels, then interpolate T/z at layer P
    '''
    profiles['P_layer'] = np.nanmean(
        np.column_stack(
            (profiles['P_level'][:-1],profiles['P_level'][1:])),1)
    f = interp1d(profiles['P_level'],profiles['T_level'])
    profiles['T_layer'] = f(profiles['P_layer'])
    f = interp1d(profiles['P_level'],profiles['z_level'])
    profiles['z_layer'] = f(profiles['P_layer'])
    profiles['z_level_calc'] = np.concatenate(([0],
                    np.cumsum(
                        8.3145*profiles['T_layer']/0.02896/F_gravity(profiles['z_layer'])*\
                            np.log(profiles['P_level'][:-1]/profiles['P_level'][1:])*1e-3)
                        ))
    f = interp1d(profiles['P_level'],profiles['z_level_calc'])
    profiles['z_layer_calc'] = f(profiles['P_layer'])
    return profiles

def F_noise_model(radiance,dw,nsample,dt,dp,f_number,system_efficiency,readout_e,
                  radiance_unit='photons/s/cm2/sr/nm'):
    rus = radiance_unit.split(' ')
    if len(rus) == 1:
        factor = 1.
        unit = rus[0]
    elif len(rus) == 2:
        factor = float(rus[0])
        unit = rus[1]
    # signal electrons
    S = np.pi/4*nsample*(factor*radiance)*dt*dw*(dp/f_number)**2*system_efficiency
    # noise electrons
    N = np.sqrt(S+readout_e**2)
    SNR = S/N
    return SNR

def convert_cov_cor(cov=None,cor=None,stds=None):
    '''conversion between covariance matrix and correlation matrix/diag std
    cov:
        covariance matrix
    cor:
        correlation coefficient matrix
    stds:
        sqrt(diag(cov))
    '''
    if cov is None:
        return np.outer(stds,stds) * cor
    if cor is None:
        if stds is None:
            stds = np.sqrt(np.diag(cov))
        return cov / np.outer(stds,stds)

def get_Gamma(x0s,x1s):
    '''get a matrix to map profile defined at x0s to one defined at x1s
    x0s:
        original vertical coordinate
    x1s:
        target vertical coordinate
    returns:
        gamma_matrix
    '''
    gamma_matrix = np.zeros((len(x1s),len(x0s)))
    if not np.all(x0s[:-1] >= x0s[1:]):
        logging.error('x0s shall be descending!')
        return
    if not np.all(x1s[:-1] >= x1s[1:]):
        logging.error('x1s shall be descending!')
        return
    for ix1,x1 in enumerate(x1s):
        if x1 >= x0s[0]:
            gamma_matrix[ix1,0] = 1.
            continue
        if x1 <= x0s[-1]:
            gamma_matrix[ix1,-1] = 1.
            continue
        nearest_index = np.argmin(np.abs(x0s-x1))
        nearest_x0 = x0s[nearest_index]
        if nearest_x0 >= x1:
            next_index = nearest_index + 1
        else:
            next_index = nearest_index - 1
        next_x0 = x0s[next_index]
        if x1 == nearest_x0:
            nearest_weight = 1.
            next_weight = 0.
        else:
            nearest_weight = 1/np.abs(x1-nearest_x0)
            next_weight = 1/np.abs(x1-next_x0)
        gamma_matrix[ix1,nearest_index] = nearest_weight/(nearest_weight+next_weight)
        gamma_matrix[ix1,next_index] = next_weight/(nearest_weight+next_weight)

    return gamma_matrix

class Longwave(object):
    '''class representing a band in the longwave. Custom RTM based on EPS237'''
    def __init__(self,start_w,end_w,gas_names,
                 absco_path_pattern='/home/kangsun/N2O/n2o_run/data/splat_data/SAO_crosssections/splatv2_xsect/HITRAN2020_*_4500-4650nm_0p00_0p002dw.nc',
                 ):
        '''
        start/end_w:
            start/end wavelength of the sensor in nm
        gas_names:
            list of gas names, e.g.,['N2O']
        '''
        self.logger = logging.getLogger(__name__)
        self.start_w = start_w
        self.end_w = end_w
        self.gas_names = gas_names
        
        abscos = {}
        for igas,gas in enumerate(gas_names):
            absco_fn = absco_path_pattern.replace('*',gas)
            self.logger.info(f'loading {absco_fn}')
            with Dataset(absco_fn,'r') as nc:
                absco_T = nc['Temperature'][:].filled(np.nan)
                absco_P = nc['Pressure'][:].filled(np.nan)
                absco_B = nc['Broadener_01_VMR'][:].filled(np.nan)
                absco_w = nc['Wavelength'][:].filled(np.nan)
                w_mask = (absco_w>=start_w) & (absco_w<=end_w)
                absco_w = absco_w[w_mask]
                absco_sigma = nc['CrossSection'][:,:,:,w_mask].filled(np.nan)
            abscos[gas] = dict(absco_T=absco_T,absco_P=absco_P,absco_B=absco_B,\
                               absco_w=absco_w,absco_sigma=absco_sigma)
            if igas == 0:
                self.w1 = absco_w
                self.dw1 = np.abs(np.mean(np.diff(self.w1)))
            else:
                if not np.array_equal(self.w1, absco_w):
                    self.logger.warning(f'wavelength grid of {gas} differs from {gas_names[0]}')
        self.logger.info('absco at {:.3f}-{:.3f} nm, sampling at {:.3f} nm'.format(absco_w.min(),absco_w.max(),self.dw1))
        self.abscos = abscos   
    
    def set_property(self,vza=0.,Ts=None,TC=0.,emissivity=1.,
                     dw=0.1,nsample=3,hw1e=None,
                     dt=0.1,dp=18e-4,f_number=2,system_efficiency=0.5,readout_e=60,
                     profile_path='/home/kangsun/N2O/n2o_run/data/additional_inputs/test_profile.nc',
                     radiance_unit='1e14 photons/s/cm2/sr/nm'):
        '''update property without reloading absco tables, which is slow
        vza:
            viewing zenith angle in degree
        Ts:
            surface temperature in K. if none, use bottom level temperature
        TC:
            surface temperature - lowest atmospheric level temperature
        emissivity:
            surface emissivity
        dw:
            sensor spectral sampling in nm
        nsample:
            slit width/ILS FWHM as multiple of dw
        hw1e:
            gaussian half widith at 1/e. if none, calculate as dw*nsample/1.665109
        '''
        self.radiance_unit = radiance_unit
        hw1e = hw1e or dw*nsample/1.665109
        self.dw = dw
        self.hw1e = hw1e
        self.nsample = nsample
        self.dt = dt
        self.dp = dp
        self.f_number = f_number
        self.system_efficiency = system_efficiency
        self.readout_e = readout_e
        gas_names = self.gas_names
        profiles = {}
        self.logger.info(f'loading {profile_path}')
        # the profiles go from surface to toa, opposite from splat profiles
        with Dataset(profile_path,'r') as nc:
            profiles['P_level'] = nc['pedge'][:].squeeze().filled(np.nan)[::-1]
            profiles['T_level'] = nc['Tedge'][:].squeeze().filled(np.nan)[::-1]
            for gas in gas_names:
                profiles[gas] = nc[gas][:].squeeze().filled(np.nan)[::-1]
            if 'H2O' not in gas_names:
                profiles['H2O'] = nc['H2O'][:].squeeze().filled(np.nan)[::-1]
        
        # translate to ccm's variable name, pedge_us is toa->sfc
        pedge_us = profiles['P_level'][::-1]
        Tedge_us = profiles['T_level'][::-1]
        lmx = pedge_us.shape[0]-1
        # Vertical grid
        H = R*Tedge_us/9.81/28.97e-3
        # zedge is also toa->sfc
        zedge = np.zeros_like(Tedge_us)
        for l in range(lmx-1,-1,-1):
            zedge[l] = zedge[l+1] + H[l+1]*1e-3*np.log(pedge_us[l+1]/pedge_us[l])
        profiles['z_level'] = zedge[::-1]  
        # # somehow there is no alitutde in the test_profile.nc file
        # profiles['z_level'] = np.array([80.0,60.0,55.0,50.0,47.5,45.0,42.5,40.0,\
        #                        37.5,35.0,32.5,30.0,27.5,25.0,24.0,23.0,\
        #                        22.0,21.0,20.0,19.0,18.0,17.0,16.0,15.0,\
        #                        14.0,13.0,12.0,11.0,10.0,9.0,8.0,7.0,6.0,\
        #                        5.0,4.0,3.0,2.0,1.0,0.0])[::-1]
        profiles = F_level2layer(profiles)
        profiles['dz'] = np.abs(profiles['z_level_calc'][1:]-profiles['z_level_calc'][:-1])
        profiles['air_density'] = np.zeros(profiles['dz'].shape,dtype=np.float64)
        self.profiles = profiles
        self.nlevel = len(profiles['P_level'])
        self.nlayer = self.nlevel-1
        for ilayer in range(self.nlayer):
            P = self.profiles['P_layer'][ilayer]*100.# hPa to Pa
            T = self.profiles['T_layer'][ilayer]
            B = self.profiles['H2O'][ilayer]
            # air density in molec/cm3
            self.profiles['air_density'][ilayer] = F_get_dry_air_density(P,T,B)
        self.Ts = Ts or self.profiles['T_level'][np.argmin(self.profiles['z_level'])]+TC
        self.emissivity = emissivity
        self.vza = vza
        
    def level2layer(self,keys=None):
        '''
        level to layer for pressure, temperature, and altitude. assuming layer P
        averged from levels, then interpolate T/z at layer P
        '''
        keys = keys or ['P','T','z']
        for k in keys:
            if k == 'P':
                self.profiles['P_layer'] = np.nanmean(
                    np.column_stack(
                        (self.profiles['P_level'][:-1],self.profiles['P_level'][1:])),1)
            else:
                f = interp1d(self.profiles['P_level'],self.profiles[f'{k}_level'])
                self.profiles[f'{k}_layer'] = f(self.profiles['P_layer'])
    
    def plot_jacobian(self,key,ax=None,vertical_coordinate='P_layer',
                      **kwargs):
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(10,5),constrained_layout=True)
        else:
            fig = None
        ydata = self.profiles[vertical_coordinate]
        xdata = self.jacs['Wavelength']
        pc = ax.pcolormesh(*F_center2edge(xdata,ydata),self.jacs[key],**kwargs)
        if vertical_coordinate=='P_layer':
            ax.invert_yaxis()
        return dict(fig=fig,ax=ax,pc=pc)
    
    def get_jacobians(self,keys,if_convolve=True,
                      **kwargs):
        jacs = {}
        
        if if_convolve:
            HW1E = self.hw1e
            dw1 = self.dw1
            ndx = np.ceil(HW1E*5/dw1);
            xx = np.arange(ndx*2)*dw1-ndx*dw1;
            ILS = 1/np.sqrt(np.pi)/HW1E*np.exp(-np.power(xx/HW1E,2))*dw1
            dILSdHW1E = 1/np.sqrt(np.pi)*(-1/np.power(HW1E,2)+2*np.power(xx,2)/np.power(HW1E,4))*np.exp(-np.power(xx/HW1E,2))*dw1
            w2 = arange_(self.start_w, self.end_w, self.dw)
        else:
            w2 = self.w1
        
        jacs['Wavelength'] = w2
        for ikey,key in enumerate(keys):
            # convert wavelength,layer to layer,wavelength
            jac = self.get_jacobian(key,**kwargs).T
            
            if if_convolve:
                jacs[key] = np.zeros((jac.shape[0],len(w2)))
                for i,ja in enumerate(jac):
                    f = interp1d(self.w1, convolve_fft(ja,ILS,normalize_kernel=True),
                                 bounds_error=False,fill_value='extrapolate')
                    jacs[key][i,] = f(w2)
            else:
                jacs[key] = jac
            self.logger.info('got {} jacobian with shape {}'.format(key,jacs[key].shape))
        if not if_convolve:
            jacs['Radiance'] = self.get_radiance()
        else:
            f = interp1d(self.w1, convolve_fft(self.get_radiance(),ILS,normalize_kernel=True),
                         bounds_error=False,fill_value='extrapolate')
            jacs['Radiance'] = f(w2)
        self.logger.info('calculating radiance error')
        SNR = F_noise_model(jacs['Radiance'],self.dw,self.nsample,
                            self.dt,self.dp,self.f_number,
                            self.system_efficiency,self.readout_e,
                            self.radiance_unit)
        jacs['SNR'] = SNR
        jacs['Radiance_error'] = jacs['Radiance']/SNR
        self.jacs = jacs
    
    def get_jacobian(self,key,finite_difference=False,fd_delta=None,
                     fd_relative_delta=None,
                     wrt='mixing ratio'):
        ''' 
        key:
            a key name in self.profile or a scalar attribute of self
        finite_difference:
            true if fd, false means analytical
        fd_delta:
            delta value used in finite difference, array same shape as a profile
        fd_relative_delta:
            relative delta value used in finite difference. ignored if fd_delta
            is provided
        wrt:
            gas jacobian with respect to 'relative mixing ratio','mixing ratio', 
            'number density', or 'optical depth'
        '''
        if finite_difference:
            if key in ['Ts']:
                self.get_sigmas()
                
                key0 = getattr(self,key)
                if fd_delta is None and fd_relative_delta is None:
                    self.logger.warning('a relative fd_relative_delta of 0.001 is assumed')
                    fd_relative_delta = 1e-3
                if fd_delta is None and fd_relative_delta is not None:
                    fd_delta = key0*fd_relative_delta
                
                setattr(self,key,key0-fd_delta/2)
                r_l = self.get_radiance()
                
                setattr(self,key,key0+fd_delta/2)
                r_r = self.get_radiance()
                
                jac = ((r_r-r_l)/fd_delta)[:,np.newaxis]
            else:
                profile_c = self.profiles[key].copy()
                if key == 'T_level':
                    profile_layer_c = self.profiles['T_layer'].copy()  
                    nz = self.nlevel
                else:
                    nz = self.nlayer
                if fd_delta is None and fd_relative_delta is None:
                    self.logger.warning('a relative fd_relative_delta of 0.001 is assumed')
                    fd_relative_delta = 1e-3
                if fd_delta is None and fd_relative_delta is not None:
                    fd_delta = profile_c*fd_relative_delta
                profile_l = profile_c-fd_delta/2.
                profile_r = profile_c+fd_delta/2.
                jac = np.full((len(self.w1),nz),np.nan)
                for iz in range(nz):
                    self.profiles[key][iz] = profile_l[iz]
                    if key == 'T_level':
                        self.level2layer(keys=['T'])
                    self.get_sigmas()
                    r_l = self.get_radiance()
                    self.profiles[key][iz] = profile_r[iz]
                    if key == 'T_level':
                        self.level2layer(keys=['T'])
                    self.get_sigmas()
                    r_r = self.get_radiance()
                    jac[:,iz] = (r_r-r_l)/fd_delta[iz]
                # reset profiles and tau to original values (without perturbations)
                self.profiles[key] = profile_c.copy()
                if key == 'T_level':
                    self.profiles['T_layer'] = profile_layer_c.copy()
                self.get_sigmas()
        else:
            if key in ['Ts']:
                self.get_sigmas()
                jac = np.full((len(self.w1),1),np.nan)
                Ts = self.Ts
                if callable(self.emissivity):
                    emissivity = self.emissivity(self.w1)
                else:
                    emissivity = self.emissivity
                vza = self.vza
                mu = np.cos(vza/180*np.pi)
                radiance_unit = self.radiance_unit
                # surface planck function
                Bs,dBsdTs = BB(Ts,self.w1,radiance_unit,do_dBdT=True)
                # planck function for layer and level temperatures, sfc->toa
                B_level = np.zeros((self.nlevel,len(self.w1)))
                B_layer = np.zeros((self.nlayer,len(self.w1)))
                dB_layerdT = np.zeros((self.nlayer,len(self.w1)))
                for ilevel in range(self.nlevel):
                    B_level[ilevel,] = BB(self.profiles['T_level'][ilevel],self.w1,radiance_unit)
                for ilayer in range(self.nlayer):
                    B_layer[ilayer,], dB_layerdT[ilayer,] = BB(
                        self.profiles['T_layer'][ilayer],
                        self.w1,radiance_unit,do_dBdT=True)
                # slant optical thickness, sfc->toa
                dtau_layer_mu = self.dtau_layer/mu
                jac[:,0] = emissivity * dBsdTs * np.exp(-np.nansum(dtau_layer_mu,axis=0))
                
            elif key in ['T_layer']:
                self.get_sigmas(do_dsigmadT=True)
                jac = np.full((len(self.w1),self.nlayer),np.nan)
                Ts = self.Ts
                if callable(self.emissivity):
                    emissivity = self.emissivity(self.w1)
                else:
                    emissivity = self.emissivity
                vza = self.vza
                mu = np.cos(vza/180*np.pi)
                radiance_unit = self.radiance_unit
                # surface planck function
                Bs = BB(Ts,self.w1,radiance_unit)
                Bs *= emissivity
                # planck function for layer and level temperatures, sfc->toa
                B_level = np.zeros((self.nlevel,len(self.w1)))
                B_layer = np.zeros((self.nlayer,len(self.w1)))
                dB_layerdT = np.zeros((self.nlayer,len(self.w1)))
                for ilevel in range(self.nlevel):
                    B_level[ilevel,] = BB(self.profiles['T_level'][ilevel],self.w1,radiance_unit)
                for ilayer in range(self.nlayer):
                    B_layer[ilayer,], dB_layerdT[ilayer,] = BB(
                        self.profiles['T_layer'][ilayer],
                        self.w1,radiance_unit,do_dBdT=True)
                # slant optical thickness, sfc->toa
                dtau_layer_mu = self.dtau_layer/mu
                dtau_layer_mudT = self.dtau_layerdT/mu
                # cumulative slant optical thickness at levels, sfc->toa
                tau_level_mu = np.concatenate(
                    (np.zeros((1,len(self.w1))),np.cumsum(dtau_layer_mu,axis=0)),axis=0)
                # B_tau is the the effective Planck function varying from B_layer in the 
                # optically thin regime to B_level[1:,] in the optically thick regime.
                # see Eq. 16 in https://doi.org/10.1029/92JD01419
                a_pade2 = 0.193; b_pade2 = 0.013;
                upper_weight = a_pade2*dtau_layer_mu+b_pade2*dtau_layer_mu**2
                B_tau = (B_layer+upper_weight*B_level[1:,])/(1+upper_weight)
                dB_taudtau = (B_level[1:,]-B_layer)*(a_pade2+2*b_pade2*dtau_layer_mu)/\
                    np.square(1+upper_weight)
                dB_taudB_layer = 1/(1+upper_weight)
                dB_taudT = dB_taudtau*dtau_layer_mudT+dB_taudB_layer*dB_layerdT
                
                for ilayer in range(self.nlayer):
                    jac[:,ilayer] = -Bs*np.exp(-tau_level_mu[-1,])*dtau_layer_mudT[ilayer,]+\
                        (np.exp(-dtau_layer_mu[ilayer,])*dtau_layer_mudT[ilayer,]*B_tau[ilayer,]+\
                         (1-np.exp(-dtau_layer_mu[ilayer,]))*dB_taudT[ilayer,])*\
                            np.exp(tau_level_mu[ilayer+1,]-tau_level_mu[-1,])
                    if ilayer > 0:
                        jac[:,ilayer] -= np.nansum((1-np.exp(-dtau_layer_mu[:ilayer,]))\
                                   *B_tau[:ilayer,]\
                                   *np.exp(tau_level_mu[1:ilayer+1,]-tau_level_mu[-1,])
                                   *dtau_layer_mudT[ilayer,]\
                                   ,axis=0)
                    
            elif key in self.gas_names:
                self.get_sigmas()
                jac = np.full((len(self.w1),self.nlayer),np.nan)
                Ts = self.Ts
                if callable(self.emissivity):
                    emissivity = self.emissivity(self.w1)
                else:
                    emissivity = self.emissivity
                vza = self.vza
                mu = np.cos(vza/180*np.pi)
                radiance_unit = self.radiance_unit
                # surface planck function
                Bs = emissivity*BB(Ts,self.w1,radiance_unit)
                # planck function for layer and level temperatures, sfc->toa
                B_level = np.zeros((self.nlevel,len(self.w1)))
                B_layer = np.zeros((self.nlayer,len(self.w1)))
                for ilevel in range(self.nlevel):
                    B_level[ilevel,] = BB(self.profiles['T_level'][ilevel],self.w1,radiance_unit)
                for ilayer in range(self.nlayer):
                    B_layer[ilayer,] = BB(self.profiles['T_layer'][ilayer],self.w1,radiance_unit)
                # slant optical thickness, sfc->toa
                dtau_layer_mu = self.dtau_layer/mu
                # cumulative slant optical thickness at levels, sfc->toa
                tau_level_mu = np.concatenate(
                    (np.zeros((1,len(self.w1))),np.cumsum(dtau_layer_mu,axis=0)),axis=0)
                # B_tau is the the effective Planck function varying from B_layer in the 
                # optically thin regime to B_level[1:,] in the optically thick regime.
                # see Eq. 16 in https://doi.org/10.1029/92JD01419
                a_pade2 = 0.193; b_pade2 = 0.013;
                upper_weight = a_pade2*dtau_layer_mu+b_pade2*dtau_layer_mu**2
                B_tau = (B_layer+upper_weight*B_level[1:,])/(1+upper_weight)
                dB_dtau = (B_level[1:,]-B_layer)*(a_pade2+2*b_pade2*dtau_layer_mu)/\
                    np.square(1+upper_weight)
                
                for ilayer in range(self.nlayer):
                    jac[:,ilayer] = -Bs*np.exp(-tau_level_mu[-1,])+\
                        (np.exp(-dtau_layer_mu[ilayer,])*B_tau[ilayer,]+\
                         (1-np.exp(-dtau_layer_mu[ilayer,]))*dB_dtau[ilayer,])*\
                            np.exp(tau_level_mu[ilayer+1,]-tau_level_mu[-1,])
                    if ilayer > 0:
                        jac[:,ilayer] -= np.nansum((1-np.exp(-dtau_layer_mu[:ilayer,]))\
                                   *B_tau[:ilayer,]\
                                   *np.exp(tau_level_mu[1:ilayer+1,]-tau_level_mu[-1,])\
                                   ,axis=0)
                    if wrt in ['mixing ratio']:
                        jac[:,ilayer] *= self.sigmas[key][ilayer,]*\
                            self.profiles['dz'][ilayer]*1e5/mu*\
                                self.profiles['air_density'][ilayer]
                    elif wrt in ['relative mixing ratio']:
                        jac[:,ilayer] *= self.sigmas[key][ilayer,]*\
                            self.profiles['dz'][ilayer]*1e5/mu*\
                                self.profiles['air_density'][ilayer]*\
                                    self.profiles[key][ilayer]
                    elif wrt in ['number density']:
                        jac[:,ilayer] *= self.sigmas[key][ilayer,]*\
                            self.profiles['dz'][ilayer]*1e5/mu
            else:
                self.logger.error('not implemented')
                return
        return jac                
        
    def get_sigmas(self,do_dsigmadT=False):
        sigmas = {}
        for gas in self.gas_names:
            sigmas[gas] = np.zeros((self.nlayer,len(self.w1)))
        dtau_layer = np.zeros((self.nlayer,len(self.w1)))
        if do_dsigmadT:
            dtau_layerdT = np.zeros((self.nlayer,len(self.w1)))
        for ilayer in range(self.nlayer):
            P = self.profiles['P_layer'][ilayer]*100.# hPa to Pa
            T = self.profiles['T_layer'][ilayer]
            B = self.profiles['H2O'][ilayer]
            dz = self.profiles['dz'][ilayer]*1e5 # km to cm
            # air density in molec/cm3
            air_density = self.profiles['air_density'][ilayer]
            for gas in self.gas_names:
                if do_dsigmadT:
                    sigmas[gas][ilayer,], dsigmadT = F_interp_absco(
                        self.abscos[gas]['absco_P'], 
                        self.abscos[gas]['absco_T'], 
                        self.abscos[gas]['absco_B'], 
                        self.abscos[gas]['absco_w'], 
                        self.abscos[gas]['absco_sigma'], 
                        P, T, B, self.w1,do_dsigmadT=True)
                else:
                    sigmas[gas][ilayer,] = F_interp_absco(self.abscos[gas]['absco_P'], 
                                             self.abscos[gas]['absco_T'], 
                                             self.abscos[gas]['absco_B'], 
                                             self.abscos[gas]['absco_w'], 
                                             self.abscos[gas]['absco_sigma'], 
                                             P, T, B, self.w1)
                # gas density in molec/cm3
                gas_density = air_density*self.profiles[gas][ilayer]
                # optical thickness in that layer
                dtau_layer[ilayer,] += sigmas[gas][ilayer,]*gas_density*dz
                if do_dsigmadT:
                    dtau_layerdT[ilayer,] += dsigmadT*gas_density*dz
        self.sigmas = sigmas
        self.dtau_layer = dtau_layer
        if do_dsigmadT:
            self.dtau_layerdT = dtau_layerdT
    
    def update_profile(self,key,absolute_value=None,reset_profiles=None):
        '''update gas profile value
        reset_profile:
            dict. self.profile will be reset to it before the perturbation
        '''
        if reset_profiles is not None:
            self.profiles = reset_profiles.copy()
        if absolute_value is not None:
            self.profiles[key] = np.ones_like(self.profiles[key])*absolute_value
        dtau_layer = np.zeros((self.nlayer,len(self.w1)))
        for ilayer in range(self.nlayer):
            dz = self.profiles['dz'][ilayer]*1e5 # km to cm
            # air density in molec/cm3
            air_density = self.profiles['air_density'][ilayer]
            for gas in self.gas_names:
                # gas density in molec/cm3
                gas_density = air_density*self.profiles[gas][ilayer]
                # optical thickness in that layer
                dtau_layer[ilayer,] += self.sigmas[gas][ilayer,]*gas_density*dz
        self.dtau_layer = dtau_layer
    
    def get_radiance(self):
        '''
        note the profile dimensions go from surface to TOA in this function
        '''
        Ts = self.Ts
        if callable(self.emissivity):
            emissivity = self.emissivity(self.w1)
        else:
            emissivity = self.emissivity
        vza = self.vza
        mu = np.cos(vza/180*np.pi)
        radiance_unit = self.radiance_unit
        # surface planck function
        Bs = emissivity*BB(Ts,self.w1,radiance_unit)
        # planck function for layer and level temperatures, sfc->toa
        B_level = np.zeros((self.nlevel,len(self.w1)))
        B_layer = np.zeros((self.nlayer,len(self.w1)))
        for ilevel in range(self.nlevel):
            B_level[ilevel,] = BB(self.profiles['T_level'][ilevel],self.w1,radiance_unit)
        for ilayer in range(self.nlayer):
            B_layer[ilayer,] = BB(self.profiles['T_layer'][ilayer],self.w1,radiance_unit)
        # slant optical thickness, sfc->toa
        dtau_layer_mu = self.dtau_layer/mu
        # cumulative slant optical thickness at levels, sfc->toa
        tau_level_mu = np.concatenate(
            (np.zeros((1,len(self.w1))),np.cumsum(dtau_layer_mu,axis=0)),axis=0)
        # B_tau is the the effective Planck function varying from B_layer in the 
        # optically thin regime to B_level[1:,] in the optically thick regime.
        # see Eq. 16 in https://doi.org/10.1029/92JD01419
        a_pade2 = 0.193; b_pade2 = 0.013;
        upper_weight = a_pade2*dtau_layer_mu+b_pade2*dtau_layer_mu**2
        B_tau = (B_layer+upper_weight*B_level[1:,])/(1+upper_weight)
        # radiance as of https://doi.org/10.1029/92JD01419
        radiance = Bs*np.exp(-tau_level_mu[-1,:])\
            +np.nansum((1-np.exp(-dtau_layer_mu))\
                       *B_tau\
                       *np.exp(tau_level_mu[1:,]-tau_level_mu[-1,])\
                       ,axis=0)
        self.B_level = B_level
        self.Bs = Bs
        return radiance


class Shortwave(object):
    '''class representing a band in the shortwave. Wrapping SPLAT'''
    def __init__(self,start_w,end_w,gas_names,sza=30.,vza=0.,
                 dw=0.1,nsample=3,hw1e=None,
                 dt=0.1,dp=18e-4,f_number=2,system_efficiency=0.5,readout_e=60,
                 splat_path='/home/kangsun/N2O/sci-level2-splat/build/splat.exe',
                 control_template_path='/home/kangsun/N2O/n2o_run/control/forward_template.control',
                 working_dir='/home/kangsun/N2O/n2o_run',
                 profile_path='/home/kangsun/N2O/n2o_run/data/additional_inputs/test_profile.nc',
                 radiance_unit='1e14 photons/s/cm2/sr/nm'):
        '''
        splat_path:
            path of splat.exe
        control_template_path:
            path to a template control file, variables in {{}} will be replaced
        working_dir:
            dir to run splat
        profile_path:
            path to profile nc file input to splat. splat profiles are converted from surface to toa
        '''
        self.logger = logging.getLogger(__name__)
        self.start_w = start_w
        self.end_w = end_w
        self.sza = sza
        self.vza = vza
        hw1e = hw1e or dw*nsample/1.665109
        self.dw = dw
        self.hw1e = hw1e
        self.nsample = nsample
        self.dt = dt
        self.dp = dp
        self.f_number = f_number
        self.system_efficiency = system_efficiency
        self.readout_e = readout_e
        self.radiance_unit = radiance_unit
        self.gas_names = gas_names
        self.splat_path = splat_path
        self.working_dir = working_dir
        self.profile_path = profile_path
        self.control_template_path = control_template_path
        profiles = {}
        self.logger.info(f'loading {profile_path}')
        # the profiles go from surface to toa, opposite from splat profiles
        with Dataset(profile_path,'r') as nc:
            profiles['P_level'] = nc['pedge'][:].squeeze().filled(np.nan)[::-1]
            profiles['T_level'] = nc['Tedge'][:].squeeze().filled(np.nan)[::-1]
            for gas in gas_names:
                profiles[gas] = nc[gas][:].squeeze().filled(np.nan)[::-1]
            if 'H2O' not in gas_names:
                profiles['H2O'] = nc['H2O'][:].squeeze().filled(np.nan)[::-1]
        
        # translate to ccm's variable name, pedge_us is toa->sfc
        pedge_us = profiles['P_level'][::-1]
        Tedge_us = profiles['T_level'][::-1]
        lmx = pedge_us.shape[0]-1
        # Vertical grid
        H = R*Tedge_us/9.81/28.97e-3
        # zedge is also toa->sfc
        zedge = np.zeros_like(Tedge_us)
        for l in range(lmx-1,-1,-1):
            zedge[l] = zedge[l+1] + H[l+1]*1e-3*np.log(pedge_us[l+1]/pedge_us[l])
        profiles['z_level'] = zedge[::-1]  
        # # somehow there is no alitutde in the test_profile.nc file
        # profiles['z_level'] = np.array([80.0,60.0,55.0,50.0,47.5,45.0,42.5,40.0,\
        #                        37.5,35.0,32.5,30.0,27.5,25.0,24.0,23.0,\
        #                        22.0,21.0,20.0,19.0,18.0,17.0,16.0,15.0,\
        #                        14.0,13.0,12.0,11.0,10.0,9.0,8.0,7.0,6.0,\
        #                        5.0,4.0,3.0,2.0,1.0,0.0])[::-1]
        profiles = F_level2layer(profiles)
        profiles['dz'] = np.abs(profiles['z_level_calc'][1:]-profiles['z_level_calc'][:-1])
        profiles['air_density'] = np.zeros(profiles['dz'].shape,dtype=np.float64)
        self.profiles = profiles
        self.nlevel = len(profiles['P_level'])
        self.nlayer = self.nlevel-1
        for ilayer in range(self.nlayer):
            P = self.profiles['P_layer'][ilayer]*100.# hPa to Pa
            T = self.profiles['T_layer'][ilayer]
            B = self.profiles['H2O'][ilayer]
            # air density in molec/cm3
            self.profiles['air_density'][ilayer] = F_get_dry_air_density(P,T,B)
    
    def plot_jacobian(self,key,ax=None,vertical_coordinate='P_layer',
                      **kwargs):
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(10,5),constrained_layout=True)
        else:
            fig = None
        ydata = self.profiles[vertical_coordinate]
        xdata = self.jacs['Wavelength']
        pc = ax.pcolormesh(*F_center2edge(xdata,ydata),self.jacs[key],**kwargs)
        if vertical_coordinate=='P_layer':
            ax.invert_yaxis()
        return dict(fig=fig,ax=ax,pc=pc)
    
    def get_jacobians(self,keys=None,keynames=None,finite_difference=False,delete_nc=False,
                      if_convolve=True,wrt='mixing ratio',
                      **kwargs):
        jacs = {}
        keys = keys or ['RTM_Band1/Radiance_I']+\
            [f'RTM_Band1/{g.upper()}_TraceGasJacobian_I' for g in self.gas_names]
        keynames = keynames or ['Radiance']+self.gas_names
        if finite_difference:
            pass
        else:
            self.run_splat(**kwargs)
            if if_convolve:
                HW1E = self.hw1e
                dw1 = self.dw1
                ndx = np.ceil(HW1E*5/dw1);
                xx = np.arange(ndx*2)*dw1-ndx*dw1;
                ILS = 1/np.sqrt(np.pi)/HW1E*np.exp(-np.power(xx/HW1E,2))*dw1
                dILSdHW1E = 1/np.sqrt(np.pi)*(-1/np.power(HW1E,2)+2*np.power(xx,2)/np.power(HW1E,4))*np.exp(-np.power(xx/HW1E,2))*dw1
                w2 = arange_(self.start_w, self.end_w, self.dw)
            with Dataset(self.output_path,'r') as nc:
                w1 = nc['RTM_Band1/Wavelength'][:,0,0].filled(np.nan)
                if not if_convolve:
                    mask = (w1>=self.start_w) & (w1<=self.end_w)
                    w2 = w1[mask]
                for k,kn in zip(keys,keynames):
                    if nc[k].ndim == 3:
                        if if_convolve:
                            s1 = convolve_fft(nc[k][:,0,0].filled(np.nan),ILS,normalize_kernel=True)
                            f = interp1d(w1,s1,bounds_error=False,fill_value='extrapolate')
                            jacs[kn] = f(w2)
                        else:
                            jacs[kn] = nc[k][mask,0,0].filled(np.nan)
                    elif nc[k].ndim == 4:
                        jac = np.zeros((nc[k].shape[0],len(w2)))
                        for i in range(nc[k].shape[0]):
                            if if_convolve:
                                f = interp1d(w1, convolve_fft(nc[k][i,:,0,0].filled(np.nan),ILS,normalize_kernel=True),
                                             bounds_error=False,fill_value='extrapolate')
                                jac[i,] = f(w2)
                            else:
                                jac[i,] = nc[k][i,mask,0,0].filled(np.nan)
                        # make profile jacobians go from sfc to toa
                        jac = jac[::-1,]
                        if wrt in ['mixing ratio']:
                            self.logger.info(f'converting c*dI/dc to dI/dc using {kn} mixing ratios in self.profiles')
                            jac /= self.profiles[kn][:,np.newaxis]
                        jacs[kn] = jac
            jacs['Wavelength'] = w2
            if 'Radiance' in jacs.keys():
                self.logger.info('calculating radiance error')
                SNR = F_noise_model(jacs['Radiance'],self.dw,self.nsample,
                                    self.dt,self.dp,self.f_number,
                                    self.system_efficiency,self.readout_e,
                                    self.radiance_unit)
                jacs['SNR'] = SNR
                jacs['Radiance_error'] = jacs['Radiance']/SNR
            else:
                self.logger.warning('Radiance not found in jacs. no error calculation')
            self.jacs = jacs
            if delete_nc:
                os.remove(self.output_path)
         
    def run_splat(self,control_path=None,control_template_path=None,
                  output_path=None,profile_path=None,rerun_splat=True,
                  **kwargs):
        profile_path = profile_path or self.profile_path
        output_path = output_path or os.path.join(self.working_dir,'splat_output.nc')
        control_path = control_path or os.path.join(self.working_dir,'splat_run.control')
        control_template_path = control_template_path or self.control_template_path
        self.output_path = output_path
        self.control_path = control_path
        self.control_template_path = control_template_path
        # Variables to substitute in control file
        varlst = {}
        varlst['output_file'] = output_path
        varlst['sza'] = self.sza
        varlst['vza'] = self.vza
        varlst['vaa'] = kwargs.pop('vaa',10)
        varlst['saa'] = kwargs.pop('saa',0)
        varlst['aza'] = kwargs.pop('aza',10)
        varlst['start_w1'] = kwargs.pop('start_w1',self.start_w-1)
        varlst['end_w1'] = kwargs.pop('end_w1',self.end_w+1)
        self.dw1 = kwargs.pop('dw1',0.01)
        varlst['dw1'] = self.dw1
        
        varlst['splat_data_dir'] = kwargs.pop('splat_data_dir','data/splat_data/')
        varlst['atmosphere_apriori_file'] = os.path.relpath(profile_path,self.working_dir)
        if not rerun_splat:
            self.logger.info('won\'t run splat, using {} directly!'.format(self.output_path))
            return
        with open(control_template_path,'r') as template_f:
            ctrl = template_f.read()
            for v in varlst.keys():
                ctrl = ctrl.replace('{{'+v+'}}',str(varlst[v]))
            with open(control_path,'w') as f:
                f.write(ctrl)
        
        self.run(control_path)
        self.logger.info(f'output saved at {output_path}')
    
    def run(self,control_path):
        cwd = os.getcwd()
        os.chdir(self.working_dir)
        self.logger.info(f'running splat using {control_path}')
        os.system(f'{self.splat_path} {control_path}')
        os.chdir(cwd)

class Parameters(OrderedDict):
    def __init__(self):
        pass
    
    def add(self,param):
        OrderedDict.__setitem__(self,param.name,param)
    
    def get_prior(self):
        nstates = []
        for name, par in self.items():
            if not par.vary:
                continue
            nstates.append(par.nstate)
        
        Sa = np.zeros((np.sum(nstates),np.sum(nstates)))
        count = 0
        xa = np.zeros(np.sum(nstates))
        params_names = []
        for (name,par) in self.items():
            if not par.vary:
                continue
            params_names.append(name)
            Sa[count:count+par.nstate,count:count+par.nstate] = par.prior_error_matrix
            xa[count:count+par.nstate] = par.prior
            count += par.nstate
        return xa, Sa, nstates, params_names
    
    def get_interference_errors(self,A,h):
        '''calculate interference errors between two parameters
        A:
            global averaging kernel matrix
        h:
            air mass weighting factor
        '''
        for (name,par) in self.items():
            if not par.vary:
                continue
            # loop over interferers
            for (name_i,par_i) in self.items():
                if not par_i.vary or name_i == name:
                    continue
                avk_block = A[par.start_idx:par.start_idx+par.nstate,
                              par_i.start_idx:par_i.start_idx+par_i.nstate]
                Si = avk_block@par_i.prior_error_matrix@avk_block.T
                setattr(par,f'{name_i}_interference_error_matrix', Si)
                if par.nstate == 1:
                    continue
                setattr(par,f'column_{name_i}_interference_error', 
                        np.sqrt(h@Si@h))
                
    def get_column_metrics(self,names,h):
        '''get column quantities of profiles
        names:
            a list of keys
        h:
            air mass weighting factor
        '''
        for (name,par) in self.items():
            if not par.vary or name not in names:
                continue
            setattr(par,'column_posterior_error',
                    np.sqrt(h@par.posterior_error_matrix@h))
            setattr(par,'column_measurement_error',
                    np.sqrt(h@par.measurement_error_matrix@h))
            setattr(par,'column_prior_error',
                    np.sqrt(h@par.prior_error_matrix@h))
            setattr(par,'column_AVK',
                    h@par.averaging_kernel/h)
            setattr(par,'dofs',
                    par.averaging_kernel.trace())
    
    def update_vectors(self,vector_name,vector):
        for (name,par) in self.items():
            if not par.vary:
                continue
            new_values = vector[par.start_idx:par.start_idx+par.nstate]
            setattr(self[name],vector_name,new_values)
    
    def update_matrices(self,matrix_name,matrix):
        for (name,par) in self.items():
            if not par.vary:
                continue
            setattr(self[name],matrix_name,
                    matrix[par.start_idx:par.start_idx+par.nstate,
                           par.start_idx:par.start_idx+par.nstate])

class Parameter:
    def __init__(self,name,prior,value=None,prior_error=None,
                 correlation_matrix=None,correlation_km=None,z_km=None,
                 vary=True):
        '''initialize Parameter class
        name:
            name of the state vector component
        prior:
            prior value(s) of the state component
        value:
            current value, copy prior if none
        prior_error:
            prior error value(s) of the state component
        correlation_matrix:
            matrix of error correlation coefficient. if provided, 
            correlation_km and z_km will not be used
        correlation_km:
            if it is a profile, use this correlation length to construct prior
            matrix
        z_km:
            altitudes in km
        vary:
            whether this parameter is optimized or not
        '''
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.vary = vary
        self.prior = prior
        if value is None:
            self.value = prior
        if prior_error is None:
            self.prior_error = 0.1 * prior
            self.logger.warning('assuming 10% prior error')
        else:
            self.prior_error = prior_error
        
        if np.isscalar(prior):
            self.nstate = 1
            self.prior_error_matrix = np.array([[self.prior_error**2]])
            return
        self.nstate = len(prior)
        
        if correlation_matrix is not None:
            self.prior_error_matrix = convert_cov_cor(cor=correlation_matrix,
                                                      stds=self.prior_error)
        else:
            self.prior_error_matrix = np.diag(self.prior_error**2)
            self.correlation_km = correlation_km
            if correlation_km is not None and z_km is not None:
                for iz,(ape_i,z_i) in enumerate(zip(self.prior_error,z_km)):
                    for jz,(ape_j,z_j) in enumerate(zip(self.prior_error,z_km)):
                        if iz == jz: 
                            continue
                        elif iz < jz:
                            self.prior_error_matrix[iz,jz] = ape_i*ape_j*\
                            np.exp(-np.abs(z_i-z_j)/correlation_km)
                        else:# prior matrix is symmetric
                            self.prior_error_matrix[iz,jz] =\
                            self.prior_error_matrix[jz,iz]

class LSA(object):
    def __init__(self,bands,param_names,band_weights=None):
        '''
        bands:
            a list of Longwave/Shortwave objects
        param_names:
            a list of parameter names for inversion
        band_weights:
            if provided, a list the same size as bands to weigh each band by
            dividing the bands radiance error with the corresponding number
        '''
        self.logger = logging.getLogger(__name__)
        self.bands = bands
        if band_weights is None:
            band_weights = np.ones(len(bands))
        masks = [(band.jacs['Wavelength']>=band.start_w) &
                 (band.jacs['Wavelength']<=band.end_w) &
                 (~np.isnan(band.jacs['Radiance'])) for band in bands]
        self.masks = masks
        param_names = np.array(param_names)
        param_mask = np.isin(param_names,
                             list(set().union(
                                 *[set(band.jacs.keys()) for band in bands])))
        if not all(param_mask):
            self.logger.warning(
                '{} not in any bands'.format(param_names[~param_mask]))
        self.param_names = param_names[param_mask]
        
        self.param_hints = {}
                    
        self.radiance_error = np.concatenate(
            [band.jacs['Radiance_error'][mask]/weight
             for band,mask,weight in zip(bands,masks,band_weights)])
        self.radiance = np.concatenate(
            [band.jacs['Radiance'][mask] for band,mask in zip(bands,masks)])
        
        self.SNR = np.concatenate([band.jacs['SNR'][mask] 
                                   for band,mask in zip(bands,masks)])
        self.ny = np.sum([np.sum(mask) for mask in masks])
        air_col = bands[0].profiles['air_density'] * bands[0].profiles['dz']*1e5 # molec/cm2
        self.h = air_col/np.sum(air_col)
    
    def set_prior(self,name,**kwargs):
        if name not in self.param_hints:
            self.param_hints[name] = {}

        for key, val in kwargs.items():
            self.param_hints[name][key] = val
    
    def make_params(self):
        params = Parameters()
        for name in self.param_names:
            par = Parameter(name,**self.param_hints[name])
            params.add(par)
        return params
    
    def retrieve(self):
        params = self.make_params()
        xa, Sa, nstates, state_names = params.get_prior()
        K = np.zeros((self.ny,np.sum(nstates)))
        for iband, (band,mask) in enumerate(zip(self.bands,self.masks)):
            if iband == 0:
                start_y = 0
            else:
                start_y = np.sum([len(mask) for mask in self.masks[0:iband]])
            
            for ipar, (name,nstate) in enumerate(zip(state_names,nstates)):
                if name not in band.jacs.keys():
                    continue
                if ipar == 0:
                    start_x = 0
                else:
                    start_x = np.sum(nstates[0:ipar])
                setattr(params[name],'start_idx',start_x)
                K[start_y:start_y+np.sum(mask),start_x:start_x+nstate] =\
                    band.jacs[name].T
        
        Sy = np.diag(np.square(self.radiance_error))
        G = Sa@K.T@np.linalg.inv(K@Sa@K.T+Sy)
        A = G@K
        Shat = np.linalg.inv(K.T@np.linalg.inv(Sy)@K+np.linalg.inv(Sa))
        self.Sa = Sa
        self.G = G
        self.A = A
        self.K = K
        self.Shat = Shat
        self.Sm = G@Sy@G.T
        params.update_matrices(matrix_name='posterior_error_matrix',matrix=Shat)
        params.update_matrices(matrix_name='measurement_error_matrix',matrix=self.Sm)
        params.update_vectors(vector_name='posterior_error',vector=np.sqrt(np.diag(Shat)))
        params.update_matrices(matrix_name='averaging_kernel',matrix=A)
        params.update_vectors(vector_name='dofs_vector',vector=np.diag(A))
        gas_names = self.param_names[
            np.isin(
                self.param_names,list(
                    set().union(*[band.gas_names for band in self.bands]))
                )
            ]
        params.get_column_metrics(gas_names, self.h)
        params.get_interference_errors(A, self.h)
        self.params = params

class CrIS(dict):
    def __init__(self,l1_filename,l2_keys,
                 l2_filenames=None,
                 l2_path_pattern=None):
        self.logger = logging.getLogger(__name__)
        self.nc1 = Dataset(l1_filename,'r')
        self.l2_keys = l2_keys
        if l2_filenames is None:
            l2_filenames = [l2_path_pattern.replace('*',key) 
                            for key in l2_keys]
        self.nc2s = [Dataset(l2_filename,'r') for l2_filename in l2_filenames]
    
    def find_pixels(self, west, south, east, north, 
                    min_dofs=1, quality=1, land_flag=1):
        granule_number = self.nc1.granule_number
        self.logger.info(f'Level 1 granule number is {granule_number}')
        nc2 = self.nc2s[0]
        l2_mask = nc2['Geolocation/CrIS_Granule'][:] == granule_number
        self.logger.info(f'There are {np.sum(l2_mask)} level 2 pixels in the granule')
        
        lon = nc2['Longitude'][:]
        lat = nc2['Latitude'][:]
        
        lon_mask = (lon >= west) & (lon < east)
        lat_mask = (lat >= south) & (lat < north)
        
        l2_mask = l2_mask & lat_mask & lon_mask
        
        for nc2 in self.nc2s:
            l2_mask = l2_mask & \
                (nc2['DOFs'][:] >= min_dofs) & \
                    (nc2['Quality'][:] == quality) &\
                        (nc2['LandFlag'][:] == land_flag)
        self.logger.info(f'There are {np.sum(l2_mask)} level 2 pixels with further filtering')
        
        atrack = []
        xtrack = []
        fov = []
        
        atracks = self.nc2s[0]['Geolocation/CrIS_Atrack_Index'][l2_mask]
        xtracks = self.nc2s[0]['Geolocation/CrIS_Xtrack_Index'][l2_mask]
        fovs = self.nc2s[0]['Geolocation/CrIS_Pixel_Index'][l2_mask]
        
        for ipixel in range(np.sum(l2_mask)):
            pixel_l2_atrack = atracks[ipixel] + 1
            pixel_l2_xtrack = xtracks[ipixel] + 1
            pixel_l2_fov = fovs[ipixel] + 1
            
            atrack.append(pixel_l2_atrack)
            xtrack.append(pixel_l2_xtrack)
            fov.append(pixel_l2_fov)
        
        pixel_df = pd.DataFrame({
            'atrack': atrack,
            'xtrack': xtrack,
            'fov': fov
        })
        
        return pixel_df
        
    def sample_profile(self,sample_pressure,l2_keys):
        '''sample CrIS profile to a new pressure grid
        sample_pressure:
            new pressure grid, has to be strictly descending: sfc -> toa
        l2_keys:
            CrIS profile names, N2O, CH4, H2O, TATM
        '''
        x0s = self['Pressure']
        x1s = sample_pressure
        self['sampled_Pressure'] = x1s
        # x0s and x1s have to be strictly descending: pressure sfc -> toa
        gamma_matrix = np.zeros((len(x1s),len(x0s)))
        for ix1,x1 in enumerate(x1s):
            if x1 >= x0s[0]:
                gamma_matrix[ix1,0] = 1.
                continue
            if x1 <= x0s[-1]:
                gamma_matrix[ix1,-1] = 1.
                continue
            nearest_index = np.argmin(np.abs(x0s-x1))
            nearest_x0 = x0s[nearest_index]
            if nearest_x0 >= x1:
                next_index = nearest_index + 1
            else:
                next_index = nearest_index - 1
            next_x0 = x0s[next_index]
            nearest_weight = 1/np.abs(x1-nearest_x0)
            next_weight = 1/np.abs(x1-next_x0)
            gamma_matrix[ix1,nearest_index] = nearest_weight/(nearest_weight+next_weight)
            gamma_matrix[ix1,next_index] = next_weight/(nearest_weight+next_weight)
        self['gamma_matrix'] = gamma_matrix
        for key in l2_keys:
            self['sampled_{}'.format(key)] = gamma_matrix@self[key]
            self['sampled_{}_PriorCovariance'.format(key)] = \
                gamma_matrix@self[key+'_PriorCovariance']@gamma_matrix.T
            self['sampled_{}_PriorError'.format(key)] = \
                np.sqrt(
                    np.diag(
                        self['sampled_{}_PriorCovariance'.format(key)]))
            self['sampled_{}_PriorCorr'.format(key)] = convert_cov_cor(
                cov=self['sampled_{}_PriorCovariance'.format(key)])
    
    def save_splat_profile_like_GOSAT(self,pysplat_dir,
                                      file_dir=None,l2_keys=None,file_path=None):
        '''sample the cris posterior profiles into 20 level/19 layers, defined
        by GOSAT ap/bp/cp. then save these profiles in splat profile nc format
        '''
        sys.path.append(pysplat_dir)
        from apriori import profile as Profile
        if file_path is None:
            file_path = os.path.join(file_dir,'CrIS_{}.nc'.format(self.fov_obs_id)
                )
        if l2_keys is None:
            l2_keys=['N2O','CH4','H2O','TATM']
        
        # cris pixel surface pressure
        cris_psurf = self['Pressure'][0]
        # calculate GOSAT pressure level
        # assuming constant tropopause pressure for simplicity
        cris_ptrop = 200.
        # 20-level pressure
        P_level = GOSAT_AP*(cris_psurf-cris_ptrop) + GOSAT_BP*cris_ptrop + GOSAT_CP
        # 19-layer pressure
        P_layer = np.nanmean(np.column_stack((P_level[:-1],P_level[1:])),1)
        self['sampled_plevel'] = P_level
        self['sampled_player'] = P_layer
        gamma_matrix_plevel = get_Gamma(x0s=self['Pressure'],x1s=P_level)
        self['sampled_Tlevel'] = gamma_matrix_plevel@self['TATM']
        gamma_matrix = get_Gamma(x0s=self['Pressure'],x1s=P_layer)
        # self['sampled_TATM'] = gamma_matrix@self['TATM']
        for l2_key in l2_keys:
            self['sampled_{}'.format(l2_key)] = gamma_matrix@self[l2_key]
            self['sampled_{}_PriorCovariance'.format(l2_key)] = \
                gamma_matrix@self[l2_key+'_PriorCovariance']@gamma_matrix.T
            self['sampled_{}_PriorError'.format(l2_key)] = \
                np.sqrt(
                    np.diag(
                        self['sampled_{}_PriorCovariance'.format(l2_key)]))
            self['sampled_{}_PriorCorr'.format(l2_key)] = convert_cov_cor(
                cov=self['sampled_{}_PriorCovariance'.format(l2_key)])
        
        # translate to ccm's variable name, pedge_us is toa->sfc
        pedge_us = self['sampled_plevel'][::-1]
        Tedge_us = self['sampled_Tlevel'][::-1]
        # Make up a geolocation grid
        imx = 1
        jmx = 1
        lmx = pedge_us.shape[0]-1
        tmx = 1

        # Vertical grid
        H = R*Tedge_us/9.81/28.97e-3
        zedge = np.zeros_like(Tedge_us)
        for l in range(lmx-1,-1,-1):
            zedge[l] = zedge[l+1] + H[l+1]*1e-3*np.log(pedge_us[l+1]/pedge_us[l])
        # zmid = 0.5*(zedge[1:]+zedge[0:-1])
        xedge = np.linspace(-180.0,180.0,imx+1)
        yedge = np.linspace(-90.0,90.0,jmx+1)
        xmid = 0.5*(xedge[0:imx]+xedge[1:imx+1])
        ymid = 0.5*(yedge[0:jmx]+yedge[1:jmx+1])
        time = np.zeros(tmx)
        pedge = np.zeros((imx,jmx,lmx+1,tmx))
        Tedge = np.zeros((imx,jmx,lmx+1,tmx))
        zsurf = np.zeros((imx,jmx))
        for l in range(lmx+1):
            pedge[:,:,l,:] = pedge_us[l]
            Tedge[:,:,l,:] = Tedge_us[l]
        # Initialize output profile
        prof = Profile(file_path,xmid,ymid,time,xedge,yedge,pedge,Tedge,zsurf)

        # Add Temperature error covariance
        Tmid_err = np.ones((1,1,lmx,1))*0.01
        prof.add_T_prior(Tmid_err,ErrorType='DIAGONAL')

        # Temperature shift
        Tshift_err = np.ones((1,1,1))*4.0
        prof.add_Tshift_prior(Tshift_err)

        # Add surface pressure error
        psurf_err = np.ones((1,1,1))*4.0
        prof.add_psurf_prior(psurf_err)

        prof.add_profile_var('N2', 0.78084*np.ones((1,1,lmx,1)))
        prof.add_profile_var('O2', 0.20946*np.ones((1,1,lmx,1)))
        prof.add_profile_var('Ar', 0.00934*np.ones((1,1,lmx,1)))
        prof.add_profile_var('CO2', 0.0004*np.ones((1,1,lmx,1)))
        prof.add_diagonal_prof_prior('CO2', Sdiag=np.ones((imx,jmx,lmx,1)))
        # Add some zeroed species as a kludge for now
        var = np.zeros((1,1,lmx,1))
        for subgas in ['CO','OCS','O3','NO2','NO','HNO3','O2DG','PA1',
                       'HCHO','SO2','BrO','IO','GLYX']:
            prof.add_profile_var(subgas,var)
            prof.add_diagonal_prof_prior(name=subgas, Sdiag=np.ones((imx,jmx,lmx,1)))


        for gas_name in l2_keys:
            if gas_name == 'TATM':
                continue
            vmr = np.zeros((imx,jmx,lmx,1))
            for l in range(lmx):
                vmr[:,:,l,0] = self['sampled_{}'.format(gas_name)][lmx-1-l]
            prof.add_profile_var(name=gas_name, var=vmr)
            prof.add_diagonal_prof_prior(name=gas_name, Sdiag=np.ones((imx,jmx,lmx,1)))

        # SULFATE
        aod = np.ones((imx,jmx,1))*2.0
        aod_err = np.ones((imx,jmx,1))*0.2*2.0
        zmin = np.ones((imx,jmx,1))*2.0
        zmax = np.ones((imx,jmx,1))*6.0

        # Alpha
        alpha = np.ones((imx,jmx,1,1))*3.0
        alpha_err = np.ones((1,1,1))*1.0 #Scol_diag_kludge(pedge,alpha,3.0,is_mixratio=False)

        prof.add_box_aerosol('SU',aod,zmin,zmax)
        prof.add_box_prior('SU', aod_err)

        # Add a test aerosol optical property parameter
        prof.add_aeroptprop_var('SU_alpha',alpha,shift_err=alpha_err)
        # prof.add_diagonal_prof_prior('SU_alpha',alpha_err)

        # Make OC,BC,SF,SC the same
        prof.add_box_aerosol('OC',aod,zmin,zmax)
        prof.add_box_prior('OC', aod_err)
        prof.add_box_aerosol('BC',aod,zmin,zmax)
        prof.add_box_prior('BC', aod_err)
        prof.add_box_aerosol('SF',aod,zmin,zmax)
        prof.add_box_prior('SF', aod_err)
        prof.add_box_aerosol('SC',aod,zmin,zmax)
        prof.add_box_prior('SC', aod_err)

        # DUST
        aod = np.ones((imx,jmx,1))*1.0
        aod_err = np.ones((imx,jmx,1))*10.0
        pkhght = np.ones((imx,jmx,1))*1.0
        pkhght_err = np.ones((imx,jmx,1))*10.0
        pkwdth = np.ones((imx,jmx,1))*0.5
        pkwdth_err = np.ones((imx,jmx,1))*10.0

        zmin = np.ones((imx,jmx,1))*0.0
        zmax = np.ones((imx,jmx,1))*50.0

        prof.add_gdf_aerosol('DU', aod, zmin, zmax, pkhght, pkwdth)
        prof.add_gdf_prior('DU', aod_err, pkhght_err, pkwdth_err)

        prof.close()
        
        return file_path
        
    def update_splat_profile_nc(self,infile,l2_keys,outfile=None):
        '''update the splat profile netcdf file using cris profiles
        infile:
            path to the template splat profile nc
        l2_keys:
            name of cris profiles to sample to splat pressure grid. 
            TATM will be sampled to pedge, others pmid.
            these will replace original profiles
        '''
        if outfile is None:
            outfile = os.path.join(
                os.path.split(infile)[0],'CrIS_{}.nc'.format(self.fov_obs_id)
                )
        os.system(f'cp {infile} {outfile}')
        with Dataset(outfile,'r+') as nc:
            P_level = nc['pedge'][:].squeeze().filled(np.nan)[::-1]
            P_layer = np.nanmean(np.column_stack((P_level[:-1],P_level[1:])),1)
            for l2_key in l2_keys:
                if l2_key in ['TATM']:
                    self.sample_profile(sample_pressure=P_level, 
                                        l2_keys=['TATM'])
                    nc['Tedge'][:] = self['sampled_TATM'][np.newaxis,
                                                          ::-1,
                                                          np.newaxis,
                                                          np.newaxis]
                else:
                    self.sample_profile(sample_pressure=P_layer, 
                                        l2_keys=[l2_key])
                    nc[l2_key][:] = self[f'sampled_{l2_key}'][np.newaxis,
                                                          ::-1,
                                                          np.newaxis,
                                                          np.newaxis]
        return outfile
    
    def load_pixel(self,atrack_1based=27,
                   xtrack_1based=15,fov_1based=7,granule_number=None):
        
        self.atrack_1based = atrack_1based
        self.xtrack_1based = xtrack_1based
        self.fov_1based = fov_1based
        self.granule_number = granule_number or \
            int(self.nc1.product_name_granule_number[1:])
        pixel_l1_id = '{}.{:02d}E{:02d}.{:01d}'.format(self.nc1.gran_id,
                                                      atrack_1based,
                                                      xtrack_1based,
                                                      fov_1based)
        self.fov_obs_id = pixel_l1_id
        l1_mask = self.nc1['fov_obs_id'][:] == pixel_l1_id
        if np.sum(l1_mask) != 1:
            self.logger.error('{} l1 pixel found'.format(np.sum(l1_mask)))
            return
        wnum_mw = self.nc1['wnum_mw'][:].data
        wvl_mw = 1e7/wnum_mw
        rad_mw = self.nc1['rad_mw'][:][l1_mask][0,:].data*\
            wnum_mw/wvl_mw*1e-3/(PLANCK_CONSTANT*LIGHT_SPEED/wvl_mw*1e9)*1e-4
        wnum_sw = self.nc1['wnum_sw'][:].data
        wvl_sw = 1e7/wnum_sw
        rad_sw = self.nc1['rad_sw'][:][l1_mask][0,:].data*\
            wnum_sw/wvl_sw*1e-3/(PLANCK_CONSTANT*LIGHT_SPEED/wvl_sw*1e9)*1e-4
        self['wnum_mw'] = wnum_mw
        self['wvl_mw'] = wvl_mw
        self['rad_mw'] = rad_mw
        self['wnum_sw'] = wnum_sw
        self['wvl_sw'] = wvl_sw
        self['rad_sw'] = rad_sw
        l2_mask = (self.nc2s[0]['Geolocation/CrIS_Granule'][:] == self.granule_number)&\
                  (self.nc2s[0]['Geolocation/CrIS_Atrack_Index'][:] == atrack_1based-1)&\
                  (self.nc2s[0]['Geolocation/CrIS_Xtrack_Index'][:] == xtrack_1based-1)&\
                  (self.nc2s[0]['Geolocation/CrIS_Pixel_Index'][:] == fov_1based-1)
        if np.sum(l2_mask) != 1:
            self.logger.error('{} l2 pixel found'.format(np.sum(l2_mask)))
            return
        pressure = self.nc2s[0]['Pressure'][l2_mask,:][0,:].data
        mask = pressure > 0# remove -999
        self['Pressure'] = pressure[mask]
        
        for l2_key, nc2 in zip(self.l2_keys,self.nc2s):
            self[l2_key] = nc2['Species'][l2_mask,:][0,:][mask].data
            self[l2_key+'_Prior'] = nc2[
                'ConstraintVector'][l2_mask,:][0,:][mask].data
            self[l2_key+'_PriorCovariance'] = nc2[
                'Characterization/PriorCovariance'][l2_mask,:,:][0,::][
                np.ix_(mask,mask)].data
            self[l2_key+'_ObservationErrorCovariance'] = nc2[
                'ObservationErrorCovariance'][l2_mask,:,:][0,::][
                np.ix_(mask,mask)].data
            self[l2_key+'_TotalErrorCovariance'] = nc2[
                'Characterization/TotalErrorCovariance'][l2_mask,:,:][0,::][
                np.ix_(mask,mask)].data
        
            for cov_key in ['PriorCovariance',
                            'ObservationErrorCovariance',
                            'TotalErrorCovariance']:
                fld_key = l2_key+'_'+cov_key
                self[fld_key.replace('Covariance','Corr')] = \
                    convert_cov_cor(cov=self[fld_key])
        
        self['Ts'] = self.nc2s[0]['Retrieval/SurfaceTemperature'][l2_mask].data
        self['Ts_Prior'] = self.nc2s[0][
            'Characterization/SurfaceTempConstraint'][l2_mask].data
        self['Ts_ObservationError'] = self.nc2s[0][
            'Characterization/SurfaceTempObservationError'][l2_mask].data
        self['emissivity'] = self.nc2s[0][
            'Characterization/Emissivity'][l2_mask,:][0,:].data
        self['wnum_emissivity'] = self.nc2s[0][
            'Characterization/Emissivity_Wavenumber'][l2_mask,:][0,:].data
        self['wvl_emissivity'] = 1e7/self['wnum_emissivity']
        l2_latlon = np.array([self.nc2s[0]['Longitude'][l2_mask],self.nc2s[0]['Latitude'][l2_mask]])
        l1_latlon = np.array([self.nc1['lon'][:][l1_mask],self.nc1['lat'][:][l1_mask]])
        np.testing.assert_array_equal(l1_latlon, l2_latlon)
        self['lon'] = self.nc2s[0]['Longitude'][l2_mask].data
        self['lat'] = self.nc2s[0]['Latitude'][l2_mask].data
        
    def close_nc(self):
        self.nc1.close()
        for nc2 in self.nc2s:
            nc2.close()