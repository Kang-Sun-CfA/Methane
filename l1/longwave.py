#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:42:34 2024

@author: kangsun
"""

from netCDF4 import Dataset
import numpy as np
from matplotlib import pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
from scipy.interpolate import RegularGridInterpolator, interp1d
from astropy.convolution import convolve_fft
import sys,os,glob
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

def BB(T_K,w_nm,radiance_unit='photons/s/cm2/sr/nm',
       c=2.99792458e8,h=6.62607004e-34,kB=1.38064852e-23):
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
    if unit=='W/m2/sr/nm':
        r = 2*h*np.power(c,2)*np.power(w,-5)/(np.exp(h*c/w/kB/T)-1)*1e-9
    elif unit=='photons/s/cm2/sr/nm':
        r = 2*c*np.power(w,-4)/(np.exp(h*c/w/kB/T)-1)*1e-13
    elif unit=='photons/s/cm2/sr/cm-1':
        r = 2*c*np.power(w,-4)/(np.exp(h*c/w/kB/T)-1)*1e-13*np.power(w_nm,2)*1e-7
    return r/factor

def F_interp_absco(absco_P,absco_T,absco_B,absco_w,absco_sigma,
                   Pq,Tq,Bq,wq,T_ext=15):
    ''' 
    absco_* should be the same format as in the absco table saved by splat
    P should be in Pa; T in K; B in volume/volume
    T_ext should be larger than the temperature grid size
    '''
    local_P_id = np.argmin(np.abs(absco_P-Pq))
    if local_P_id == 0:
        P_mask = (absco_P<=absco_P[local_P_id+1])
    elif local_P_id == len(absco_P)-1:
        P_mask = (absco_P>=absco_P[local_P_id-1])
    else:
        P_mask = (absco_P>=absco_P[local_P_id-1])&(absco_P<=absco_P[local_P_id+1])
    aP = absco_P[P_mask]
    aT = absco_T[P_mask,]
    asigma = absco_sigma[P_mask,]
    T_mask = np.zeros(aT.shape,dtype=bool)
    for ilayer,iT in enumerate(aT):
        T_mask[ilayer,] = (iT >= Tq-T_ext) & (iT <= Tq+T_ext)
    T_grid = aT[0,T_mask[0,]]
    for ilayer in range(len(aP)):
        if not np.array_equal(T_grid,aT[ilayer,T_mask[ilayer,]]):
            logging.error('unique temperature grid is not regular!')
            return
    sigma_grid = np.zeros((len(aP),len(T_grid),*asigma.shape[2:]))
    for ilayer in range(len(aP)):
        sigma_grid[ilayer,...] = asigma[ilayer,T_mask[ilayer,],...]
    func = RegularGridInterpolator((aP,T_grid,absco_B,absco_w), sigma_grid)
    return func((Pq,Tq,Bq,wq))

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
    
class Longwave(object):
    def __init__(self,start_w,end_w,gas_names,
                 vza=0.,Ts=None,TC=0.,emissivity=1.,
                 dw=0.1,nsample=3,hw1e=None,
                 absco_path_pattern='/home/kangsun/N2O/n2o_run/data/splat_data/SAO_crosssections/splatv2_xsect/HITRAN2020_*_4500-4650nm_0p00_0p002dw.nc',
                 profile_path='/home/kangsun/N2O/n2o_run/data/additional_inputs/test_profile.nc',
                 radiance_unit='1e14 photons/s/cm2/sr/nm'):
        '''
        start/end_w:
            start/end wavelength of the sensor in nm
        gas_names:
            list of gas names, e.g.,['N2O']
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
        self.logger = logging.getLogger(__name__)
        self.start_w = start_w
        self.end_w = end_w
        self.gas_names = gas_names
        self.radiance_unit = radiance_unit
        hw1e = hw1e or dw*nsample/1.665109
        self.dw = dw
        self.hw1e = hw1e
        self.nsample = nsample
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
        self.abscos = abscos
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
        
        # somehow there is no alitutde in the test_profile.nc file
        profiles['z_level'] = np.array([80.0,60.0,55.0,50.0,47.5,45.0,42.5,40.0,\
                               37.5,35.0,32.5,30.0,27.5,25.0,24.0,23.0,\
                               22.0,21.0,20.0,19.0,18.0,17.0,16.0,15.0,\
                               14.0,13.0,12.0,11.0,10.0,9.0,8.0,7.0,6.0,\
                               5.0,4.0,3.0,2.0,1.0,0.0])[::-1]
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
            if 'T_' in key:
                finite_difference = True
            else:
                finite_difference = False
            # convert wavelength,layer to layer,wavelength
            jac = self.get_profile_jacobian(key,finite_difference,**kwargs).T
            
            if if_convolve:
                jacs[key] = np.zeros((jac.shape[0],len(w2)))
                for i,ja in enumerate(jac):
                    f = interp1d(self.w1, convolve_fft(ja,ILS,normalize_kernel=True),
                                 bounds_error=False,fill_value='extrapolate')
                    jacs[key][i,] = f(w2)
            else:
                jacs[key] = jac
        
        if not if_convolve:
            jacs['Radiance'] = self.get_radiance()
        else:
            f = interp1d(self.w1, convolve_fft(self.get_radiance(),ILS,normalize_kernel=True),
                         bounds_error=False,fill_value='extrapolate')
            jacs['Radiance'] = f(w2)
        self.jacs = jacs
    
    def get_profile_jacobian(self,key,finite_difference=True,fd_delta=None,
                             fd_relative_delta=None,
                             wrt='relative mixing ratio'):
        ''' 
        key:
            a key name in self.profile
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
            if key in ['T_level','T_layer']:
                self.logger.error('not implemented')
                return
            self.get_sigmas()
            jac = np.full((len(self.w1),self.nlayer),np.nan)
            Ts = self.Ts
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
        return jac                
        
    def get_sigmas(self):
        sigmas = {}
        for gas in self.gas_names:
            sigmas[gas] = np.zeros((self.nlayer,len(self.w1)))
        dtau_layer = np.zeros((self.nlayer,len(self.w1)))
        for ilayer in range(self.nlayer):
            P = self.profiles['P_layer'][ilayer]*100.# hPa to Pa
            T = self.profiles['T_layer'][ilayer]
            B = self.profiles['H2O'][ilayer]
            dz = self.profiles['dz'][ilayer]*1e5 # km to cm
            # air density in molec/cm3
            air_density = self.profiles['air_density'][ilayer]
            for gas in self.gas_names:
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
        self.sigmas = sigmas
        self.dtau_layer = dtau_layer
    
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
    def __init__(self,start_w,end_w,gas_names,sza=30.,vza=0.,
                 dw=0.1,nsample=3,hw1e=None,
                 splat_path='/home/kangsun/N2O/sci-level2-splat/build/splat.exe',
                 control_template_path='/home/kangsun/N2O/n2o_run/control/forward_template.control',
                 working_dir='/home/kangsun/N2O/n2o_run',
                 profile_path='/home/kangsun/N2O/n2o_run/data/additional_inputs/test_profile.nc'):
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
        self.gas_names = gas_names
        self.splat_path = splat_path
        self.working_dir = working_dir
        self.profile_path = profile_path
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
        
        # somehow there is no alitutde in the test_profile.nc file
        profiles['z_level'] = np.array([80.0,60.0,55.0,50.0,47.5,45.0,42.5,40.0,\
                               37.5,35.0,32.5,30.0,27.5,25.0,24.0,23.0,\
                               22.0,21.0,20.0,19.0,18.0,17.0,16.0,15.0,\
                               14.0,13.0,12.0,11.0,10.0,9.0,8.0,7.0,6.0,\
                               5.0,4.0,3.0,2.0,1.0,0.0])[::-1]
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
                
    def get_jacobians(self,keys=None,keynames=None,finite_difference=False,delete_nc=False,
                      if_convolve=True,
                     **kwargs):
        jacs = {}
        keys = keys or ['RTM_Band1/Radiance_I']+\
            [f'RTM_Band1/{g.upper()}_TraceGasJacobian_I' for g in self.gas_names]
        keynames = keynames or [k.split('/')[-1] for k in keys]
        
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
                        jacs[kn] = jac
            jacs['Wavelength'] = w2
            self.jacs = jacs
            if delete_nc:
                os.remove(self.output_path)
        
    
    def run_splat(self,control_path=None,control_template_path=None,
                  output_path=None,profile_path=None,rerun_splat=True,
                  **kwargs):
        profile_path = profile_path or self.profile_path
        output_path = output_path or os.path.join(self.working_dir,'splat_output.nc')
        control_path = control_path or os.path.join(self.working_dir,'splat_run.control')
        control_template_path = control_template_path or \
            os.path.join(self.working_dir,'control/forward_template.control')
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
            return
        with open(control_template_path,'r') as template_f:
            ctrl = template_f.read()
            for v in varlst.keys():
                ctrl = ctrl.replace('{{'+v+'}}',str(varlst[v]))
            with open(control_path,'w') as f:
                f.write(ctrl)
        
        self.run(control_path)
    
    def run(self,control_path):
        cwd = os.getcwd()
        os.chdir(self.working_dir)
        os.system(f'{self.splat_path} {control_path}')
        os.chdir(cwd)