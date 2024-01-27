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
from scipy.interpolate import RegularGridInterpolator
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
    w = w_nm*1e-9# wavelength in m
    T = T_K
    if radiance_unit=='W/m2/sr/nm':
        return 2*h*np.power(c,2)*np.power(w,-5)/(np.exp(h*c/w/kB/T)-1)*1e-9
    elif radiance_unit=='photons/s/cm2/sr/nm':
        return 2*c*np.power(w,-4)/(np.exp(h*c/w/kB/T)-1)*1e-13
    elif radiance_unit=='photons/s/cm2/sr/cm-1':
        return 2*c*np.power(w,-4)/(np.exp(h*c/w/kB/T)-1)*1e-13*np.power(w_nm,2)*1e-7

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

class Longwave_Emitter(object):
    def __init__(self,start_w,end_w,gas_names,
                 absco_path_pattern='/home/kangsun/N2O/n2o_run/data/splat_data/SAO_crosssections/splatv2_xsect/HITRAN2020_*_4500-4650nm_0p00_0p002dw.nc',
                 profile_path='/home/kangsun/N2O/n2o_run/data/additional_inputs/test_profile.nc',
                 radiance_unit='photons/s/cm2/sr/nm'):
        self.logger = logging.getLogger(__name__)
        self.start_w = start_w
        self.end_w = end_w
        self.gas_names = gas_names
        self.radiance_unit = radiance_unit
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
            else:
                if not np.array_equal(self.w1, absco_w):
                    self.logger.warning(f'wavelength grid of {gas} differs from {gas_names[0]}')
        self.abscos = abscos
        profiles = {}
        self.logger.info(f'loading {profile_path}')
        with Dataset(profile_path,'r') as nc:
            profiles['P_level'] = nc['pedge'][:].squeeze().filled(np.nan)
            profiles['T_level'] = nc['Tedge'][:].squeeze().filled(np.nan)
            for gas in gas_names:
                profiles[gas] = nc[gas][:].squeeze().filled(np.nan)
            if 'H2O' not in gas_names:
                profiles['H2O'] = nc['H2O'][:].squeeze().filled(np.nan)
        profiles['P_layer'] = np.nanmean(
            np.column_stack(
                (profiles['P_level'][:-1],profiles['P_level'][1:])),1)
        profiles['T_layer'] = np.nanmean(
            np.column_stack(
                (profiles['T_level'][:-1],profiles['T_level'][1:])),1)
        # somehow there is no alitutde in the test_profile.nc file
        profiles['z_level'] = np.array([80.0,60.0,55.0,50.0,47.5,45.0,42.5,40.0,\
                               37.5,35.0,32.5,30.0,27.5,25.0,24.0,23.0,\
                               22.0,21.0,20.0,19.0,18.0,17.0,16.0,15.0,\
                               14.0,13.0,12.0,11.0,10.0,9.0,8.0,7.0,6.0,\
                               5.0,4.0,3.0,2.0,1.0,0.0])
        profiles['dz'] = np.abs(profiles['z_level'][:-1]-profiles['z_level'][1:])
        self.profiles = profiles
        self.nlevel = len(profiles['P_level'])
        self.nlayer = len(profiles['P_layer'])
    
    def get_profile_jacobian(self,key,finite_difference=True,fd_delta=None,
                             fd_relative_delta=None,get_radiance_kw=None):
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
        get_radiance_kw:
            keyword arguements to self.get_radiance
        '''
        if finite_difference:
            profile_c = self.profiles[key].copy()
            if fd_delta is None and fd_relative_delta is None:
                self.logger.warning('a relative fd_relative_delta of 0.001 is assumed')
                fd_relative_delta = 1e-3
            if fd_delta is None and fd_relative_delta is not None:
                fd_delta = np.ones_like(profile_c)*fd_relative_delta
            profile_l = profile_c-fd_delta/2.
            profile_r = profile_c+fd_delta/2.
            if get_radiance_kw is None:
                get_radiance_kw = dict(vza=0)
            jac = np.full((len(self.w1),self.nlayer),np.nan)
            for ilayer in range(self.nlayer):
                self.profiles[key][ilayer] = profile_l[ilayer]
                self.get_sigmas()
                r_l = self.get_radiance(**get_radiance_kw)
                self.profiles[key][ilayer] = profile_r[ilayer]
                self.get_sigmas()
                r_r = self.get_radiance(**get_radiance_kw)
                jac[:,ilayer] = (r_r-r_l)/fd_delta[ilayer]
            self.profiles[key] = profile_c.copy()
            self.get_sigmas()
        else:
            if key == 'T_layer':
                self.logger.error('not implemented')
                return
            
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
            air_density = P/T/1.38064852e-23*1e-6
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
            P = self.profiles['P_layer'][ilayer]*100.# hPa to Pa
            T = self.profiles['T_layer'][ilayer]
            B = self.profiles['H2O'][ilayer]
            dz = self.profiles['dz'][ilayer]*1e5 # km to cm
            # air density in molec/cm3
            air_density = P/T/1.38064852e-23*1e-6
            for gas in self.gas_names:
                # gas density in molec/cm3
                gas_density = air_density*self.profiles[gas][ilayer]
                # optical thickness in that layer
                dtau_layer[ilayer,] += self.sigmas[gas][ilayer,]*gas_density*dz
        self.dtau_layer = dtau_layer
    
    def get_radiance(self,vza,Ts=None,emissivity=None):
        '''
        note the profile dimensions go from surface to TOA in this function
        '''
        if Ts is None:
            Ts = self.profiles['T_level'][np.argmin(self.profiles['z_level'])]
        if emissivity is None:
            emissivity = 1.
        mu = np.cos(vza/180*np.pi)
        radiance_unit = self.radiance_unit
        # surface planck function
        Bs = BB(Ts,self.w1,radiance_unit)
        # planck function for layer and level temperatures, sfc->toa
        B_level = np.zeros((self.nlevel,len(self.w1)))
        B_layer = np.zeros((self.nlayer,len(self.w1)))
        for ilevel in range(self.nlevel):
            B_level[ilevel,] = BB(self.profiles['T_level'][ilevel],self.w1,radiance_unit)
        for ilayer in range(self.nlayer):
            B_layer[ilayer,] = BB(self.profiles['T_layer'][ilayer],self.w1,radiance_unit)
        B_level = B_level[::-1,]
        B_layer = B_layer[::-1,]
        # slant optical thickness, sfc->toa
        dtau_layer_mu = self.dtau_layer[::-1,]/mu
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
        radiance = Bs*np.exp(-np.nansum(dtau_layer_mu,axis=0))\
            +np.nansum((1-np.exp(-dtau_layer_mu))\
                       *B_tau\
                       *np.exp(tau_level_mu[1:,]-tau_level_mu[-1,])\
                       ,axis=0)
        self.B_level = B_level[::-1,]
        self.Bs = Bs
        return radiance