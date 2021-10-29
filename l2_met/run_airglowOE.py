# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 11:12:29 2021
script to run airglowOE retrieval. To run a retrieval:
    conda activate airglowOE;
    python run_airglowOE.py control.txt
where control.txt can be generated by F_generate_control in airglowOE.py.
airglowOE.py is the main library, maintained at 
https://github.com/Kang-Sun-CfA/Methane/blob/master/l2_met/airglowOE.py
@author: kangsun
"""

import sys, os, yaml
import numpy as np
if sys.platform == 'win32':# windows for testing
    with open(r'C:\research\CH4\airglowOE\airglowOE_control.txt','r') as stream:
        control = yaml.full_load(stream)
else:
    if len(sys.argv) == 1:
        control_path = 'airglowOE_control.txt'
    else:
        control_path = str(sys.argv[1])
    with open(control_path,'r') as stream:
        control = yaml.full_load(stream)

sys.path.append(control['hapi directory'])
sys.path.append(control['airglowOE directory'])
from hapi import db_begin, fetch
from airglowOE import sciaOrbit, F_fit_profile, Level2_Saver

import logging
if control['if verbose']:
    logging.basicConfig(level=logging.INFO)
if 'if use msis' not in control.keys():
    control['if use msis'] = True
if 'if save single-pixel file' not in control.keys():
    control['if save single-pixel file'] = False
# open a sciamachy orbit level 1b file
s = sciaOrbit(sciaPath=control['sciamachy file path'])
# decide if the orbit is mesosphere-lower thermosphere (mlt) limb scan (50-150 km) 
# or normal mesosphere-upper stratosphere (mus) limb scan (0-100 km)
# the tangent height order appears to be reversed in mlt scans
try:
    if_mlt_or_not = s.ifMLT()
except:
    logging.warning(control['sciamachy file path']+' gives error!')
    sys.exit()
#%%
def F_save_single_pixel(fn,iy,ix,ny,nx=8,nth=15,**kwargs):
    '''
    save data into flexible mat files pixel-wise
    fn:
        file name
    iy:
        along track index, 0-based
    ix:
        across track index, 0-based
    ny:
        along track index, 0-based
    nx:
        total across track indices, should be 8
    nth:
        total tangent heights, should be 15
    '''
    from scipy.io import savemat
    save_dict = {}
    for (k,v) in kwargs.items():
        if v.shape == (ny,nx,nth):
            data = v[iy,ix,:]
        elif v.shape == (ny,nth):
            data = v[iy,:]
        elif v.shape == (ny,nx):
            data = v[iy,ix]
        else:
            logging.info('the dimension of {} cannot be recognized, skipping'.format(k))
            continue
        logging.info('saving {}'.format(k))
        save_dict[k] = data
    savemat(fn,save_dict)
def F_x2(sza,sza1=70,sza2=95,minTH1=25,minTH2=40):
    if sza<=sza1:
        return minTH1
    if sza>=sza2:
        return minTH2
    return ((sza-sza1)/(sza2-sza1))**2*(minTH2-minTH1)+minTH1
#%%
if if_mlt_or_not:
    D_startWavelength = control['delta start wavelength']
    D_endWavelength = control['delta end wavelength']
    D_minTH = control['delta min tangent height mlt']
    D_maxTH = control['delta max tangent height mlt']
    D_w1_step = control['delta w1 step']
    D_n_nO2 = control['delta number of loosen O2 layers mlt']
    if control['if A band']:
        S_startWavelength = control['sigma start wavelength']
        S_endWavelength = control['sigma end wavelength']
        S_minTH = control['sigma min tangent height mlt']
        S_maxTH = control['sigma max tangent height mlt']
        S_w1_step = control['sigma w1 step']
        S_n_nO2 = control['sigma number of loosen O2 layers mlt']
else:
    D_startWavelength = control['delta start wavelength']
    D_endWavelength = control['delta end wavelength']
    D_minTH = control['delta min tangent height']
    D_maxTH = control['delta max tangent height']
    D_w1_step = control['delta w1 step']
    D_n_nO2 = control['delta number of loosen O2 layers']
    if control['if A band']:
        S_startWavelength = control['sigma start wavelength']
        S_endWavelength = control['sigma end wavelength']
        S_minTH = control['sigma min tangent height']
        S_maxTH = control['sigma max tangent height']
        S_w1_step = control['sigma w1 step']
        S_n_nO2 = control['sigma number of loosen O2 layers']

# load hitran database
db_begin(control['hitran database path'])
if not os.path.exists(os.path.join(control['hitran database path'],
                                   'O2_{:.1f}-{:.1f}.data'.format(D_startWavelength,D_endWavelength))):
    fetch('O2_{:.1f}-{:.1f}'.format(D_startWavelength,D_endWavelength),7,1,
          1e7/D_endWavelength,1e7/D_startWavelength)
if control['if A band']:
    if not os.path.exists(os.path.join(control['hitran database path'],
                                   'O2_{:.1f}-{:.1f}.data'.format(S_startWavelength,S_endWavelength))):
        fetch('O2_{:.1f}-{:.1f}'.format(S_startWavelength,S_endWavelength),7,1,
          1e7/S_endWavelength,1e7/S_startWavelength)

# load singlet delta band data
s.loadData(if_close_file=False,startWavelength=1200,endWavelength=1340)
# parse data to regular along-track/across track grids, usually 30 along-track, 8 across-track
D_granules = s.divideProfiles(radiancePerElectron=control['delta radiance per electron'])
D_ngranule = len(D_granules)
D_nft = D_granules[0]['tangent_height'].shape[1]
D_nth = D_granules[0]['tangent_height'].shape[0]
if control['if A band']: 
    # load singlet sigma band data
    s.loadData(if_close_file=True,startWavelength=750,endWavelength=780)
    S_granules = s.divideProfiles(radiancePerElectron=control['sigma radiance per electron'])
    S_ngranule = len(S_granules)
    S_nft = S_granules[0]['tangent_height'].shape[1]
    S_nth = S_granules[0]['tangent_height'].shape[0]
    # the shape and geometry should be exactly the same for singlet delta and sigma bands
    if S_ngranule != D_ngranule:
        sys.exit('along track dimensions do not match between sigma and delta bands!')
    if S_nft != D_nft:
        sys.exit('across track dimensions do not match between sigma and delta bands!')
    if S_nth != D_nth:
        sys.exit('vertical dimensions do not match between sigma and delta bands!')
else:
    s.nc.close()
#%%
import time
# set aside memory for data arrays
D_nO2s = np.full((D_ngranule,D_nft,D_nth),np.nan,dtype=np.float32)
D_T = np.full((D_ngranule,D_nft,D_nth),np.nan,dtype=np.float32)
D_nO2s_dofs = np.full((D_ngranule,D_nft,D_nth),np.nan,dtype=np.float32)
D_T_dofs = np.full((D_ngranule,D_nft,D_nth),np.nan,dtype=np.float32)
D_nO2s_e = np.full((D_ngranule,D_nft,D_nth),np.nan,dtype=np.float32)
D_T_msis = np.full((D_ngranule,D_nft,D_nth),np.nan,dtype=np.float32)
D_nO2_msis = np.full((D_ngranule,D_nft,D_nth),np.nan,dtype=np.float32)
if D_n_nO2 > 0:
    D_nO2Scale = np.full((D_ngranule,D_nft,D_nth),np.nan,dtype=np.float32)
    D_nO2Scale_dofs = np.full((D_ngranule,D_nft,D_nth),np.nan,dtype=np.float32)
    D_nO2Scale_e = np.full((D_ngranule,D_nft,D_nth),np.nan,dtype=np.float32)
D_T_e = np.full((D_ngranule,D_nft,D_nth),np.nan,dtype=np.float32)
D_HW1E = np.full((D_ngranule,D_nft),np.nan,dtype=np.float32)
D_HW1E_dofs = np.full((D_ngranule,D_nft),np.nan,dtype=np.float32)
D_HW1E_e = np.full((D_ngranule,D_nft),np.nan,dtype=np.float32)
D_w_shift = np.full((D_ngranule,D_nft),np.nan,dtype=np.float32)
D_w_shift_dofs = np.full((D_ngranule,D_nft),np.nan,dtype=np.float32)
D_w_shift_e = np.full((D_ngranule,D_nft),np.nan,dtype=np.float32)
D_chi2 = np.full((D_ngranule,D_nft),np.nan,dtype=np.float32)
D_rmse = np.full((D_ngranule,D_nft),np.nan,dtype=np.float32)
D_if_success = np.zeros((D_ngranule,D_nft),dtype=np.int8)
D_Jprior = np.full((D_ngranule,D_nft),np.nan,dtype=np.float32)
D_niter = np.full((D_ngranule,D_nft),np.nan,dtype=np.int8)
D_tangent_height = np.full((D_ngranule,D_nft,D_nth),np.nan,dtype=np.float32)
D_solar_zenith_angle = np.full((D_ngranule,D_nft,D_nth),np.nan,dtype=np.float32)
D_time = np.full((D_ngranule,D_nth),np.nan,dtype=np.float64)
D_dZ = np.full((D_ngranule,D_nft,D_nth),np.nan,dtype=np.float32)
D_latitude = np.full((D_ngranule,D_nft,D_nth),np.nan,dtype=np.float32)
D_longitude = np.full((D_ngranule,D_nft,D_nth),np.nan,dtype=np.float32)

for (igranule,granule) in enumerate(D_granules):
    if granule['tangent_height'].shape[0] != D_nth:
        continue
    for ift in range(D_nft):
        th_idx = np.argsort(granule['tangent_height'][:,ift])# TH has to go from low to high
        D_latitude[igranule,ift,] = granule['latitude'][th_idx,ift]
        D_longitude[igranule,ift,] = granule['longitude'][th_idx,ift]
        D_tangent_height[igranule,ift,] = granule['tangent_height'][th_idx,ift]
    D_solar_zenith_angle[igranule,] = granule['solar_zenith_angle'].T
    D_time[igranule,:] = granule['time']
#%%
# loop through along- and across-track
for (igranule,granule) in enumerate(D_granules):
    if granule['tangent_height'].shape[0] != D_nth:
        continue
    if igranule < control['delta start along-track (0-based)'] or igranule > control['delta end along-track (0-based)']:
        continue
    for ift in range(D_nft):
        if ift < control['delta start across-track (0-based)'] or ift > control['delta end across-track (0-based)']:
            continue
        logging.warning('delta granule {}, footprint {}'.format(igranule,ift))
        try:
            # a series of tangent height radiance spectra are concatenated and fit together
            Time = time.time()
            if if_mlt_or_not:
                pixel_minTH = D_minTH
            else:
                pixel_sza=np.nanmean(granule['solar_zenith_angle'][:,ift])
                pixel_minTH = F_x2(pixel_sza,sza1=70,sza2=95,minTH1=D_minTH,minTH2=40)
                if pixel_minTH != D_minTH:
                    logging.warning('min TH is changed from {} to {:.2f} km at sza of {:.2f}'.format(D_minTH,pixel_minTH,pixel_sza))
            result = F_fit_profile(tangent_height=granule['tangent_height'][:,ift],
                                   radiance=granule['radiance'][:,ift,:].squeeze(),
                                   radiance_error=granule['radiance_error'][:,ift,:].squeeze(),
                                   wavelength=granule['wavelength'],
                                   startWavelength=D_startWavelength,
                                   endWavelength=D_endWavelength,
                                   minTH=pixel_minTH,maxTH=D_maxTH,w1_step=D_w1_step,
                                   n_nO2=D_n_nO2,msis_pt=control['if use msis'],time=granule['time'],
                                   latitude=granule['latitude'][:,ift],
                                   longitude=granule['longitude'][:,ift],nO2s_prior_option='constant',
                                   max_diverging_step=3,max_iter=6)
            D_nO2s[igranule,ift,result.THMask] = result.params['nO2s_profile'].value
            D_nO2s_dofs[igranule,ift,result.THMask] = result.params['nO2s_profile'].dofs
            D_nO2s_e[igranule,ift,result.THMask] = result.params['nO2s_profile'].posterior_error
            D_T[igranule,ift,result.THMask] = result.params['T_profile'].value
            D_T_dofs[igranule,ift,result.THMask] = result.params['T_profile'].dofs
            D_T_e[igranule,ift,result.THMask] = result.params['T_profile'].posterior_error
            D_nO2_msis[igranule,ift,result.THMask] = result.nO2_profile
            if hasattr(result,'T_profile_msis'):
                D_T_msis[igranule,ift,result.THMask] = result.T_profile_msis
            # save O2 number density scaling factor if at least one layer is adjusted
            if D_n_nO2 > 0:
                tmp = np.full_like(result.params['nO2s_profile'].value,np.nan)
                tmp[0:D_n_nO2] = result.params['nO2Scale_profile'].value
                D_nO2Scale[igranule,ift,result.THMask] = tmp

                tmp = np.full_like(result.params['nO2s_profile'].value,np.nan)
                tmp[0:D_n_nO2] = result.params['nO2Scale_profile'].dofs
                D_nO2Scale_dofs[igranule,ift,result.THMask] = tmp

                tmp = np.full_like(result.params['nO2s_profile'].value,np.nan)
                tmp[0:D_n_nO2] = result.params['nO2Scale_profile'].posterior_error
                D_nO2Scale_e[igranule,ift,result.THMask] = tmp

            D_tangent_height[igranule,ift,result.THMask] = result.tangent_height
            D_dZ[igranule,ift,result.THMask] = result.dZ
            D_HW1E[igranule,ift] = result.params['HW1E'].value
            D_HW1E_dofs[igranule,ift] = result.params['HW1E'].dofs
            D_HW1E_e[igranule,ift] = result.params['HW1E'].posterior_error
            D_w_shift[igranule,ift] = result.params['w_shift'].value
            D_w_shift_dofs[igranule,ift] = result.params['w_shift'].dofs
            D_w_shift_e[igranule,ift] = result.params['w_shift'].posterior_error
            D_chi2[igranule,ift] = result.chi2
            D_Jprior[igranule,ift] = result.Jprior
            D_if_success[igranule,ift] = np.int8(result.if_success)
            D_rmse[igranule,ift] = result.rmse
            D_niter[igranule,ift] = result.niter
            logging.warning('chi2={:.2f},rmse={:.2E},niter={},if_success={}'.format(result.chi2,result.rmse,result.niter,result.if_success))
            logging.warning('takes {:.2f} s'.format(time.time()-Time))
        except Exception as e:
            print(e)
            D_if_success[igranule,ift] = False

#%% save data to a netcdf4 file if control['if save single-pixel file'] is False
if not control['if save single-pixel file']:
    save_path = os.path.join(control['save directory'],os.path.splitext(os.path.split(control['sciamachy file path'])[-1])[0]+control['file suffix']+'.nc')
    if os.path.exists(save_path):
        os.remove(save_path)
    f = Level2_Saver()
    f.create(filename=save_path,longitude=D_longitude,latitude=D_latitude)
    f.set_variable(f.ncid.variables['longitude'],D_longitude)
    f.set_variable(f.ncid.variables['latitude'],D_latitude)
    f.set_variable(f.ncid.variables['tangent_height'],D_tangent_height)
    f.set_variable(f.ncid.variables['layer_thickness'],D_dZ)
    f.set_variable(f.ncid.variables['solar_zenith_angle'],D_solar_zenith_angle)
    f.set_variable(f.ncid.variables['time'],D_time)
    # create the singlet delta group
    f.create_singlet_delta_group(group_name='singlet_delta',if_save_nO2Scale= (D_n_nO2>0))
    # save data to the singlet delta group
    f.set_variable(f.ncdelta.variables['excited_O2'],D_nO2s)
    f.set_variable(f.ncdelta.variables['excited_O2_dofs'],D_nO2s_dofs)
    f.set_variable(f.ncdelta.variables['excited_O2_error'],D_nO2s_e)
    
    f.set_variable(f.ncdelta.variables['temperature'],D_T)
    f.set_variable(f.ncdelta.variables['temperature_dofs'],D_T_dofs)
    f.set_variable(f.ncdelta.variables['temperature_error'],D_T_e)
    
    f.set_variable(f.ncdelta.variables['temperature_msis'],D_T_msis)
    f.set_variable(f.ncdelta.variables['O2_msis'],D_nO2_msis)
    
    f.set_variable(f.ncdelta.variables['HW1E'],D_HW1E)
    f.set_variable(f.ncdelta.variables['HW1E_dofs'],D_HW1E_dofs)
    f.set_variable(f.ncdelta.variables['HW1E_error'],D_HW1E_e)
    
    f.set_variable(f.ncdelta.variables['w_shift'],D_w_shift)
    f.set_variable(f.ncdelta.variables['w_shift_dofs'],D_w_shift_dofs)
    f.set_variable(f.ncdelta.variables['w_shift_error'],D_w_shift_e)
    
    f.set_variable(f.ncdelta.variables['chi2'],D_chi2)
    f.set_variable(f.ncdelta.variables['rmse'],D_rmse)
    f.set_variable(f.ncdelta.variables['if_success'],D_if_success)
    f.set_variable(f.ncdelta.variables['Jprior'],D_Jprior)
    f.set_variable(f.ncdelta.variables['number_of_iterations'],D_niter)
    if D_n_nO2 > 0:
        f.set_variable(f.ncdelta.variables['O2_scaling'],D_nO2Scale)
        f.set_variable(f.ncdelta.variables['O2_scaling_dofs'],D_nO2Scale_dofs)
        f.set_variable(f.ncdelta.variables['O2_scaling_error'],D_nO2Scale_e)
else:
    for iy in range(control['delta start along-track (0-based)'],control['delta end along-track (0-based)']+1):
        for ix in range(control['delta start across-track (0-based)'],control['delta end across-track (0-based)']+1):
            save_path = os.path.join(control['save directory'],os.path.splitext(os.path.split(control['sciamachy file path'])[-1])[0]+'_{}_{}_delta.mat'.format(iy,ix))
            F_save_single_pixel(save_path,iy,ix,D_ngranule,D_nft,D_nth,
                                longitude=D_longitude,
                                latitude=D_latitude,
                                tangent_height=D_tangent_height,
                                time=D_time,
                                temperature=D_T,
                                temperature_dofs=D_T_dofs,
                                temperature_error=D_T_e,
                                temperature_msis=D_T_msis,
                                exited_O2=D_nO2s,
                                exited_O2_dofs=D_nO2s_dofs,
                                exited_O2_error=D_nO2s_e,
                                chi2=D_chi2,
                                rmse=D_rmse)
#%%
if not control['if A band']: 
    if not control['if save single-pixel file']:
        f.close()
else:
    
    S_nO2s = np.full((S_ngranule,S_nft,S_nth),np.nan,dtype=np.float32)
    S_T = np.full((S_ngranule,S_nft,S_nth),np.nan,dtype=np.float32)
    S_nO2s_dofs = np.full((S_ngranule,S_nft,S_nth),np.nan,dtype=np.float32)
    S_T_dofs = np.full((S_ngranule,S_nft,S_nth),np.nan,dtype=np.float32)
    S_nO2s_e = np.full((S_ngranule,S_nft,S_nth),np.nan,dtype=np.float32)
    S_T_msis = np.full((S_ngranule,S_nft,S_nth),np.nan,dtype=np.float32)
    S_nO2_msis = np.full((S_ngranule,S_nft,S_nth),np.nan,dtype=np.float32)
    if S_n_nO2 > 0:
        S_nO2Scale = np.full((S_ngranule,S_nft,S_nth),np.nan,dtype=np.float32)
        S_nO2Scale_dofs = np.full((S_ngranule,S_nft,S_nth),np.nan,dtype=np.float32)
        S_nO2Scale_e = np.full((S_ngranule,S_nft,S_nth),np.nan,dtype=np.float32)
    S_T_e = np.full((S_ngranule,S_nft,S_nth),np.nan,dtype=np.float32)
    S_HW1E = np.full((S_ngranule,S_nft),np.nan,dtype=np.float32)
    S_HW1E_dofs = np.full((D_ngranule,D_nft),np.nan,dtype=np.float32)
    S_HW1E_e = np.full((D_ngranule,D_nft),np.nan,dtype=np.float32)
    S_w_shift = np.full((D_ngranule,D_nft),np.nan,dtype=np.float32)
    S_w_shift_dofs = np.full((D_ngranule,D_nft),np.nan,dtype=np.float32)
    S_w_shift_e = np.full((D_ngranule,D_nft),np.nan,dtype=np.float32)
    S_chi2 = np.full((S_ngranule,S_nft),np.nan,dtype=np.float32)
    S_rmse = np.full((S_ngranule,S_nft),np.nan,dtype=np.float32)
    S_if_success = np.zeros((S_ngranule,S_nft),dtype=np.int8)
    S_Jprior = np.full((S_ngranule,S_nft),np.nan,dtype=np.float32)
    S_niter = np.full((S_ngranule,S_nft),np.nan,dtype=np.int8)
    S_tangent_height = np.full((S_ngranule,S_nft,S_nth),np.nan,dtype=np.float32)
    S_solar_zenith_angle = np.full((S_ngranule,S_nft,S_nth),np.nan,dtype=np.float32)
    S_time = np.full((S_ngranule,S_nth),np.nan,dtype=np.float64)
    S_dZ = np.full((S_ngranule,S_nft,S_nth),np.nan,dtype=np.float32)
    S_latitude = np.full((S_ngranule,S_nft,S_nth),np.nan,dtype=np.float32)
    S_longitude = np.full((S_ngranule,S_nft,S_nth),np.nan,dtype=np.float32)
    
    for (igranule,granule) in enumerate(S_granules):
        if granule['tangent_height'].shape[0] != S_nth:
            continue
        for ift in range(D_nft):
            th_idx = np.argsort(granule['tangent_height'][:,ift])# TH has to go from low to high
            S_latitude[igranule,ift,] = granule['latitude'][th_idx,ift]
            S_longitude[igranule,ift,] = granule['longitude'][th_idx,ift]
            S_tangent_height[igranule,ift,] = granule['tangent_height'][th_idx,ift]
        S_solar_zenith_angle[igranule,] = granule['solar_zenith_angle'].T
        S_time[igranule,:] = granule['time']
    for (igranule,granule) in enumerate(S_granules):
        if granule['tangent_height'].shape[0] != S_nth:
            continue
        if igranule < control['sigma start along-track (0-based)'] or igranule > control['sigma end along-track (0-based)']:
            continue
        for ift in range(S_nft):
            if ift < control['sigma start across-track (0-based)'] or ift > control['sigma end across-track (0-based)']:
                continue
            logging.warning('sigma granule {}, footprint {}'.format(igranule,ift))
            try:
                Time = time.time()
                result = F_fit_profile(tangent_height=granule['tangent_height'][:,ift],
                                       radiance=granule['radiance'][:,ift,:].squeeze(),
                                       radiance_error=granule['radiance_error'][:,ift,:].squeeze(),
                                       wavelength=granule['wavelength'],
                                       startWavelength=S_startWavelength,
                                       endWavelength=S_endWavelength,
                                       minTH=S_minTH,maxTH=S_maxTH,w1_step=S_w1_step,
                                       n_nO2=S_n_nO2,msis_pt=control['if use msis'],time=granule['time'],
                                       latitude=granule['latitude'][:,ift],
                                       longitude=granule['longitude'][:,ift],nO2s_prior_option='constant',
                                       max_diverging_step=3,max_iter=6)
                S_nO2s[igranule,ift,result.THMask] = result.params['nO2s_profile'].value
                S_nO2s_dofs[igranule,ift,result.THMask] = result.params['nO2s_profile'].dofs
                S_nO2s_e[igranule,ift,result.THMask] = result.params['nO2s_profile'].posterior_error
                S_T[igranule,ift,result.THMask] = result.params['T_profile'].value
                S_T_dofs[igranule,ift,result.THMask] = result.params['T_profile'].dofs
                S_T_e[igranule,ift,result.THMask] = result.params['T_profile'].posterior_error
                S_nO2_msis[igranule,ift,result.THMask] = result.nO2_profile
                if hasattr(result,'T_profile_msis'):
                    S_T_msis[igranule,ift,result.THMask] = result.T_profile_msis
                if S_n_nO2 > 0:
                    tmp = np.full_like(result.params['nO2s_profile'].value,np.nan)
                    tmp[0:S_n_nO2] = result.params['nO2Scale_profile'].value
                    S_nO2Scale[igranule,ift,result.THMask] = tmp
                    
                    tmp = np.full_like(result.params['nO2s_profile'].value,np.nan)
                    tmp[0:S_n_nO2] = result.params['nO2Scale_profile'].dofs
                    S_nO2Scale_dofs[igranule,ift,result.THMask] = tmp
                    
                    tmp = np.full_like(result.params['nO2s_profile'].value,np.nan)
                    tmp[0:S_n_nO2] = result.params['nO2Scale_profile'].posterior_error
                    S_nO2Scale_e[igranule,ift,result.THMask] = tmp
                    
                S_tangent_height[igranule,ift,result.THMask] = result.tangent_height
                S_dZ[igranule,ift,result.THMask] = result.dZ
                S_HW1E[igranule,ift] = result.params['HW1E'].value
                S_HW1E_dofs[igranule,ift] = result.params['HW1E'].dofs
                S_HW1E_e[igranule,ift] = result.params['HW1E'].posterior_error
                S_w_shift[igranule,ift] = result.params['w_shift'].value
                S_w_shift_dofs[igranule,ift] = result.params['w_shift'].dofs
                S_w_shift_e[igranule,ift] = result.params['w_shift'].posterior_error
                S_chi2[igranule,ift] = result.chi2
                S_Jprior[igranule,ift] = result.Jprior
                S_rmse[igranule,ift] = result.rmse
                S_if_success[igranule,ift] = np.int8(result.if_success)
                S_niter[igranule,ift] = result.niter
                logging.warning('chi2={:.2f},rmse={:.2E},niter={},if_success={}'.format(result.chi2,result.rmse,result.niter,result.if_success))
                logging.warning('takes {:.2f} s'.format(time.time()-Time))
            except Exception as e:
                print(e)
                S_if_success[igranule,ift] = False

    logging.warning('singlet sigma takes {:.2f} s'.format(time.time()-Time))
    if not control['if save single-pixel file']:
        # create the singlet sigma group
        f.create_singlet_sigma_group(group_name='singlet_sigma',if_save_nO2Scale= (S_n_nO2>0))
        # save data to the singlet sigma group
        f.set_variable(f.ncsigma.variables['excited_O2'],S_nO2s)
        f.set_variable(f.ncsigma.variables['excited_O2_dofs'],S_nO2s_dofs)
        f.set_variable(f.ncsigma.variables['excited_O2_error'],S_nO2s_e)
        
        f.set_variable(f.ncsigma.variables['temperature'],S_T)
        f.set_variable(f.ncsigma.variables['temperature_dofs'],S_T_dofs)
        f.set_variable(f.ncsigma.variables['temperature_error'],S_T_e)
        
        f.set_variable(f.ncsigma.variables['temperature_msis'],S_T_msis)
        f.set_variable(f.ncsigma.variables['O2_msis'],S_nO2_msis)
        
        f.set_variable(f.ncsigma.variables['HW1E'],S_HW1E)
        f.set_variable(f.ncsigma.variables['HW1E_dofs'],S_HW1E_dofs)
        f.set_variable(f.ncsigma.variables['HW1E_error'],S_HW1E_e)
        
        f.set_variable(f.ncsigma.variables['w_shift'],S_w_shift)
        f.set_variable(f.ncsigma.variables['w_shift_dofs'],S_w_shift_dofs)
        f.set_variable(f.ncsigma.variables['w_shift_error'],S_w_shift_e)
        
        f.set_variable(f.ncsigma.variables['chi2'],S_chi2)
        f.set_variable(f.ncsigma.variables['rmse'],S_rmse)
        f.set_variable(f.ncsigma.variables['if_success'],S_if_success)
        f.set_variable(f.ncsigma.variables['Jprior'],S_Jprior)
        f.set_variable(f.ncsigma.variables['number_of_iterations'],S_niter)
        if S_n_nO2 > 0:
            f.set_variable(f.ncsigma.variables['O2_scaling'],S_nO2Scale)
            f.set_variable(f.ncsigma.variables['O2_scaling_dofs'],S_nO2Scale_dofs)
            f.set_variable(f.ncsigma.variables['O2_scaling_error'],S_nO2Scale_e)
        f.close()
    else:
        for iy in range(control['sigma start along-track (0-based)'],control['sigma end along-track (0-based)']+1):
            for ix in range(control['sigma start across-track (0-based)'],control['sigma end across-track (0-based)']+1):
                save_path = os.path.join(control['save directory'],os.path.splitext(os.path.split(control['sciamachy file path'])[-1])[0]+'_{}_{}_sigma.mat'.format(iy,ix))
                F_save_single_pixel(save_path,iy,ix,S_ngranule,S_nft,S_nth,
                                    longitude=S_longitude,
                                    latitude=S_latitude,
                                    tangent_height=S_tangent_height,
                                    time=S_time,
                                    temperature=S_T,
                                    temperature_dofs=S_T_dofs,
                                    temperature_error=S_T_e,
                                    temperature_msis=S_T_msis,
                                    exited_O2=S_nO2s,
                                    exited_O2_dofs=S_nO2s_dofs,
                                    exited_O2_error=S_nO2s_e,
                                    chi2=S_chi2,
                                    rmse=S_rmse)
