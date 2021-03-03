# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:09:38 2021

@author: kangsun
"""
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import datetime as dt
import sys, os
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from astropy.convolution import convolve_fft
from scipy.interpolate import splrep, splev
import logging
import warnings
import inspect
from collections import OrderedDict

from hapi import db_begin, fetch, arange_, absorptionCoefficient_Doppler, \
absorptionCoefficient_Voigt, absorptionCoefficient_Voigt_jac

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
        
    def getAbsorption(self,nu=None,finiteDifference=True,dT=0.01,WavenumberWing=3.,sourceTableName=None):
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
                            nO2s_profile,T_profile,HW1E,w_shift,
                            nu=[],einsteinA=None):
    '''
    forward model to simulate scia-observed limb spectra for a profile scan
    output jacobians
    '''
    if len(nu) == 0:
        nu = 1e7/w1
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
    # get xsection (sigma_), emission, and jacobians of emission for each layer
    for ith in range(nth):
        l = layer(p=p_profile[ith],T=T_profile[ith],
                  minWavelength=np.min(w1),maxWavelength=np.max(w1),einsteinA=einsteinA)
        l.getAbsorption(nu=nu)
        sigma_[ith,] = l.sigma*l.nO2# this is optical depth divided by length
        dsigmadT_[ith,] = l.dsigmadT*l.nO2-l.sigma*l.p/1.38065e-23*0.2095*1e-6/l.T**2# this is d(optical depth divided by length)/dT
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
        for (count,j) in enumerate(emitting_layer_idx):
            if count == 0:
                obs_R1[i,] = obs_R1[i,]+emission_[j,]/(4*np.pi)*L[i,j]# no absorption for the closest shell
                obs_dR1dT[i,:,j] = obs_dR1dT[i,:,j]+dedT_[j,]/(4*np.pi)*L[i,j]
                obs_dR1dnO2s[i,:,j] = obs_dR1dnO2s[i,:,j]+dednO2s_[j,]/(4*np.pi)*L[i,j]
            else:
                obs_R1[i,] = obs_R1[i,]+emission_[j,]/(4*np.pi)*L[i,j]*np.exp(-cum_tau[count-1,])
                obs_dR1dT[i,:,j] = obs_dR1dT[i,:,j]+dedT_[j,]/(4*np.pi)*L[i,j]*np.exp(-cum_tau[count-1,])
                for (count1,k) in enumerate(emitting_layer_idx[count+1:]):
                    obs_dR1dT[i,:,j] = obs_dR1dT[i,:,j]+emission_[k,]/(4*np.pi)*L[i,k]*np.exp(-cum_tau[count+count1,])*(-L[i,j]*dsigmadT_[j,])
                obs_dR1dnO2s[i,:,j] = obs_dR1dnO2s[i,:,j]+dednO2s_[j,]/(4*np.pi)*L[i,j]*np.exp(-cum_tau[count-1,])
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
                               nO2Scale_profile,T_profile_reference=[],
                               nu=[],einsteinA=None):
    '''
    forward model to simulate scia-observed limb spectra for a profile scan
    output jacobians
    '''
    if len(nu) == 0:
        nu = 1e7/w1
    if len(T_profile_reference) == 0:
        T_profile_reference = T_profile.copy()
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
    # this is O2 number density from ideal gas law
    nO2_profile_full = np.array([p_profile[i]/T_profile_reference[i]/1.38065e-23*0.2095*1e-6 for i in range(len(T_profile))])
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
        sigma_[ith,] = l.sigma*l.nO2# this is optical depth divided by length
        dsigmadnO2_[ith,] = l.sigma# this is d(optical depth divided by length)/dnO2
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
        for (count,j) in enumerate(emitting_layer_idx):
            if count == 0:
                obs_R1[i,] = obs_R1[i,]+emission_[j,]/(4*np.pi)*L[i,j]# no absorption for the closest shell
                obs_dR1dT[i,:,j] = obs_dR1dT[i,:,j]+dedT_[j,]/(4*np.pi)*L[i,j]
                obs_dR1dnO2s[i,:,j] = obs_dR1dnO2s[i,:,j]+dednO2s_[j,]/(4*np.pi)*L[i,j]
            else:
                obs_R1[i,] = obs_R1[i,]+emission_[j,]/(4*np.pi)*L[i,j]*np.exp(-cum_tau[count-1,])
                obs_dR1dT[i,:,j] = obs_dR1dT[i,:,j]+dedT_[j,]/(4*np.pi)*L[i,j]*np.exp(-cum_tau[count-1,])
                for (count1,k) in enumerate(emitting_layer_idx[count+1:]):
                    obs_dR1dT[i,:,j] = obs_dR1dT[i,:,j]+emission_[k,]/(4*np.pi)*L[i,k]*np.exp(-cum_tau[count+count1,])*(-L[i,j]*dsigmadT_[j,])
                    obs_dR1dnO2[i,:,j] = obs_dR1dnO2[i,:,j]+emission_[k,]/(4*np.pi)*L[i,k]*np.exp(-cum_tau[count+count1,])*(-L[i,j]*dsigmadnO2_[j,])
                obs_dR1dnO2s[i,:,j] = obs_dR1dnO2s[i,:,j]+dednO2s_[j,]/(4*np.pi)*L[i,j]*np.exp(-cum_tau[count-1,])
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
                                          nO2s_profile,T_profile,HW1E,w_shift,
                                          nu=[],einsteinA=None):
    '''
    forward model to simulate scia-observed limb spectra for a profile scan
    output jacobians
    '''
    if len(nu) == 0:
        nu = 1e7/w1
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
    # get xsection (sigma_), emission, and jacobians of emission for each layer
    for ith in range(nth):
        l = layer(p=p_profile[ith],T=T_profile[ith],
                  minWavelength=np.min(w1),maxWavelength=np.max(w1),einsteinA=einsteinA)
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
        mngr = plt.get_current_fig_manager()
        geom = mngr.window.geometry()
        x,y,dx,dy = geom.getRect()
        mngr.window.setGeometry(0,100,dx,dy)
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
    
    def retrieve(self,radiance,radiance_error,params=None,max_iter=100,**kwargs):
        
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
        dsigma2 = np.inf
        result = Retrieval_Results()
        while(dsigma2 > nstates and count < max_iter):
            self.logger.info('Iteration {}'.format(count))
            if count != 0:
                params.update_vectors(vector_name='value',vector=beta)
            obs_R2,jacobians = self.evaluate(params,**kwargs)
            yhat = obs_R2.ravel()
            all_jacobians = [jacobians[name] for name in params_names]
            K = np.column_stack(all_jacobians)
            dbeta = np.linalg.inv(Sa_inv+K.T@Sy_inv@K)@(K.T@Sy_inv@(yy-yhat)-Sa_inv@(beta-beta0))
            dsigma2 = dbeta.T@(K.T@Sy_inv@(yy-yhat)+Sa_inv@(beta-beta0))
            self.logger.info('dsigma2: {}'.format(dsigma2))
            self.logger.info(' '.join('{:2E}'.format(b) for b in beta))
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
    sample prior information at tangent height
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

def F_fit_profile(tangent_height,radiance,radiance_error,wavelength,
                  startWavelength=1240,endWavelength=1300,
                  minTH=None,maxTH=None,w1_step=None,einsteinA=None,
                  if_attenuation=True,n_nO2=0,nO2Scale_error=0.1,
                  max_iter=10):
    
    if endWavelength < 800:
        HW1E_prior = 0.3
        if minTH is None:
            minTH = 50;maxTH = 120;
        if w1_step is None:
            w1_step = -0.0002
        if einsteinA is None:
            einsteinA = 0.08693#10.1016/j.jqsrt.2010.05.011
    
    if startWavelength > 1100:
        HW1E_prior = 0.8
        if minTH is None:
            minTH = 20;maxTH = 120
        if w1_step is None:
            w1_step = -0.001
        if einsteinA is None:
            einsteinA = 2.27e-4
    
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
    p_profile, T_profile = F_sample_standard_atm(tangent_height)
    # w1 is the high res wavelength grid. has to be descending
    w1 = arange_(endWavelength,startWavelength,-np.abs(w1_step))#-0.0005
    T_profile_e = np.ones(T_profile.shape)*20
    T_profile_e[tangent_height<50] = 2
    nO2s_profile_e = np.ones(nO2s_profile.shape)*nO2s_profile
    nO2s_profile_e[nO2s_profile_e<0.1*np.mean(nO2s_profile)] = 0.1*np.mean(nO2s_profile)
    p_profile_middle = p_profile+np.append(np.diff(p_profile),0.)/2
    
    if not if_attenuation:
        aOE = Forward_Model(func=F_airglow_forward_model_no_absorption,
                                  independent_vars=['w1','wavelength','L','p_profile','nu','einsteinA'],
                                  param_names=['nO2s_profile','T_profile','HW1E','w_shift'])
        aOE.set_prior('nO2s_profile',prior=nO2s_profile,prior_error=nO2s_profile_e,p_profile=p_profile_middle,correlation_scaleHeight=1)
        aOE.set_prior('T_profile',prior=T_profile,prior_error=T_profile_e,p_profile=p_profile_middle,correlation_scaleHeight=1,vmin=50,vmax=500)
        aOE.set_prior('HW1E',prior=HW1E_prior,prior_error=HW1E_prior/2)
        aOE.set_prior('w_shift',prior=0.,prior_error=1.)
        result = aOE.retrieve(radiance,radiance_error,max_iter=max_iter,
                              w1=w1,wavelength=wavelength,
                              L=L,p_profile=p_profile_middle,einsteinA=einsteinA)
        result.tangent_height = tangent_height
        result.THMask = THMask
        result.dZ = dZ
        result.p_profile_middle = p_profile_middle
        result.p_profile = p_profile
        result.T_profile_prior = T_profile
        result.nO2s_profile_prior = nO2s_profile
        return result
    
    if n_nO2 == 0:
        aOE = Forward_Model(func=F_airglow_forward_model,
                                  independent_vars=['w1','wavelength','L','p_profile','nu','einsteinA'],
                                  param_names=['nO2s_profile','T_profile','HW1E','w_shift'])
        aOE.set_prior('nO2s_profile',prior=nO2s_profile,prior_error=nO2s_profile_e,p_profile=p_profile_middle,correlation_scaleHeight=1)
        aOE.set_prior('T_profile',prior=T_profile,prior_error=T_profile_e,p_profile=p_profile_middle,correlation_scaleHeight=1,vmin=50,vmax=500)
        aOE.set_prior('HW1E',prior=HW1E_prior,prior_error=HW1E_prior/2)
        aOE.set_prior('w_shift',prior=0.,prior_error=1.)
        result = aOE.retrieve(radiance,radiance_error,max_iter=max_iter,
                              w1=w1,wavelength=wavelength,
                              L=L,p_profile=p_profile_middle,einsteinA=einsteinA)
    else:
        nO2Scale_profile = np.ones(n_nO2)
        nO2Scale_profile_e = np.ones(n_nO2)*nO2Scale_error
        aOE = Forward_Model(func=F_airglow_forward_model_nO2Scale,
                                  independent_vars=['w1','wavelength','L','p_profile','nu','T_profile_reference','einsteinA'],
                                  param_names=['nO2s_profile','T_profile','HW1E','w_shift','nO2Scale_profile'])
        aOE.set_prior('nO2s_profile',prior=nO2s_profile,prior_error=nO2s_profile_e,p_profile=p_profile_middle,correlation_scaleHeight=1)
        aOE.set_prior('T_profile',prior=T_profile,prior_error=T_profile_e,p_profile=p_profile_middle,correlation_scaleHeight=1,vmin=50,vmax=500)
        aOE.set_prior('HW1E',prior=HW1E_prior,prior_error=HW1E_prior/2)
        aOE.set_prior('w_shift',prior=0.,prior_error=1.)
        aOE.set_prior('nO2Scale_profile',prior=nO2Scale_profile,prior_error=nO2Scale_profile_e,
                      p_profile=p_profile_middle[0:len(nO2Scale_profile)],correlation_scaleHeight=1,vmin=0,vary=True)
        result = aOE.retrieve(radiance,radiance_error,max_iter=max_iter,
                              w1=w1,wavelength=wavelength,
                              L=L,p_profile=p_profile_middle,einsteinA=einsteinA)
    result.tangent_height = tangent_height
    result.THMask = THMask
    result.dZ = dZ
    result.p_profile_middle = p_profile_middle
    result.p_profile = p_profile
    result.T_profile_prior = T_profile
    result.nO2s_profile_prior = nO2s_profile
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
            setattr(self,data_names[i],self.fid[f][:])
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
        self.ace_lon = ace_fid['longitude'][:][time_mask,]
        self.ace_lat = ace_fid['latitude'][:][time_mask,]
        self.ace_time = ace_seconds_since2000[time_mask]
        self.ace_temperature = ace_fid['temperature'][:][time_mask,]
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
