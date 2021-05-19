# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 21:08:43 2020

@author: kangsun
"""
import numpy as np
import sys, os
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.linear_model import Lars, LinearRegression
from scipy.interpolate import interp1d

whichBand = 'O2'
whichMonth = 'March'
whichMachine = 'cspc'
ifPlotDiagnose = False
ifSavgolFigure = False

if whichBand == 'CH4':
    center_w_vec = np.array([1593,1600,1610,1620,1630,1640,1650,1660,1670],dtype=np.float)
    if whichMachine == 'UB':
        isrf_dir = '/home/kangsun/CH4/ISRF/output'
        fig_dir = None
    elif whichMachine == 'cspc':
        isrf_dir = r'C:\Users\Carly\summer\ISRF\output'  
        fig_dir = r'C:\Users\Carly\summer\methaneAIR_figures'
    # max spectral pixel extent in issf/isrf, one-sided (see K_far setting in run_isrf.m)
    max_pix = 7.5
    # polynomial order of wavelength calibration
    n_wavcal_poly = 1
    # rough definition of positive tail start value for use in savgol filter
    tail = 0.37
    # spatial pixel exception for median filter replacement
    ft_except = np.array([502,503,504,505,506])

elif whichBand == 'O2':
    if whichMonth == 'Jan':
        center_w_vec = np.arange(1254,1317+1,7,dtype=np.float)
        ft_except = np.array([780,781,782])
        if whichMachine == 'UB':
            isrf_dir = '/home/kangsun/CH4/ISRF/output'
        elif whichMachine == 'cspc':
            isrf_dir = r'C:\Users\Carly\summer\ISRF\output\jan'
            fig_dir = isrf_dir
        
    elif whichMonth == 'March':
        center_w_vec = np.concatenate([np.arange(1249,1263,2), np.arange(1268,1324,7)])
        ft_except = np.array([780-34, 781-34, 782-34, 783-34,])
        if whichMachine == 'UB':
            isrf_dir = None
        elif whichMachine == 'cspc':
            isrf_dir = r'C:\Users\Carly\summer\ISRF\output\mar'
            fig_dir = isrf_dir   
    max_pix = 7.5
    n_wavcal_poly = 1
    tail = 0.28
    
ncenter = len(center_w_vec)
if 'center_mat' in locals():
    del center_mat

for (icenter,wv) in enumerate(center_w_vec):
    print('loading isrf wavelength %.1f'%wv+' nm')
    d = loadmat(os.path.join(isrf_dir,whichBand+'_ISRF_%.1f'%wv+'.mat'))
    if 'center_mat' not in locals():
        nft = d['nft'].squeeze()
        center_mat = np.empty((ncenter,d['nft'].squeeze()),dtype = np.float)
        center_mat_median = np.empty((ncenter,d['nft'].squeeze()),dtype = np.float)
    center_mat[icenter,:] = d['final_center_pix_vec'].squeeze()
    center_mat_median[icenter,:] = d['final_center_pix_vec'].squeeze()-\
    np.nanmedian(d['final_center_pix_vec'].squeeze())

center_mat_median2 = np.nanmedian(center_mat_median,axis=0)

if ifPlotDiagnose:
    plt.clf()
    plt.plot(np.arange(center_mat.shape[1]),center_mat_median.T,'.')
    plt.plot(np.arange(center_mat.shape[1]),center_mat_median2,'-k')

#%%
center_mat_smooth = np.empty(center_mat.shape,dtype=np.float)
for icenter in range(center_mat.shape[0]):
    ydata = center_mat[icenter,:].copy()
    xdata = center_mat_median2.copy()
    mask = (~np.isnan(xdata)) & (~np.isnan(ydata))
    xdata = xdata[mask];ydata = ydata[mask]
    reg = LinearRegression(fit_intercept=True)
    p1 = reg.fit(X=xdata[:,np.newaxis],y=ydata).coef_
    p0 = reg.fit(X=xdata[:,np.newaxis],y=ydata).intercept_
    if whichBand == 'O2' or center_w_vec[icenter] >= 1670:
        center_mat_smooth[icenter,:] = p0+p1*center_mat_median2
    else:
        center_mat_smooth[icenter,:] = center_mat[icenter,:]

#if ifPlotDiagnose:
#    plt.clf()
#    plt.plot(xdata,ydata,'ok')
#    plt.plot(xdata,p0+xdata*p1,'-b')
#    reg_ols = LinearRegression()
#    p1_ols = reg_ols.fit(xdata[:,np.newaxis],ydata).coef_
#    p0_ols = reg_ols.fit(xdata[:,np.newaxis],ydata).intercept_
#    plt.plot(xdata,p0_ols+xdata*p1_ols,'-r')
#    plt.legend(['data','LARS robust fit','OLS'])

if ifPlotDiagnose:
    plt.clf()
    for icenter in range(center_mat.shape[0]):
        local_center = center_mat[icenter,:]
        local_center = local_center - np.nanmedian(local_center)
        local_center_smooth = center_mat_smooth[icenter,:]
        local_center_smooth = local_center_smooth - np.nanmedian(local_center_smooth)
        #plt.plot(np.arange(center_mat.shape[1]),local_center,'o')
        #plt.plot(np.arange(center_mat.shape[1]),local_center_smooth,'*')
        
wavcal_poly = np.full((nft,n_wavcal_poly+1),np.nan,dtype=np.float)
for ift in range(nft):
    xdata = center_mat_smooth[:,ift]
    if np.sum(np.isnan(xdata)) > 2:
        print('Footprint %d appears to be empty'%ift)
        continue
    ydata = center_w_vec
    wavcal_poly[ift,:] = np.flip(np.polyfit(xdata,ydata,n_wavcal_poly))
    if ifPlotDiagnose and ift in np.array([50, 100, 550]):
        plt.figure()
        plt.subplot(211)
        plt.plot(xdata,ydata,'ok',xdata,np.polyval(np.flip(wavcal_poly[ift,:]),xdata))
        plt.subplot(212)
        plt.plot(xdata,ydata-np.polyval(np.polyfit(xdata,ydata,1),xdata),'ok',
                 xdata,np.polyval(np.flip(wavcal_poly[ift,:]),xdata)\
                 -np.polyval(np.polyfit(xdata,ydata,1),xdata))

#%%
# from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

def iterative_savgol(y_isrf, window_length=81, polyorder=3, 
                     logResidualThreshold=[0.5,0.25,0.1,0.05,0.01], nIteration=None, tail=0.28):
    if nIteration == None:
        nIteration = len(logResidualThreshold)
    isrf_savgol=savgol_filter(y_isrf,window_length=window_length, polyorder=polyorder)
    y_isrf_filtered = y_isrf.copy()
    isrf_dw = np.linspace(-0.75,0.75,1501,dtype=np.float)
    for i in range(nIteration):
        log_resids = np.abs(np.log10(isrf_savgol)-np.log10(y_isrf))
        if i < len(logResidualThreshold):
            threshold = logResidualThreshold[i]
        else:
            threshold = logResidualThreshold[-1]
        loc = np.where((log_resids > threshold) & (np.abs(isrf_dw)>=tail))
        y_isrf_filtered[loc] = isrf_savgol[loc]
        isrf_savgol=savgol_filter(y_isrf_filtered,window_length=window_length, polyorder=polyorder)         
    return isrf_savgol

if ifPlotDiagnose:
    plt.close('all')
isrf_dw = np.linspace(-0.75,0.75,1501,dtype=np.float)
if 'isrf_lowess' in locals():
    del isrf_lowess
if ifPlotDiagnose:
    center_range = [6] #1
    ft_range = [1140] #469
else:
    center_range = range(ncenter)
    ft_range = range(nft)

for icenter in center_range:
    print('loading isrf wavelength %.1f'%center_w_vec[icenter]+' nm')
    d = loadmat(os.path.join(isrf_dir,whichBand+'_ISRF_%.1f'%center_w_vec[icenter]+'.mat'))
    if 'isrf_savgol' not in locals():
#        isrf_lowess = np.full((d['nft'].squeeze(),ncenter,len(isrf_dw)),np.nan)
        isrf_savgol = np.full((d['nft'].squeeze(),ncenter,len(isrf_dw)),np.nan)
    for ift in ft_range:#range(nft):
        x = d['xx_all_row'][ift,:]-d['final_center_pix_vec'].squeeze()[ift]
        y = d['yy_all_row'][ift,:]
        mask = (~np.isnan(x))&(~np.isnan(y))
        if np.sum(mask) < 100:
            print('footprint %d appears to be empty'%ift)
            continue
        x = x[mask];y = y[mask]
        # smooth oversampled issf by lowess
#        y_smooth = lowess(y,x,frac=0.025,return_sorted=False)
        nm_per_pix = 0.
        for ipoly in range(1,n_wavcal_poly+1):# loop over 1, 2, 3, 4 if n_wavcal_poly=4
            nm_per_pix = nm_per_pix+wavcal_poly[ift,ipoly]*ipoly*\
            np.power(d['final_center_pix_vec'].squeeze()[ift],ipoly-1)
        # linear-interpolate the smoothed oversampled issf to wavelength grid, making it isrf
#        interp_func = interp1d(x*-nm_per_pix,y_smooth,bounds_error=False,fill_value=np.nan)
#        isrf_lowess[ift,icenter,:] = interp_func(isrf_dw)
        # linear interpolate oversampled issf to wavelength grid and flip to make it isrf
        interp_func = interp1d(x*-nm_per_pix,y,bounds_error=False,fill_value=np.nan)
        y_isrf = interp_func(isrf_dw)
        y_isrf[np.isnan(y_isrf)] = np.nanmin(y_isrf)
        # smooth the isrf, which is on regular wavelength grid isrf_dw by savitzky golay fileter
#        isrf_savgol[ift,icenter,:] = savgol_filter(y_isrf,window_length=81,polyorder=3)
        isrf_savgol[ift,icenter,:] = iterative_savgol(y_isrf, window_length=81, polyorder=3, logResidualThreshold=[0.5,0.25,0.1,0.05,0.01], nIteration=None, tail=tail)
        mask = (isrf_dw <-np.abs(nm_per_pix*max_pix)) | (isrf_dw > np.abs(nm_per_pix*max_pix))
        isrf_savgol[ift,icenter,mask] = 0
        
#        isrf_lowess[ift,icenter,mask] = 0

        if ifSavgolFigure:
            single_isrf_savgol = np.full((d['nft'].squeeze(),ncenter,len(isrf_dw)),np.nan)
            single_isrf_savgol[ift,icenter,:] = savgol_filter(y_isrf,window_length=81,polyorder=3)
            single_isrf_savgol[ift,icenter,mask] = 0
            np.save(os.path.join(fig_dir, 'savgol_plot_data_'+whichBand+'.npy'),\
                    {'raw_x': x*-nm_per_pix, 'raw_y': y, 'isrf_dw':isrf_dw, 'single_isrf_savgol': single_isrf_savgol, \
                     'isrf_savgol':isrf_savgol, 'center_wv': icenter, 'ft': ift})
                
        if ifPlotDiagnose:
            plt.figure()
            plt.subplot(211)
            plt.plot(x*-nm_per_pix,y,'o',markersize=2,color='grey')
            plt.plot(isrf_dw,isrf_savgol[ift,icenter,:].squeeze())
            
            plt.subplot(212)
            plt.semilogy(x*-nm_per_pix,y,'o',markersize=2,color='grey')
            plt.semilogy(isrf_dw,isrf_savgol[ift,icenter,:].squeeze())

        # normalize isrf so it integrates to unity
        isrf_savgol[ift,icenter,:] = isrf_savgol[ift,icenter,:]\
        /np.trapz(isrf_savgol[ift,icenter,:],isrf_dw)
#        isrf_lowess[ift,icenter,:] = isrf_lowess[ift,icenter,:]\
#        /np.trapz(isrf_lowess[ift,icenter,:],isrf_dw)      

#%% bad pixel smoothing
isrf = isrf_savgol.copy()
isrf_median = median_filter(isrf,size=3)
rms = np.sqrt(np.sum(np.power(isrf_median-isrf,2),axis=2)/(np.count_nonzero(~np.isnan(isrf), axis=2)-1))
ifPlotDiagnose=True
if ifPlotDiagnose:
    plt.clf()
    plt.plot(np.arange(nft), rms, '-o', markersize=4)
    plt.legend(['%d nm'%w for w in center_w_vec[:-1]], bbox_to_anchor=(1,1))
    plt.xlabel('Spatial pixel')
    plt.ylabel('RMSE')
    #plt.ylim(0,0.10)
    #plt.axhline(0.014)
    #plt.axhline(np.nanmean(rms)+2*np.nanstd(rms), color='k')
    plt.show()
#%%
# SEPARATE THRESHOLD FOR EACH WAVELENGTH
smoothed_isrf = median_filter(isrf, size=(5,3,1))
num_outliers = 0 #number of pixels replaced
for i in range(len(center_w_vec)):
    threshold = np.nanmean(rms[:,i]) + 3*np.nanstd(rms[:,i])
    outliers = np.asarray(np.nonzero(rms[:,i] > threshold))
    num_outliers += outliers.shape[1]
    except_filter = ~np.isin(outliers, ft_except)
    fil_outliers = outliers[except_filter]
    isrf[fil_outliers,i,:] = smoothed_isrf[fil_outliers,i,:]
    print(threshold)
print(num_outliers)

if ifPlotDiagnose:
    #rms_new = np.sqrt(np.sum(np.power(isrf_median-isrf,2),axis=2))
    rms_new = np.sqrt(np.sum(np.power(isrf_median-isrf,2),axis=2)/(np.count_nonzero(~np.isnan(isrf), axis=2)-1))
    plt.clf()
    plt.plot(np.arange(nft), rms_new, '-o', markersize=4)
    plt.legend(['%d nm'%w for w in center_w_vec], bbox_to_anchor=(1,1))
    plt.xlabel('Spatial pixel')
    plt.ylabel('RMSE')
    plt.show()
    
#%% 
from netCDF4 import Dataset
# file name of the spec cal data
isrf_dir = r'C:\Users\Carly\summer\ISRF\output\nov_updates'
output_fn = os.path.join(isrf_dir,'methaneair_'+whichBand.lower()+'_spectroscopic_calibration_%d_footprints.nc'%nft)
nc = Dataset(output_fn,'w')
nc.createDimension('delta_wavelength',len(isrf_dw))
nc.createDimension('central_wavelength',ncenter)
nc.createDimension('ground_pixel',nft)
nc.createDimension('polynomial',n_wavcal_poly+1)

var_isrf_dw = nc.createVariable('delta_wavelength',np.float,('delta_wavelength',))
var_isrf_dw.units = 'nm'
var_isrf_dw.long_name = 'wavelength grid of ISRF'
var_isrf_dw[:] = isrf_dw

var_isrf_w = nc.createVariable('central_wavelength',np.float,('central_wavelength',))
var_isrf_w.units = 'nm'
var_isrf_w.long_name = 'wavelength grid where ISRF was measured'
var_isrf_w[:] = center_w_vec

var_isrf = nc.createVariable('isrf',np.float,('ground_pixel','central_wavelength','delta_wavelength'))
var_isrf.units = 'nm^-1'
var_isrf.long_name = 'ISRF'
var_isrf[:] = isrf

var_wavcal = nc.createVariable('pix2nm_polynomial',np.float,('ground_pixel','polynomial'))
var_wavcal.long_name = 'wavelength calibration coefficients, starting from intercept'
var_wavcal[:] = wavcal_poly

nc.close()
#%%
if ifSavgolFigure:
    fig, ax = plt.subplots(2,1, sharey=True, figsize=(6,5), dpi=200)
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    ch4_data = np.load(os.path.join(fig_dir, 'savgol_plot_data_CH4.npy'), allow_pickle=True).flat[0]
    o2_data = np.load(os.path.join(fig_dir, 'savgol_plot_data_O2.npy'), allow_pickle=True).flat[0]
    # order everything ch4, o2
    data = [ch4_data, o2_data]
    inset_loc = [(473,335), (467,167)]
    inset_lim = [(0.55, 0.70, 1.5e-4, 1.5e-3), (0.38, 0.52, 1.5e-4, 1.49e-3)]
    plot_lims = [(-0.7,0.7), (-0.605,0.605)]
    titles = [r'(a) CH$_4$ band', r'(b) O$_2$ band']
    i = 0
    for d in data: 
        axins = zoomed_inset_axes(ax[i], 3, bbox_to_anchor=inset_loc[i])
        x1, x2, y1, y2 = inset_lim[i][0], inset_lim[i][1], inset_lim[i][2], inset_lim[i][3]
        axins.set_xlim(x1, x2) # apply the x-limits
        axins.set_ylim(y1, y2) # apply the y-limits
        axins.xaxis.set_visible(False)
        axins.yaxis.set_visible(False)
        axins.semilogy(d['raw_x'],d['raw_y'],'o',markersize=2,color='grey', label='Raw data')
        axins.semilogy(d['isrf_dw'], d['single_isrf_savgol'][d['ft'], d['center_wv'],:].squeeze(), \
                       linewidth = 0.95, color='r', label='Single pass filter')
        axins.semilogy(d['isrf_dw'],d['isrf_savgol'][d['ft'], d['center_wv'],:].squeeze(), \
                       linewidth=1, color = 'navy', label='Iterative filter')
        mark_inset(ax[i], axins, loc1=2, loc2=4, fc="none", ec="0.5")
        fig.subplots_adjust(hspace = 0.55)
        ax[i].semilogy(d['raw_x'],d['raw_y'],'o',markersize=2,color='grey', label='Raw data')
        ax[i].semilogy(d['isrf_dw'], d['single_isrf_savgol'][d['ft'], d['center_wv'],:].squeeze(), \
                       linewidth = 0.95, color='r', label='Single pass filter')
        ax[i].semilogy(d['isrf_dw'],d['isrf_savgol'][d['ft'], d['center_wv'],:].squeeze(), \
                       linewidth=1, color = 'navy', label='Iterative filter')
        ax[i].set_xlim(plot_lims[i])
        ax[i].legend(loc='lower center', fontsize=10, frameon=False)
        ax[i].set_title(titles[i], fontweight='bold')
        ax[i].set_ylabel('Signal')
        ax[i].set_xlabel('Relative wavelength (nm)')
        i+=1
plt.savefig(os.path.join(fig_dir,'savgol_demo_both.pdf'), bbox_inches='tight')    
# Note: The inset axes look terrible on the preview, but show up properly in the saved pdf