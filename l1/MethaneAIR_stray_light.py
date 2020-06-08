# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 11:39:46 2020

@author: kangsun
"""
import numpy as np
import pandas as pd
import glob
#BPM_ch4 = np.genfromtxt(r'C:\research\CH4\stray_light\CH4_bad_pix.csv',
#                        delimiter=',',
#                        dtype=np.int)
def twoDGaussian(inp,amplitude,xo,yo,sigma_x,sigma_y,theta):
    x = inp['x'];y = inp['y']
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

class mergedFrame():
    """
    the mergedFrame object characterizes the combination of multiple exposure times
    at given wavelength/spatial position. The result is a 1280 by 1024 detector
    image.
    Key functions are fitting a 2d gaussian to identify peak column/row and plotting
    the merged frame
    """
    def __init__(self,frames,wv,whichBand='O2'):
        """
        build the merged frame using multiple exposures
        frames:
            a list of masked array, output from Exposure.readdata(wv), start
            from the longest exposure
        wv:
            wavelength in nm for the merged frames
        whichBand:
            'CH4' or 'O2'
        """
        for i in range(1,len(frames)):
            if i == 1:
                data = np.ma.where(np.ma.getmask(frames[0]),frames[1],frames[0])
            else:
                data = np.ma.where(np.ma.getmask(data),frames[i],data)
        # flip row order if it is O2 camera
        if whichBand == 'O2':
            data = data[::-1,:]
        self.mergedFrameData = data
        [rCenter,cCenter] = np.unravel_index(np.argmax(data),data.shape)
        self.rCenterPrior = rCenter;self.cCenterPrior = cCenter;
        self.nrow = 1280;self.ncol = 1024
        self.rgrid = np.arange(self.nrow,dtype=np.float64)
        self.cgrid = np.arange(self.ncol,dtype=np.float64)
        
    def fitPeak(self,rRange=100,cRange=100):
        """
        fit analytical function to find the central column/row
        rRange/cRange:
            one-side extents to subset the FPA
        """
        from scipy.optimize import curve_fit
        cgrid = self.cgrid;rgrid = self.rgrid
        cCenter = self.cCenterPrior;rCenter = self.rCenterPrior
        cmesh,rmesh = np.meshgrid(cgrid,rgrid)
        rIndex = (rgrid >= rCenter-rRange) & (rgrid <= rCenter+rRange)
        cIndex = (cgrid >= cCenter-cRange) & (cgrid <= cCenter+cRange)
        rMeshLocal = rmesh[np.ix_(rIndex,cIndex)]
        cMeshLocal = cmesh[np.ix_(rIndex,cIndex)]
        dataLocal = self.mergedFrameData[np.ix_(rIndex,cIndex)]
        inp = {};inp['x'] = cMeshLocal;inp['y'] = rMeshLocal
        initialGuess = (np.max(dataLocal),cCenter,rCenter,2.,2.,0.)
        popt,pcov = curve_fit(twoDGaussian,inp,dataLocal.ravel(),p0=initialGuess)
        self.rCenterPosterior = popt[2]
        self.cCenterPosterior = popt[1]
        self.peakAmptitude = popt[0]
        self.peakAngle = popt[5]
        self.popt = popt
        self.poptError = np.sqrt(pcov.diagonal())
    
    def plotMergedFrame(self,rRange=-1,cRange=-1,
                        clim=(1e5,1e12),figureFileName='',
                        scale='log',normalizeCenter=False):
        """
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        if normalizeCenter:
            data = self.mergedFrameData/self.popt[0]
        else:
            data = self.mergedFrameData
        if scale == 'log':
            plt.pcolormesh(self.cgrid-0.5,self.rgrid-0.5,data,norm=LogNorm(),
                           cmap='jet')
        else:
            plt.pcolormesh(self.cgrid-0.5,self.rgrid-0.5,data,
                           cmap='viridis')
        if cRange < 0:
            plt.xlim((0,self.ncol-1))
            plt.ylim((130,1000))
            plt.axis('off')
        else:
            plt.xlim((-cRange+self.cCenterPosterior,cRange+self.cCenterPosterior))
            plt.ylim((-rRange+self.rCenterPosterior,rRange+self.rCenterPosterior))
        plt.clim(clim)
#        plt.colorbar()
        if figureFileName:
            plt.savefig(figureFileName,dpi=150)
        
    def unloadData(self):
        """
        unload the merged frames to save some memory
        """
        if hasattr(self,'mergedFrameData'):
            print('Merged frame is not there!')
        else:
            print('Unloading merged frame...')
            del self.mergedFrameData

class Exposure():
    """
    the Exposure object characterizes a laser sweep across a range of columns
    at a constant spatial angle/position and exposure time. 
    The exact row number may vary a little. The data should be organized in the
    same way as Jonathan Franklin did.
    Key functions include reading the data, subtracting dark, masking bad pixels,
    and performa radiometric calibration
    written by Jonathan Franklin with minor edits by Kang Sun
    """
    def __init__(self,exptime,basedir):
        self.exptime = exptime
        self.basedir = basedir
        temp = glob.glob('{}/*.csv'.format(basedir))
        self.csvfil = temp[0]
        self.csv = pd.read_csv(self.csvfil,skipinitialspace=True,skiprows=4)
        self.csv['npy'] = pd.Series(
                ['{}/npy/{}.npy'.format(self.basedir,i.strip('.seq')) \
                 for i in self.csv['SeqName']])
        self.darks = self.csv[self.csv['Dark'] == 1]
        self.brights = self.csv[self.csv['Dark'] == 0]
    def getdarks(self):
        ## Only data in last dark for some reason...
        ## Plan was to average the first and last darks...
        basedark = np.load(self.darks['npy'].iloc[-1]).T
        self.dark = basedark
        
    def getBPM(self,BPM=[],BPM_path=''):
        """
        load bad pixel map(BPB)
        """
        if BPM_path:
            self.BPM = np.genfromtxt(BPM_path,delimiter=',',dtype=np.int)
        else:
            self.BPM = BPM
    
    def getRadCal(self,radCalCoef,radCalPath=''):
        """
        load radiometric calibration
        """
        if radCalPath:
            from scipy.io import loadmat
            self.radCalCoef = loadmat(radCalPath)['coef']
        else:
            self.radCalCoef = radCalCoef
    def readdata(self,wvlen,limit=10000.):
        npfil = self.brights.loc[self.brights['Wavelength'] == wvlen].iloc[0]['npy']
        temp = np.load(npfil).T
        satmap = np.ma.masked_where(temp > limit,temp)
        
        ## Dark correction
        satmap = satmap - self.dark
        ## remove bad pixels
        if hasattr(self,'BPM'):
            satmap = np.ma.masked_where(self.BPM > 0,satmap)
        else:
            print('Bad pixel map is absent!')
        ## apply radiometric calibration
        if hasattr(self,'radCalCoef'):
            radCalCoef = self.radCalCoef;
            new_satmap = np.zeros(satmap.shape,dtype=np.float64)
            for ipoly in range(radCalCoef.shape[-1]):
                new_satmap = new_satmap+radCalCoef[...,ipoly].squeeze()*np.power(satmap,ipoly+1) 
            satmap = new_satmap
        else:
            print('Radiometric calibration coefficients are absent!')
        ## Kludge to avoid issues with taking log later.
        #satmap[satmap <= 0] = 1.
        laser_dBm = self.brights.loc[self.brights['Wavelength'] == wvlen].iloc[0]['Power']
        laser_mW = np.power(10.,laser_dBm/10.)
        satmap = satmap / self.exptime / laser_mW
        return(satmap)