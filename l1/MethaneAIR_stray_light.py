# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 11:39:46 2020

@author: kangsun
"""
import numpy as np
import pandas as pd
import glob
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
#BPM_ch4 = np.genfromtxt(r'C:\research\CH4\stray_light\CH4_bad_pix.csv',
#                        delimiter=',',
#                        dtype=np.int)

class medianStrayLight():
    """
    the medianStrayLight object combines multiple mergedFrame objects by interpolating
    them on a common column difference/row difference mesh grid, stacking them
    together, and taking the median
    """
    def __init__(self,colAllGrid,rowAllGrid,whichBand='O2'):
        """
        """
        self.whichBand = whichBand
        self.nColAll = len(colAllGrid)
        self.nRowAll = len(rowAllGrid)
        self.interpMat = np.empty((self.nRowAll,self.nColAll,0),dtype=np.float32)
        [colAllMesh,rowAllMesh] = np.meshgrid(colAllGrid,rowAllGrid)
        self.colAllMesh = colAllMesh
        self.rowAllMesh = rowAllMesh
        # might need these?
        self.colAllGrid = colAllGrid
        self.rowAllGrid = rowAllGrid
        
    def loadMergedFrame(self, mergedFramePath, filter_rows=[[1,0]], 
                        filter_columns=[[1,0]]):
        """
        load merged frame data and concatenate to interpMat
        """
        # load mergedFramePath
        from scipy.io import loadmat
            
        mergedFrame = loadmat(mergedFramePath)
        mergedFrameData = mergedFrame['mergedFrameData']
        popt = np.transpose(mergedFrame['popt'])
        mergedFrameData = mergedFrameData/popt[0]
        
        # filter out unwanted pixels (e.g., columns < 400 for CH4)
        # write this in later?
        if self.whichBand == 'CH4':
            mergedFrameData[:,0:400] = np.nan
        # shift the mergedFrame so the peak is at row 0, column, interpolate to colAllMesh, rowAllMesh
        columns = np.arange(1024) - popt[1]
        rows = np.arange(1280) - popt[2]
        f = RegularGridInterpolator((rows, columns), mergedFrameData, \
                                              bounds_error=False, fill_value=np.nan)
        interpdata = f((self.rowAllMesh, self.colAllMesh))
        interpdata.astype('float32')
        
        # filter out pixels at unwanted spatial stray light locations, the result is interpdata
        peak_col = np.abs(self.colAllGrid[0])
        peak_row = np.abs(self.rowAllGrid[0])
        if filter_rows[0][0] < filter_rows[0][1]:
            for i in range(len(filter_rows)):
                interpdata[peak_row+filter_rows[i][0]:peak_row+filter_rows[i][1], \
                            peak_col+filter_columns[i][0]:peak_col+filter_columns[i][1]] = np.nan
            
        # stack the interpolated data together
        self.interpMat = np.concatenate((self.interpMat,interpdata[...,np.newaxis]),axis=2)
        
    def takeMedian(self):
        self.medianStrayLight = np.nanmedian(self.interpMat,axis=2)
       
    def plotMedianStrayLightPcolor(self):
        """
        plot the median stray light with pcolormesh
        """
        from matplotlib.colors import LogNorm
        plt.pcolormesh(self.colAllGrid-0.5, self.rowAllGrid-0.5, self.medianStrayLight, \
                       norm=LogNorm(), cmap='jet')
        plt.clim(1e-8, 1)
        plt.colorbar()
        plt.xlabel('Column distance')
        plt.ylabel('Row distance')
        plt.xlim(-1024/2, 1024/2)
        plt.ylim(-500, 500)
        
    def plotMedianStrayLightSurface(self):
        """
        plot the median stray light with plot_surface
        """
        # issue with scaling z-axis to log scale
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.colors import LogNorm
        fig = plt.figure()
        ax = fig.gca(projection='3d')   
        plotmat = self.medianStrayLight/np.nanmax(self.medianStrayLight)
        plotmat[plotmat<=0] = np.nan
        plotmat = np.log10(plotmat)
        ax.set_zlim(-8,0)
        ax.set_ylim(np.min(self.rowAllGrid), np.max(self.rowAllGrid))
        ax.set_xlim(np.min(self.colAllGrid), np.max(self.colAllGrid))
        ax.plot_surface(self.colAllMesh, self.rowAllMesh, plotmat, \
                       vmin = -8, vmax = 0,
                       rcount = 500, ccount = 500, cmap='jet')
        
    def plotSpectralStrayLight(self):
        """
        plot spectral stray light
        """
        plt.plot(self.colAllGrid, np.nanmax(self.medianStrayLight, 0), 'o', markersize=2.5)
        plt.yscale('log')
        plt.xlabel('Column distance')
        
    def plotSpatialStrayLight(self):
        """
        plot spatial stray light
        """
        plt.plot(self.rowAllGrid, np.nanmax(self.medianStrayLight, 1),'o', markersize=2.5)
        plt.yscale('log')
        plt.xlabel('Row distance')


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
    def __init__(self,frames,wv):
        """
        build the merged frame using multiple exposures
        frames:
            a list of masked array, output from Exposure.readdata(wv), start
            from the longest exposure
        wv:
            wavelength in nm for the merged frames
        """
        for i in range(1,len(frames)):
            if i == 1:
                data = np.ma.where(np.ma.getmask(frames[0]),frames[1],frames[0])
            else:
                data = np.ma.where(np.ma.getmask(data),frames[i],data)
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
        self.laser_mW = laser_mW
        return(satmap)