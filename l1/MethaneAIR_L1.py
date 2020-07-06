# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:13:35 2020

@author: kangsun
"""
import numpy as np
import datetime as dt
import struct
import os
import logging
from scipy.io import loadmat
from netCDF4 import Dataset
from scipy.interpolate import interp1d

def F_isrf_convolve_fft(w1,s1,w2,isrf_w,isrf_dw0,isrf_lut0):
    """
    astropy.covolute.convolve_fft-based convolution using wavelength-dependent isrf
    w1:
        high-resolution wavelength
    s1:
        high-resolution spectrum
    w2:
        low-resolution wavelength
    isrf_w:
        center wavelength grid of isrf lut
    isrf_dw:
        wavelength grid on which isrfs are defined
    isrf_lut:
        instrument spectral response function look up table
    """
    from astropy.convolution import convolve_fft
    from scipy.interpolate import RegularGridInterpolator
    from math import isclose
    if isrf_lut0.shape != (len(isrf_w),len(isrf_dw0)):
        raise ValueError('isrf_lut dimension incompatible!')
        return np.full(w2.shape,np.nan)
    # make sure w1 and isrf_dw have the same resolution
    w1_step = np.median(np.diff(w1))
    isrf_dw_step = np.median(np.diff(isrf_dw0))
    if isclose(w1_step,isrf_dw_step):
        isrf_dw = isrf_dw0
        isrf_lut = isrf_lut0
    else:# if not, interpolate isrf to a compatible grid with w1
        isrf_dw_min = np.min(isrf_dw0)
        isrf_dw_max = -isrf_dw_min
        isrf_dw = np.linspace(isrf_dw_min,isrf_dw_max,int((isrf_dw_max-isrf_dw_min)/w1_step)+1)
        isrf_lut = np.zeros((len(isrf_w),len(isrf_dw)))
        for (iw,w) in enumerate(isrf_w):
            interp_func = interp1d(isrf_dw0,isrf_lut0[iw,:])
            isrf_lut[iw,:] = interp_func(isrf_dw)
    # note that the isrf is flipped: convolution is the mirror-image of kernel averaging
    s2_fft_lut = np.array([convolve_fft(s1,isrf_lut[iw,::-1]) for (iw,w) in enumerate(isrf_w)])
    inter_func = RegularGridInterpolator((isrf_w,w1),s2_fft_lut,bounds_error=False)
    return inter_func((w2,w2))

# below ReadSeq routines are from Jonathan Franklin
class Sequence():
    pass

class FrameMeta():
    pass

def ParseFrame(framenum,height,width,bindata):
#    print(framenum)
    Meta = FrameMeta()
    
    # Calc size of actual image minus meta data
    numpixels = (height-1)*width
    fmt = '<{}h'.format(numpixels)
    fmtsize = struct.calcsize(fmt)
    dataraw = struct.unpack_from(fmt,bindata[:fmtsize])
    data = np.reshape(dataraw,(height-1,width))
      
    # Grab time stamp
    temp = struct.unpack_from('<lh',bindata[-6:])
    Meta.timestamp = dt.datetime(1970,1,1) + \
              dt.timedelta(seconds=temp[-2],microseconds=temp[-1]*1000)
    # Grab Meta Data -- Not all of this seems to have real data in it.
    metaraw = bindata[fmtsize:-6]
    temp = struct.unpack_from('<{}h'.format(width),metaraw)
    metaraw = struct.pack('>{}h'.format(width),*temp)
      
    Meta.partNum = metaraw[2:34].decode('ascii').rstrip('\x00')
    Meta.serNum = metaraw[34:48].decode('ascii').rstrip('\x00')
    Meta.fpaType = metaraw[48:64].decode('ascii').rstrip('\x00')
#    print(Meta.partNum)
#    print(Meta.serNum)
#    print(Meta.fpaType)
    Meta.crc = struct.unpack_from('>I',metaraw[64:68])[0]
    Meta.frameCounter = struct.unpack_from('>i',metaraw[68:72])[0]
    Meta.frameTime = struct.unpack_from('>f',metaraw[72:76])[0]
    Meta.intTime = struct.unpack_from('>f',metaraw[76:80])[0]
    Meta.freq = struct.unpack_from('>f',metaraw[80:84])[0]
    Meta.boardTemp = struct.unpack_from('>f',metaraw[120:124])[0]
    Meta.rawNUC = struct.unpack_from('>H',metaraw[124:126])[0]
    Meta.colOff = struct.unpack_from('>h',metaraw[130:132])[0]
    Meta.numCols = struct.unpack_from('>h',metaraw[132:134])[0] + 1
    Meta.rowOff = struct.unpack_from('>h',metaraw[136:138])[0]
    Meta.numRows = struct.unpack_from('>h',metaraw[138:140])[0] + 1
    timelist = struct.unpack_from('>7h',metaraw[192:206])
    Meta.yr = timelist[0]
    Meta.dy = timelist[1]
    Meta.hr = timelist[2]
    Meta.mn = timelist[3]
    Meta.sc = timelist[4]
    Meta.ms = timelist[5]
    Meta.microsec = timelist[6]
    Meta.fpaTemp = struct.unpack_from('>f',metaraw[476:480])[0]
    Meta.intTimeTicks = struct.unpack_from('>I',metaraw[142:146])[0]
    
    return [data,Meta]

def ReadSeq(seqfile):
    ## Pull camera (ch4 v o2) and sequence timestamp from filename.
    temp = seqfile.split('/')[-1]
    temp = temp.split('_camera_')
    Seq = Sequence()
    Seq.Camera = temp[0]
    Seq.SeqTime = dt.datetime.strptime(temp[1].strip('.seq'),'%Y_%m_%d_%H_%M_%S')
    
    ## Open file for binary read
    fin = open(seqfile,'rb')
    binhead = fin.read(8192)
    
    ## Grab meta data for sequence file.
    temp = struct.unpack('<9I',binhead[548:548+36])
    Seq.ImageWidth = temp[0]
    Seq.ImageHeight = temp[1]
    Seq.ImageBitDepth = temp[2]
    Seq.ImageBitDepthTrue = temp[3]
    Seq.ImageSizeBytes = temp[4]
    Seq.NumFrames = temp[6]
    Seq.TrueImageSize = temp[8]
    Seq.NumPixels = Seq.ImageWidth*Seq.ImageHeight
    
    ## Read raw frames
    rawframes = [fin.read(Seq.TrueImageSize) for i in range(Seq.NumFrames)]
    fin.close()
    
    ## Process each frame -- dropping filler bytes before passing raw
    # print('Reading {} frames'.format(Seq.NumFrames))
    frames = [ParseFrame(i,Seq.ImageHeight,Seq.ImageWidth,
                         rawframes[i][:Seq.ImageSizeBytes+6]) \
                    for i in range(Seq.NumFrames)]
    
    Data = np.array([dd[0] for dd in frames])
    Meta = [dd[1] for dd in frames]
    
    return(Data,Meta,Seq)
# above ReadSeq routines are from Jonathan Franklin
    
class Dark():
    """
    place holder for an object for dark frame
    """
    pass

class Granule():
    """
    place holder for an object for calibrated level 1b granule
    """
    pass

class MethaneAIR_L1(object):
    """
    level 0 to level 1b processor for MethaneAIR
    """    
    def __init__(self,whichBand,l0DataDir,l1DataDir,
                 badPixelMapPath,radCalPath,wavCalPath,windowTransmissionPath):
        """
        whichBand:
            CH4 or O2
        l0DataDir:
            directory of level 0 data
        l1DataDir:
            directory of level 1 data
        badPixelMapPath:
            path to the bad pixel map
        radCalPath:
            path to the radiometric calibration coefficients
        wavCalPath:
            path to wavelength calibration coefficients
        windowTransmissionPath:
            path to the window transmission
        """
        self.whichBand = whichBand
        self.logger = logging.getLogger(__name__)
        self.logger.info('creating an instance of level0->level1 for MethaneAIR '
                         +whichBand+ ' band')
        self.l0DataDir = l0DataDir
        if not os.path.isdir(l1DataDir):
            self.logger.warning(l1DataDir,' does not exist, creating one')
            os.mkdir(l1DataDir)
        self.l1DataDir = l1DataDir
        
        self.ncol = 1024
        self.nrow = 1280
        
        # bad pixel map
        self.logger.info('loading bad pixel map')
        self.badPixelMap = np.genfromtxt(badPixelMapPath,delimiter=',',dtype=np.int)
        self.logger.info('loading radiometric calibration coefficients')
        d = loadmat(radCalPath)['coef']
        
        # pad zero intercept and flip poly coeff order to be compatible with polyval
        radCalCoef = np.concatenate((np.zeros((*d.shape[0:2], 1)),d),axis=2)
        self.radCalCoef = np.flip(radCalCoef,axis=2)
        
        # wavelength calibration
        self.logger.info('loading wavelength calibration coefficients')
        spec_cal_fid = Dataset(wavCalPath)
        wavCalCoef = spec_cal_fid['pix2nm_polynomial_zero2four'][:]
        if wavCalCoef.shape[0] < wavCalCoef.shape[1]:
            self.logger.warning('you appear to be using an old version wavcal! tranposing...')
            self.wavCalCoef = wavCalCoef.T
        
        # wavelength calibration is independent of granules for now
        wavelength = np.array([np.polyval(self.wavCalCoef[i,::-1],np.arange(1,self.ncol+1))\
                           for i in range(self.nrow)])
        d = loadmat(windowTransmissionPath)
        # interpolate window transmission to detector wavelength
        f = interp1d(d['wavelength_nm'].squeeze(),d['transmission'].squeeze(),
                     fill_value='extrapolate')
        windowTransmission = f(wavelength)
        wavelength = np.ma.masked_array(wavelength,np.isnan(wavelength))
        windowTransmission = np.ma.masked_array(windowTransmission,
                                                np.isnan(windowTransmission))
        self.wavelength = wavelength
        self.windowTransmission = windowTransmission
                
    def F_stray_light_input(self,strayLightKernelPath,rowExtent,colExtent,
                            rowCenterMask=5,colCenterMask=7,nDeconvIter=1):
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
        nDeconvIter:
            number of iterations in the Van Crittert deconvolution
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
        strayLightFraction = np.nansum(strayLightKernel)
        
        self.strayLightKernel = strayLightKernel
        self.strayLightFraction = strayLightFraction
        self.nDeconvIter = nDeconvIter
        
    def F_grab_dark(self,darkFramePath):
        """
        function to load dark measurement level 0 file
        darkFramePath:
            path to the file
        output is a Dark object
        """
        self.logger.info('loading dark seq file '+darkFramePath)
        (Data,Meta,Seq) = ReadSeq(darkFramePath)
        Data = np.transpose(Data,(2,1,0))
#        badPixelMap3D = np.repeat(self.badPixelMap[...,np.newaxis],Seq.NumFrames,axis=2)
#        darkData = np.nanmean(np.ma.masked_where(badPixelMap3D!=0,Data),axis=2)
#        darkStd = np.nanstd(np.ma.masked_where(badPixelMap3D!=0,Data),axis=2)
        darkData = np.nanmean(Data,axis=2)
        darkStd = np.nanstd(Data,axis=2)
#        darkData[self.badPixelMap!=0] = np.nan
#        darkStd[self.badPixelMap!=0] = np.nan
        dark = Dark()
        dark.data = darkData
        dark.noise = darkStd
        dark.nFrame = Seq.NumFrames
        dark.frameTime = np.array([Meta[i].frameTime for i in range(Seq.NumFrames)])
        dark.seqDateTime = Seq.SeqTime
        dark.frameDateTime = np.array([Meta[i].timestamp for i in range(Seq.NumFrames)])
        return dark
        
#    def F_error_input(self,darkOffsetDN=1500,ePerDN=4.6,readOutError=40):
#        """
#        try realistic error estimation parameters
#        removed given Jenna Samra's email on 2020/6/9 10:47 pm
#        """
#        self.darkOffsetDN = darkOffsetDN
#        self.ePerDN = ePerDN
#        self.readOutError = readOutError
        
    def F_granule_processor(self,granulePath,dark,ePerDN=4.6,timeOnly=False):
        """
        working horse
        granulePath:
            path to a granule of level 0 data
        dark:
            a Dark object. Most useful part is dark.data, a nrow by ncol matrix 
            of dark frame for subtraction
        ePerDN:
            electrons per DN number
        timeOnly:
            if only save time stamps, no worries about other data
        output is a Granule object
        """
        if timeOnly:
            self.logger.info('loading seq file '+granulePath)
            (Data,Meta,Seq) = ReadSeq(granulePath)
            granuleFrameTime = np.array([Meta[i].frameTime for i in range(Seq.NumFrames)])
            granule = Granule()
            granule.nFrame = Seq.NumFrames
            granule.frameTime = granuleFrameTime
            granule.seqDateTime = Seq.SeqTime
            granule.frameDateTime = np.array([Meta[i].timestamp for i in range(Seq.NumFrames)])
            return granule
        darkData = dark.data
        darkStd = dark.noise
#        darkOffsetDN = self.darkOffsetDN
#        ePerDN = self.ePerDN
#        readOutError = self.readOutError
        self.logger.info('loading seq file '+granulePath)
        (Data,Meta,Seq) = ReadSeq(granulePath)
        Data = np.float32(np.transpose(Data,(2,1,0)))
        # estimate noise
        self.logger.info('estimate noise')
        Noise = np.sqrt((Data-darkData[...,np.newaxis])*ePerDN+(darkStd[...,np.newaxis]*ePerDN)**2)/ePerDN
#        Noise = np.sqrt((Data-darkOffsetDN)*ePerDN+readOutError**2)/ePerDN
        # remove dark
        Data = Data-darkData[...,np.newaxis]
        # mask bad pixels
        self.logger.info('mask bad pixels')
        badPixelMap3D = np.repeat(self.badPixelMap[...,np.newaxis],Seq.NumFrames,axis=2)
        Data[badPixelMap3D!=0] = np.nan
        Noise[badPixelMap3D!=0] = np.nan
#        Data = np.ma.masked_array(Data,
#                                  ((badPixelMap3D!=0) | (np.isnan(Data))) )
#        Noise = np.ma.masked_array(Noise,
#                                  ((badPixelMap3D!=0) | (np.isnan(Data))) )
        # normalize frame time, DN/s
        granuleFrameTime = np.array([Meta[i].frameTime for i in range(Seq.NumFrames)])
        Data = Data/granuleFrameTime[np.newaxis,np.newaxis,:]
        Noise = Noise/granuleFrameTime[np.newaxis,np.newaxis,:]
        # rad cal
        self.logger.info('radiometric calibration')
        for i in range(self.nrow):
            for j in range(self.ncol):
                if np.isnan(self.radCalCoef[i,j,0]):
                    Data[i,j,:] = np.nan
                    Noise[i,j,:] = np.nan
                    continue
                Data[i,j,:] = np.polyval(self.radCalCoef[i,j,:],Data[i,j,:])
                Noise[i,j,:] = np.polyval(self.radCalCoef[i,j,:],Noise[i,j,:])
#        newData = np.zeros(Data.shape)
#        newNoise = np.zeros(Data.shape)
#        for ipoly in range(self.radCalCoef.shape[-1]):
#            newData = newData+self.radCalCoef[...,ipoly][...,np.newaxis]*\
#            np.power(Data,ipoly+1)
#            newNoise = newNoise+self.radCalCoef[...,ipoly][...,np.newaxis]*\
#            np.power(Noise,ipoly+1)
#        Data = newData
#        Noise = newNoise
        
        if hasattr(self,'strayLightKernel'):
            self.logger.info('proceed with stray light correction')
            nDeconvIter = self.nDeconvIter
            strayLightKernel = self.strayLightKernel
            strayLightFraction = self.strayLightFraction
            from astropy.convolution import convolve_fft
            for iframe in range(Seq.NumFrames):
                tmpData = Data[:,:,iframe]
                for iDeconvIter in range(nDeconvIter):
                    tmpData = (Data[:,:,iframe]-convolve_fft(tmpData,strayLightKernel,normalize_kernel=False))\
                    /(1-strayLightFraction)
                Data[:,:,iframe] = tmpData
        else:
            self.logger.info('no data for stray light correction')
        # flip column order
        Data = Data[:,::-1,:]
        Noise = Noise[:,::-1,:]
        # window transmission correction
        Data = Data/self.windowTransmission[...,np.newaxis]
        Noise = Noise/self.windowTransmission[...,np.newaxis]
        granule = Granule()
        granule.data = Data
        granule.noise = Noise
        granule.nFrame = Seq.NumFrames
        granule.frameTime = granuleFrameTime
        granule.seqDateTime = Seq.SeqTime
        granule.frameDateTime = np.array([Meta[i].timestamp for i in range(Seq.NumFrames)])
        return granule
    
    def F_block_reduce_granule(self,granule,acrossTrackAggregation=6,
                               alongTrackAggregation=10,ifKeepTail=True):
        """
        block-reduce gradule by aggregating in across/along track or row/frame dimensions
        graule:
            outputs from F_granule_processor, a Granule object
        across/alongTrackAggregation:
            as indicated by name
        ifKeepTail:
            whether to aggregate the leftover frames after block reduce
        """
        if acrossTrackAggregation == 1 and alongTrackAggregation == 1:
            self.logger.warning('No aggregation needs to be done. Why are you calling this function?')
            return granule
        from astropy.nddata.utils import block_reduce
        newGranule = Granule()
        newGranule.seqDateTime = granule.seqDateTime
        nFootprint = np.floor(granule.data.shape[0]/acrossTrackAggregation).astype(np.int)
        nTailRow = (granule.data.shape[0]-nFootprint*acrossTrackAggregation).astype(np.int)
        self.logger.info('%d'%granule.data.shape[0]+' rows will be reduced to %d'%nFootprint+' footprints')
        self.logger.info('The last %d'%nTailRow+' rows will be thrown away')
        nFrameAggregated = np.floor(granule.data.shape[2]/alongTrackAggregation).astype(np.int)
        nTailFrame = (granule.data.shape[2]-nFrameAggregated*alongTrackAggregation).astype(np.int)
        self.logger.info('%d'%granule.data.shape[2]+' frames in the granule will be reduced to %d'%nFrameAggregated+' aggregated frames')
        
        if not ifKeepTail or nTailFrame == 0:
            self.logger.info('The last %d'%nTailFrame+' frames will be thrown away')
            newGranule.data = block_reduce(granule.data,
                                           (acrossTrackAggregation,1,alongTrackAggregation),
                                           func=np.nanmean)
            sumSquare = block_reduce(np.power(granule.noise,2),
                                    (acrossTrackAggregation,1,alongTrackAggregation),
                                    func=np.nansum)
            newGranule.noise = np.sqrt(sumSquare/acrossTrackAggregation/alongTrackAggregation)
            newGranule.frameDateTime = block_reduce(granule.frameDateTime,
                                                    alongTrackAggregation,
                                                    func=np.nanmax)
            newGranule.frameTime = block_reduce(granule.frameTime,
                                                alongTrackAggregation,
                                                func=np.nansum)
            newGranule.nFrame = nFrameAggregated
        else:
            self.logger.info('The last %d'%nTailFrame+' frame will be aggregated')
            data1 = block_reduce(granule.data,
                                (acrossTrackAggregation,1,alongTrackAggregation),
                                func=np.nanmean)
            data2 = block_reduce(granule.data[...,-nTailFrame:],
                                 (acrossTrackAggregation,1,nTailFrame),
                                 func=np.nanmean)
            self.logger.debug('data1 shape is %d'%data1.shape[0]+', %d'%data1.shape[1]+', %d'%data1.shape[2])
            self.logger.debug('data2 shape is %d'%data2.shape[0]+', %d'%data2.shape[1]+', %d'%data2.shape[2])
            newGranule.data = np.concatenate((data1,data2),axis=2)
            sumSquare = block_reduce(np.power(granule.noise,2),
                                     (acrossTrackAggregation,1,alongTrackAggregation),
                                     func=np.nansum)
            noise1 = np.sqrt(sumSquare/acrossTrackAggregation/alongTrackAggregation)
            sumSquare = block_reduce(np.power(granule.noise[...,-nTailFrame:],2),
                                     (acrossTrackAggregation,1,nTailFrame),
                                     func=np.nansum)
            noise2 = np.sqrt(sumSquare/acrossTrackAggregation/nTailFrame)
            newGranule.noise = np.concatenate((noise1,noise2),axis=2)
            frameDateTime1 = block_reduce(granule.frameDateTime,
                                          alongTrackAggregation,
                                          func=np.nanmax)
            frameDateTime2 = block_reduce(granule.frameDateTime[-nTailFrame:],
                                          nTailFrame,
                                          func=np.nanmax)
            newGranule.frameDateTime = np.concatenate((frameDateTime1,frameDateTime2))
            frameTime1 = block_reduce(granule.frameTime,
                                      alongTrackAggregation,
                                      func=np.nansum)
            frameTime2 = block_reduce(granule.frameTime[-nTailFrame:],
                                      nTailFrame,
                                      func=np.nansum)
            newGranule.frameTime = np.concatenate((frameTime1,frameTime2))
            newGranule.nFrame = nFrameAggregated+1
            if len(newGranule.frameTime) != newGranule.nFrame:
                self.logger.error('this should not happen!')
        return newGranule
    
    def F_cut_granule(self,granule,granuleSeconds=10):
        """
        cut a granule into a list of granules with shorter, regular-time intervals
        graule:
            outputs from F_granule_processor, a Granule object
        granuleSeconds:
            length of cut granule in s
        """
        if hasattr(granule,'data'):
            data = granule.data
            noise = granule.noise
        #nFrame = granule.nFrame
        frameTime = granule.frameTime
        seqDateTime = granule.seqDateTime
        frameDateTime = granule.frameDateTime
        minDateTime = np.min(frameDateTime)
        maxDateTime = np.max(frameDateTime)
        minSecond = (minDateTime-minDateTime.replace(hour=0,minute=0,second=0,microsecond=0)).total_seconds()
        startSecond = np.floor(minSecond/granuleSeconds)*granuleSeconds
        startDateTime = minDateTime.replace(hour=0,minute=0,second=0,microsecond=0)+dt.timedelta(seconds=startSecond)
        nGranule = np.floor((maxDateTime-startDateTime).total_seconds()/granuleSeconds)+1
        nGranule = np.int16(nGranule)
        secondList = np.arange(nGranule+1)*np.float(granuleSeconds)
        granuleEdgeDateTimeList = np.array([startDateTime+dt.timedelta(seconds=secondList[i])
        for i in range(nGranule+1)])
        
        self.logger.info('cutting granule into %d'%nGranule+' shorter granules with length %d'%granuleSeconds +' seconds')
        granuleList = np.ndarray(shape=(nGranule),dtype=np.object_)
        for i in range(nGranule):
            g0 = Granule()
            g0.seqDateTime = seqDateTime
            f = (frameDateTime >= granuleEdgeDateTimeList[i]) &\
            (frameDateTime < granuleEdgeDateTimeList[i+1])
            g0.nFrame = np.int16(np.sum(f))
            g0.frameDateTime = frameDateTime[f]
            g0.frameTime = frameTime[f]
            if hasattr(granule,'data'):
                g0.data = data[...,f]
                g0.noise = noise[...,f]
            granuleList[i] = g0
        return granuleList
    
    def F_save_L1B_time_only(self,granule,headerStr='MethaneAIR_L1B_CH4_timeonly_'):
        """
        save only the time stamps for calibrated data 
        granule:
            a Granule object generated by F_granule_processor or F_cut_granule
        headerStr:
            string that is different from the actual l1b files with real data 
        """
        from scipy.io import savemat
        l1FilePath = os.path.join(self.l1DataDir,headerStr
                                  +np.min(granule.frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                  +np.max(granule.frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                  +dt.datetime.now().strftime('%Y%m%dT%H%M%S')
                                  +'.mat')
        GEOS_5_tau = np.array([(granule.frameDateTime[i]-dt.datetime(1985,1,1,0,0,0)).total_seconds()/3600.
                               for i in range(granule.nFrame)])
        granuleYear = np.array([granule.frameDateTime[i].year for i in range(granule.nFrame)])
        granuleMonth = np.array([granule.frameDateTime[i].month for i in range(granule.nFrame)])
        granuleDay = np.array([granule.frameDateTime[i].day for i in range(granule.nFrame)])
        granuleHour = np.array([granule.frameDateTime[i].hour for i in range(granule.nFrame)])
        granuleMinute = np.array([granule.frameDateTime[i].minute for i in range(granule.nFrame)])
        granuleSecond = np.array([granule.frameDateTime[i].second for i in range(granule.nFrame)])
        granuleMicrosecond = np.array([granule.frameDateTime[i].microsecond for i in range(granule.nFrame)])
        self.logger.info('saving time only .mat L1B file '+l1FilePath)
        savemat(l1FilePath,{'GEOS_5_tau':GEOS_5_tau,
                            'granuleYear':granuleYear,
                            'granuleMonth':granuleMonth,
                            'granuleDay':granuleDay,
                            'granuleHour':granuleHour,
                            'granuleMinute':granuleMinute,
                            'granuleSecond':granuleSecond,
                            'granuleMicrosecond':granuleMicrosecond})
    
    def F_save_L1B_mat(self,granule,headerStr='MethaneAIR_L1B_CH4_',radianceOnly=True):
        """
        save calibrated data to level 1b file in .mat format for quick view
        granule:
            a Granule object generated by F_granule_processor or F_cut_granule
        headerStr:
            'MethaneAIR_L1B_CH4_' or 'MethaneAIR_L1B_O2_' or 'MethaneAIR_L1B_' 
        """
        from scipy.io import savemat
        l1FilePath = os.path.join(self.l1DataDir,headerStr
                                  +np.min(granule.frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                  +np.max(granule.frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                  +dt.datetime.now().strftime('%Y%m%dT%H%M%S')
                                  +'.mat')
        GEOS_5_tau = np.array([(granule.frameDateTime[i]-dt.datetime(1985,1,1,0,0,0)).total_seconds()/3600.
                               for i in range(granule.nFrame)])
        granuleYear = np.array([granule.frameDateTime[i].year for i in range(granule.nFrame)])
        granuleMonth = np.array([granule.frameDateTime[i].month for i in range(granule.nFrame)])
        granuleDay = np.array([granule.frameDateTime[i].day for i in range(granule.nFrame)])
        granuleHour = np.array([granule.frameDateTime[i].hour for i in range(granule.nFrame)])
        granuleMinute = np.array([granule.frameDateTime[i].minute for i in range(granule.nFrame)])
        granuleSecond = np.array([granule.frameDateTime[i].second for i in range(granule.nFrame)])
        granuleMicrosecond = np.array([granule.frameDateTime[i].microsecond for i in range(granule.nFrame)])
        wavelength = np.tile(self.wavelength[...,np.newaxis],granule.nFrame).transpose([0,2,1])
        data = granule.data.transpose([0,2,1])
        noise = granule.noise.transpose([0,2,1])
        # flip row order if O2 camera
        if self.whichBand == 'O2':
            wavelength = wavelength[::-1,...]
            data = data[::-1,...]
            noise = noise[::-1,...]
        self.logger.info('saving .mat L1B file '+l1FilePath)
        if radianceOnly:
            savemat(l1FilePath,{'radiance':np.asfortranarray(data).astype(np.float32),
                            'GEOS_5_tau':GEOS_5_tau,
                            'granuleYear':granuleYear,
                            'granuleMonth':granuleMonth,
                            'granuleDay':granuleDay,
                            'granuleHour':granuleHour,
                            'granuleMinute':granuleMinute,
                            'granuleSecond':granuleSecond,
                            'granuleMicrosecond':granuleMicrosecond})
            return
        savemat(l1FilePath,{'wavelength':np.asfortranarray(wavelength),
                            'radiance':np.asfortranarray(data),
                            'radiance_error':np.asfortranarray(noise),
                            'GEOS_5_tau':GEOS_5_tau,
                            'granuleYear':granuleYear,
                            'granuleMonth':granuleMonth,
                            'granuleDay':granuleDay,
                            'granuleHour':granuleHour,
                            'granuleMinute':granuleMinute,
                            'granuleSecond':granuleSecond,
                            'granuleMicrosecond':granuleMicrosecond})
        
        
    def F_save_L1B(self,granule,headerStr='MethaneAIR_L1B_CH4_'):
        """
        save calibrated data to level 1b file
        granule:
            a Granule object generated by F_granule_processor or F_cut_granule
        headerStr:
            'MethaneAIR_L1B_CH4_' or 'MethaneAIR_L1B_O2_' or 'MethaneAIR_L1B_' 
        """
        from pysplat import level1
        l1FilePath = os.path.join(self.l1DataDir,headerStr
                                  +np.min(granule.frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                  +np.max(granule.frameDateTime).strftime('%Y%m%dT%H%M%S')+'_'
                                  +dt.datetime.now().strftime('%Y%m%dT%H%M%S')
                                  +'.nc')
        GEOS_5_tau = np.array([(granule.frameDateTime[i]-dt.datetime(1985,1,1,0,0,0)).total_seconds()/3600.
                               for i in range(granule.nFrame)])
        wavelength = np.tile(self.wavelength[...,np.newaxis],granule.nFrame).transpose([0,2,1])
        data = granule.data.transpose([0,2,1])
        noise = granule.noise.transpose([0,2,1])
        # flip row order if O2 camera
        if self.whichBand == 'O2':
            wavelength = wavelength[::-1,...]
            data = data[::-1,...]
            noise = noise[::-1,...]
        l1 = level1(l1FilePath,
                    lon=np.zeros((self.nrow,granule.nFrame)),
                    lat=np.zeros((self.nrow,granule.nFrame)),
                    obsalt=np.zeros(granule.nFrame),
                    time=GEOS_5_tau)
        l1.add_radiance_band(wvl=wavelength,
                             rad=data,
                             rad_err=noise)
        l1.close()