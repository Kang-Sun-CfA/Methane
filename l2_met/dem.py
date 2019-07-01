# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 20:11:12 2019

@author: ovpr.kangsun
"""

import numpy as np
# conda install -c conda-forge opencv 
#import cv2
# conda install -c scitools/label/archive shapely
from shapely.geometry import Polygon
from netCDF4 import Dataset
import os
import glob
import logging

def F_geotiff2nc(dem_dir,if_delete_tif=False):
    """
    converting geotiff file to netcdf. GMTED2010 is supported
    dem_dir:
        directory containing DEM data
    required packages:
        gdal, netcdf4
    """
    import gdal
    tif_flist = glob.glob(os.path.join(dem_dir,'*mea075.tif'))
    for fn in tif_flist:
        #print(fn)
        f = gdal.Open(fn, gdal.GA_ReadOnly)
        f_width = f.RasterXSize
        f_height = f.RasterYSize
        #f_nbands = f.RasterCount
        
        f_data = f.ReadAsArray()
        f_trans = f.GetGeoTransform()
        
        xres = f_trans[1]
        yres = f_trans[5]
        xorig = f_trans[0]
        yorig = f_trans[3]
        xgrid = xorig+np.arange(0,f_width)*xres
        ygrid = yorig+np.arange(0,f_height)*yres
        #plt.pcolormesh(xgrid[0:3000],ygrid[0:5000],np.float16(f_data[0:5000,0:3000]))
        f.FlushCache
        nc_fn = os.path.splitext(fn)[0]+'.nc'
        nc = Dataset(os.path.join(dem_dir,nc_fn),'w',format='NETCDF4_CLASSIC')
        nc.description = 'netcdf file saved from '+fn
        nc.createDimension('lon',len(xgrid))
        nc.createDimension('lat',len(ygrid))
        lon = nc.createVariable('xgrid','f8',('lon'))
        lat = nc.createVariable('ygrid','f8',('lat'))
        elev = nc.createVariable('z','int16',('lat','lon'))
        lon[:] = xgrid
        lat[:] = ygrid
        elev[:] = f_data
        nc.close()

class dem(object):
    
    def __init__(self,dem_dir,lonc,latc,lonr=None,latr=None,
                 lon_margin=0.1,lat_margin=0.1):
        """
        initiate the dem object
        dem_dir:
            directory containing the DEM data
        lonc, latc:
            centroid coordinates of sounding, -180 to 180-
        lonr, latr:
            polygon vertices of sounding pixel
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info('creating an instance of dem logger')
        self.dem_dir = dem_dir
        sounding_data = {}
        sounding_data['latc'] = latc
        sounding_data['lonc'] = lonc
        south = np.min(latc)
        north = np.max(latc)
        if np.all(lonc>=0.) or np.all(lonc<=0.):
            west = np.min(lonc)
            east = np.max(lonc)
        else:
            west = np.min(lonc(lonc>0))
            east = np.max(lonc(lonc<0))
        if east < west:
            self.logger.error('+/-180 longitude acrossing is not supported!')
            east = east+360
        self.north = north+lat_margin
        self.south = south-lat_margin
        self.west = west-lon_margin
        self.east = east+lon_margin
        self.sounding_data = sounding_data
    
    def F_load_gmted(self):
        """
        load GMTED data based on the latlon bounds of sounding locations
        """
        west = self.west
        east = self.east
        south = self.south
        north = self.north
        lon_bound = np.linspace(-180,180,13)
        lat_bound = np.linspace(-90.,90,10)
        lat_idx_start = np.argmax(lat_bound-south >= 0)-1
        lat_idx_end = np.argmax(lat_bound-north >= 0)-1
        lon_idx_start = np.argmax(lon_bound-west >= 0)-1
        lon_idx_end = np.argmax(lon_bound-east >= 0)-1
        dem_flist = []
        xgrid_all = []
        ygrid_all = []
        z_all = [[]]
        for lat_idx in range(lat_idx_start,lat_idx_end+1):
            for lon_idx in range(lon_idx_start,lon_idx_end+1):
                if lat_bound[lat_idx] < 0:
                    lat_str = '%02d'%(np.abs(lat_bound[lat_idx]))+'S'
                else:
                    lat_str = '%02d'%(np.abs(lat_bound[lat_idx]))+'N'
                if lon_bound[lon_idx] < 0:
                    lon_str = '%03d'%(np.abs(lon_bound[lon_idx]))+'W'
                else:
                    lon_str = '%03d'%(np.abs(lon_bound[lon_idx]))+'E'
                dem_file = glob.glob(os.path.join(self.dem_dir,lat_str+lon_str+'*mea075.nc'))
                nc = Dataset(dem_file[0])
                xgrid = np.sort(nc.variables['xgrid'])
                if lat_idx == lat_idx_start:
                    xgrid_all = np.concatenate((xgrid_all,xgrid))
                ygrid = np.sort(nc.variables['ygrid'])
                if lon_idx == lon_idx_start:
                    ygrid_all = np.concatenate((ygrid_all,ygrid))
                z = nc.variables['z'][np.argsort(nc.variables['ygrid']),
                                np.argsort(nc.variables['xgrid'])]
                if lon_idx == lon_idx_start:
                    z_row = z
                else:
                    z_row = np.hstack((z_row,z))
                dem_flist = dem_flist+dem_file
            if lat_idx == lat_idx_start:
                z_all = z_row
            else:
                z_all = np.vstack((z_all,z_row))
        self.dem_flist = dem_flist
        xint = (xgrid_all >= west) & (xgrid_all <= east)
        yint = (ygrid_all >= south) & (ygrid_all <= north)
        self.xgrid = xgrid_all[xint]
        self.ygrid = ygrid_all[yint]
        self.z = z_all[np.ix_(yint,xint)]
    
    def F_interp_dem(self):
        """
        sample dem at sounding locations by simple 2d linear interpolation
        """
        from scipy.interpolate import RegularGridInterpolator
        xgrid = self.xgrid
        ygrid = self.ygrid
        z = self.z
        sounding_data = self.sounding_data
        f_interp = RegularGridInterpolator((xgrid,ygrid),z.T)
        sounding_data['z'] = f_interp((sounding_data['lonc'],sounding_data['latc']))
        self.sounding_data = sounding_data
            