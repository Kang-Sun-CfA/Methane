#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:46:01 2020

@author: kangsun
"""
import datetime
import os
merra_dir = '/mnt/Data2/MERRA/'
ges_disc_txt = '/home/kangsun/TROPOMI/downloads/subset_M2T1NXSLV_5.12.4_20200123_205409.txt'
landmark_str = 'tavg1_2d_slv_Nx.'
start_datetime = datetime.datetime(2005,7,1)
end_datetime = datetime.datetime(2005,7,5)
cwd = os.getcwd()
fid = open(ges_disc_txt,'r')
line = fid.readline()
while line:
    tmp_loc = line.find(landmark_str)
    if tmp_loc == -1:
        line = fid.readline()
        continue
    start_index = tmp_loc+len(landmark_str)
    file_datetime = datetime.datetime.strptime(line[start_index:start_index+8],'%Y%m%d')
    if (file_datetime >= start_datetime) and (file_datetime <= end_datetime):
        file_dir = os.path.join(merra_dir,file_datetime.strftime('Y%Y'),file_datetime.strftime('M%m'),file_datetime.strftime('D%d'))
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
            os.chdir(file_dir)
        # this is very stupid but cannot get wget working with single url
        tmp_fid = open('tmp.txt','w')
        tmp_fid.write(line)
        tmp_fid.close()
        run_str = 'wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --auth-no-challenge=on --keep-session-cookies --content-disposition -i tmp.txt'
        print('Downloading MERRA-2 on '+file_datetime.strftime('%Y%m%d'))
        os.system(run_str)
        os.remove('tmp.txt')
        line = fid.readline()
    else:
        line = fid.readline()
    if (file_datetime > end_datetime):
        break
fid.close()
os.chdir(cwd)