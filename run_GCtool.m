% Main program to run the GCtool for all possible spectral ranges
% Written by Kang Sun on 2017/08/31
clc
clear
%% common data for all windows
% where gc tool is
GChome_dir = '/home/kangsun/GEOCAPE-TOOL/MASTER_MODULE/';
% matlab functions for this sensitivity study
matlabcode_dir = '/home/kangsun/CH4/Methane/';
SZA = 70;% in degree
VZA = 45;
%% O2 A band, nickname 760
inp760 = [];
inp760.SZA = SZA;
inp760.VZA = VZA;
% wavelength, in nm
inp760.vStart = 700; inp760.vEnd = 780; 
% molecules to include
inp760.gas_cell = {'O2','O4','H2O'};
% spectral interval for calculation, in nm
inp760.dv_calc = 0.01;
inp760.K1 = 0.1; inp760.K2 = 0.1; inp760.K3 = 0.00001;

cd(GChome_dir)
addpath(matlabcode_dir)

outp760 = F_run_GCtool(inp760);