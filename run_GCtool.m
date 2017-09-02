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
% %% O2 A band, nickname 760
% inp760 = [];
% inp760.SZA = SZA;
% inp760.VZA = VZA;
% % wavelength, in nm
% inp760.vStart = 687; inp760.vEnd = 772; 
% % molecules to include
% inp760.gas_cell = {'O2','O4','H2O'};
% % spectral interval for calculation, in nm
% inp760.dv_calc = 0.01;
% % BRDF parameters
% inp760.K1 = 0.1; inp760.K2 = 0.1; inp760.K3 = 0.00001;
% % O2 xsection, default is HITRAN line by line
% % inp760.xsection_O2 = ':   O2  -1    6  hitran_lut/o2_lut_280-800nm_0p6fwhm_1e22vcd.nc';
% 
% cd(GChome_dir)
% addpath(matlabcode_dir)
% 
% outp760 = F_run_GCtool(inp760);
% %% O2 1-Delta band, nickname 1270
% inp1270 = [];
% inp1270.SZA = SZA;
% inp1270.VZA = VZA;
% % wavelength, in nm
% inp1270.vStart = 1240; inp1270.vEnd = 1300; 
% % molecules to include
% inp1270.gas_cell = {'O2','O4','H2O','CO2'};
% % spectral interval for calculation, in nm
% inp1270.dv_calc = 0.01;
% % BRDF parameters
% inp1270.K1 = 0.1; inp1270.K2 = 0.1; inp1270.K3 = 0.00001;
% % O2 xsection, default is HITRAN line by line
% % inp1270.xsection_O2 = ':   O2  -1    6  hitran_lut/o2_lut_280-800nm_0p6fwhm_1e22vcd.nc';
% 
% cd(GChome_dir)
% addpath(matlabcode_dir)
% 
% outp1270 = F_run_GCtool(inp1270);
%% Methane 1.6 micron band, nickname 1670
inp1670 = [];
inp1670.SZA = SZA;
inp1670.VZA = VZA;
% wavelength, in nm
inp1670.vStart = 1590; inp1670.vEnd = 1600;%1709; 
% molecules to include
inp1670.gas_cell = {'CH4','H2O','CO2'};
% spectral interval for calculation, in nm
inp1670.dv_calc = 0.012;
% BRDF parameters
inp1670.K1 = 0.1; inp1670.K2 = 0.1; inp1670.K3 = 0.00001;
% O2 xsection, default is HITRAN line by line
% inp1670.xsection_O2 = ':   O2  -1    6  hitran_lut/o2_lut_280-800nm_0p6fwhm_1e22vcd.nc';

cd(GChome_dir)
addpath(matlabcode_dir)

outp1670 = F_run_GCtool(inp1670);
%% Methane 2.3 micron band, nickname 2300
%inp2300 = [];
%inp2300.SZA = SZA;
%inp2300.VZA = VZA;
% wavelength, in nm
%inp2300.vStart = 2240; inp2300.vEnd = 2380; 
% molecules to include
inp2300.gas_cell = {'CH4','H2O','CO2','N2O','CO'};
% spectral interval for calculation, in nm
inp2300.dv_calc = 0.015;
% BRDF parameters
inp2300.K1 = 0.1; inp2300.K2 = 0.1; inp2300.K3 = 0.00001;
% O2 xsection, default is HITRAN line by line
% inp2300.xsection_O2 = ':   O2  -1    6  hitran_lut/o2_lut_280-800nm_0p6fwhm_1e22vcd.nc';

cd(GChome_dir)
addpath(matlabcode_dir)

%outp2300 = F_run_GCtool(inp2300);
