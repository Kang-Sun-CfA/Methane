% Main program to run the GCtool for all possible spectral ranges
% Written by Kang Sun on 2017/08/31

% Abandoned on 2017/9/20, because matlab command "unix()" sometimes cannot
% run unix command when aerosol is on, possibly due to memory issue.
% However the command runs fine at terminals. not understood.
clc
clear
%% common data for all windows
% where gc tool is
GChome_dir = '/home/kangsun/GEOCAPE-TOOL/MASTER_MODULE/';
% matlab functions for this sensitivity study
matlabcode_dir = '/home/kangsun/CH4/Methane/matlab_functions/';
SZA = 70;% in degree
VZA = 45;
%rband = input('Which band? Choose from 760 1270 1670 2300: ','s');
%rband = sscanf(rband,'%f');
rband = [760 1270 1670 2300];
%% O2 A band, nickname 760
if ismember(760,rband)
inp660 = [];
inp660.SZA = SZA;
inp660.VZA = VZA;
% wavelength, in nm
inp660.vStart = 670; inp660.vEnd = 720; 
% molecules to include
inp660.gas_cell = {'O2','O4','H2O'};
% spectral interval for calculation, in nm
inp660.dv_calc = 0.01;
% BRDF parameters
inp660.K1 = 0.1; inp660.K2 = 0.1; inp660.K3 = 0.00001;
% O2 xsection, default is HITRAN line by line
% inp760.xsection_O2 = ':   O2  -1    6  hitran_lut/o2_lut_280-800nm_0p6fwhm_1e22vcd.nc';

cd(GChome_dir)
addpath(matlabcode_dir)

outp660 = F_write_GCtool_input(inp660);

inp760 = [];
inp760.SZA = SZA;
inp760.VZA = VZA;
% wavelength, in nm
inp760.vStart = 700; inp760.vEnd = 780; 
% molecules to include
inp760.gas_cell = {'O2','O4','H2O'};
% spectral interval for calculation, in nm
inp760.dv_calc = 0.01;
% BRDF parameters
inp760.K1 = 0.1; inp760.K2 = 0.1; inp760.K3 = 0.00001;
% O2 xsection, default is HITRAN line by line
% inp760.xsection_O2 = ':   O2  -1    6  hitran_lut/o2_lut_280-800nm_0p6fwhm_1e22vcd.nc';

cd(GChome_dir)
addpath(matlabcode_dir)

outp760 = F_write_GCtool_input(inp760);
end
%% O2 1-Delta band, nickname 1270
if ismember(1270,rband)
inp1270 = [];
inp1270.SZA = SZA;
inp1270.VZA = VZA;
% wavelength, in nm
inp1270.vStart = 1240; inp1270.vEnd = 1320; 
% molecules to include
inp1270.gas_cell = {'O2','O4','H2O','CO2'};
% spectral interval for calculation, in nm
inp1270.dv_calc = 0.01;
% BRDF parameters
inp1270.K1 = 0.1; inp1270.K2 = 0.1; inp1270.K3 = 0.00001;
% O2 xsection, default is HITRAN line by line
% inp1270.xsection_O2 = ':   O2  -1    6  hitran_lut/o2_lut_280-800nm_0p6fwhm_1e22vcd.nc';

cd(GChome_dir)
addpath(matlabcode_dir)

outp1270 = F_write_GCtool_input(inp1270);
inp1270 = [];
inp1270.SZA = SZA;
inp1270.VZA = VZA;
% wavelength, in nm
inp1270.vStart = 1310; inp1270.vEnd = 1400; 
% molecules to include
inp1270.gas_cell = {'O2','O4','H2O','CO2'};
% spectral interval for calculation, in nm
inp1270.dv_calc = 0.01;
% BRDF parameters
inp1270.K1 = 0.1; inp1270.K2 = 0.1; inp1270.K3 = 0.00001;
% O2 xsection, default is HITRAN line by line
% inp1270.xsection_O2 = ':   O2  -1    6  hitran_lut/o2_lut_280-800nm_0p6fwhm_1e22vcd.nc';

cd(GChome_dir)
addpath(matlabcode_dir)

outp1270 = F_write_GCtool_input(inp1270);
end
%% Methane 1.6 micron band, nickname 1670
if ismember(1670,rband)
inp1670 = [];
inp1670.SZA = SZA;
inp1670.VZA = VZA;
% wavelength, in nm
inp1670.vStart = 1550; inp1670.vEnd = 1640; 
% molecules to include
inp1670.gas_cell = {'CH4','H2O','CO2'};
% spectral interval for calculation, in nm
inp1670.dv_calc = 0.01;
% BRDF parameters
inp1670.K1 = 0.1; inp1670.K2 = 0.1; inp1670.K3 = 0.00001;
% O2 xsection, default is HITRAN line by line
% inp1670.xsection_O2 = ':   O2  -1    6  hitran_lut/o2_lut_280-800nm_0p6fwhm_1e22vcd.nc';

cd(GChome_dir)
addpath(matlabcode_dir)

outp1670 = F_write_GCtool_input(inp1670);
inp1670 = [];
inp1670.SZA = SZA;
inp1670.VZA = VZA;
% wavelength, in nm
inp1670.vStart = 1630; inp1670.vEnd = 1720; 
% molecules to include
inp1670.gas_cell = {'CH4','H2O','CO2'};
% spectral interval for calculation, in nm
inp1670.dv_calc = 0.01;
% BRDF parameters
inp1670.K1 = 0.1; inp1670.K2 = 0.1; inp1670.K3 = 0.00001;
% O2 xsection, default is HITRAN line by line
% inp1670.xsection_O2 = ':   O2  -1    6  hitran_lut/o2_lut_280-800nm_0p6fwhm_1e22vcd.nc';

cd(GChome_dir)
addpath(matlabcode_dir)

outp1670 = F_write_GCtool_input(inp1670);
end
%% Methane 2.3 micron band, nickname 2300
if ismember(2300,rband)
inp2300 = [];
inp2300.SZA = SZA;
inp2300.VZA = VZA;
% wavelength, in nm
inp2300.vStart = 2230; inp2300.vEnd = 2320; 
% molecules to include
inp2300.gas_cell = {'CH4','H2O','N2O','CO'};
% spectral interval for calculation, in nm
inp2300.dv_calc = 0.01;
% BRDF parameters
inp2300.K1 = 0.1; inp2300.K2 = 0.1; inp2300.K3 = 0.00001;
% O2 xsection, default is HITRAN line by line
% inp2300.xsection_O2 = ':   O2  -1    6  hitran_lut/o2_lut_280-800nm_0p6fwhm_1e22vcd.nc';

cd(GChome_dir)
addpath(matlabcode_dir)

outp2300 = F_write_GCtool_input(inp2300);
inp2300 = [];
inp2300.SZA = SZA;
inp2300.VZA = VZA;
% wavelength, in nm
inp2300.vStart = 2310; inp2300.vEnd = 2400; 
% molecules to include
inp2300.gas_cell = {'CH4','H2O','N2O','CO'};
% spectral interval for calculation, in nm
inp2300.dv_calc = 0.01;
% BRDF parameters
inp2300.K1 = 0.1; inp2300.K2 = 0.1; inp2300.K3 = 0.00001;
% O2 xsection, default is HITRAN line by line
% inp2300.xsection_O2 = ':   O2  -1    6  hitran_lut/o2_lut_280-800nm_0p6fwhm_1e22vcd.nc';

cd(GChome_dir)
addpath(matlabcode_dir)

outp2300 = F_write_GCtool_input(inp2300);
end
