% Matlab script template to run sensitivity study
% Written by Kang Sun on 2017/09/11
clc
clear
close all
% \ for pc, / for unix
sfs = filesep;
% define directories
if ispc
    % Kang's old thinkpad
    % where gc output is
    data_dir =  'd:\Research_CfA\CH4sat\outp\';
    % source code containing matlab functions
    git_dir = 'c:\Users\Kang Sun\Documents\GitHub\Methane\';
    % matlab plotting function export_fig
    plot_function_dir = 'c:\Users\Kang Sun\Dropbox\matlab functions\export_fig\';
    % where to save your plot files
    plot_save_dir = [git_dir,'figures\'];
else
    % SAO's powerful unix workstations
    % where gc output is
    data_dir =  '/home/kangsun/GEOCAPE-TOOL/MASTER_MODULE/';
    % source code containing matlab functions
    git_dir = '/home/kangsun/CH4/Methane/';
    % matlab plotting function export_fig
    plot_function_dir = '/home/kangsun/matlab functions/export_fig/';
    % where to save your plot files
    plot_save_dir = [git_dir,'figures/'];
end

if ~exist(plot_save_dir,'dir');
    mkdir(plot_save_dir);
end
addpath([git_dir,sfs,'matlab_functions',sfs])
addpath(plot_function_dir)
cd(data_dir)

% viewing geometry scenarios
SZA_array = [70 45];% in degree
VZA_array = [45 30];

% surface albedo scenarios
ReflSpectra_array = {'conifer_ASTER.dat','sand_ASTER.dat','tapwater_ASTER_smooth.dat'};
ReflName_array = {'conifer','sand','tapwater'};

% select/loop scenarios
iRefl = 1;
iangle = 1;

SZA = SZA_array(iangle);
VZA = VZA_array(iangle);

ReflName = ReflName_array{iRefl};

% merge gc outputs, the outputs need to have the same resolution, and the
% sampling intervals have to be aligned
inp.fn1 = ['O2_670-720_0.01_',num2str(SZA),'_',num2str(VZA),...
    '_',ReflName,'GC_upwelling_output.nc'];
inp.fn2 = ['O2_700-780_0.01_',num2str(SZA),'_',num2str(VZA),...
    '_',ReflName,'GC_upwelling_output.nc'];
outp_O2_1 = F_merge_gc_output(inp);

inp.fn1 = ['O2_1240-1320_0.01_',num2str(SZA),'_',num2str(VZA),...
    '_',ReflName,'GC_upwelling_output.nc'];
inp.fn2 = ['O2_1310-1400_0.01_',num2str(SZA),'_',num2str(VZA),...
    '_',ReflName,'GC_upwelling_output.nc'];
outp_O2_2 = F_merge_gc_output(inp);

inp.fn1 = ['CH4_1550-1640_0.01_',num2str(SZA),'_',num2str(VZA),...
    '_',ReflName,'GC_upwelling_output.nc'];
inp.fn2 = ['CH4_1630-1720_0.01_',num2str(SZA),'_',num2str(VZA),...
    '_',ReflName,'GC_upwelling_output.nc'];
outp_CH4_1 = F_merge_gc_output(inp);

inp.fn1 = ['CH4_2230-2320_0.01_',num2str(SZA),'_',num2str(VZA),...
    '_',ReflName,'GC_upwelling_output.nc'];
inp.fn2 = ['CH4_2310-2400_0.01_',num2str(SZA),'_',num2str(VZA),...
    '_',ReflName,'GC_upwelling_output.nc'];
outp_CH4_2 = F_merge_gc_output(inp);
%%
clc
% define inputs that will not change in this study
inp = [];
% number of spectral windows
inp.nwin  =  2; 
% number of detector pixels
inp.npixel = [2000, 2000];
% place holder, will change for different designs
inp.wmins = [nan; nan]; 
inp.wmaxs = [nan; nan];
% number of spectral samples per fwhm
inp.nsamp = [3;    3];
% order of albedo term. 1 means only fit DC, 2 adds a slope
inp.nalb  = [1;    1];
% equivalent SNR
inp.snre  = [260;  260];
% SNRe-defining radiance
inp.snrdefine_rad = [6e11, 6e11];
% SNRe-defining spectral sampling, in nm
inp.snrdefine_dlambda = [0.1 0.1];
% gc output in a wider spectral range. IT HAS TO COVER ALL WINDOWS!!!
inp.gc_output = {outp_O2_1,outp_O2_2,outp_CH4_1,outp_CH4_2};
% fwhm of high-resolution gc runs
inp.gc_fwhm = 0.02;

% input items for F_gas_sens
% place holder, will change for different designs 
inp.included_gases = {};
% place holder, will change for different designs
inp.inc_prof = logical([]);
% if include the following variables in the sensitivity study
inp.inc_alb    = 1;
inp.inc_t      = 0;
inp.inc_aod    = 1;
inp.inc_assa   = 0;
inp.inc_cod    = 0;
inp.inc_cssa   = 0;
inp.inc_cfrac  = 0;
inp.inc_sfcprs = 0;

% a priori value relative to gas columns
inp.gascol_aperr_scale     = 10;
% a priori value relative to gas profiles, LT:0-2; UT:2-17; ST:>17 km
inp.gasprof_aperr_scale_LT = 0.1;
inp.gasprof_aperr_scale_UT = 0.05;
inp.gasprof_aperr_scale_ST = 0.02;

inp.t_aperr       = 5.0;
inp.aod_aperr     = 0.5;  %change from 1.0 to 0.5
inp.assa_aperr    = 0.05; %changed from 0.2 to 0.1 then 0.05 on April 2014
inp.cod_aperr     = 2.0;  %changed from 5.0 to 2.0 in April 2014
inp.cssa_aperr    = 0.01;
inp.cfrac_aperr   = 0.1;
inp.sfcprs_aperr  = 20.0;
inp.albs_aperr   = [0.1, 0.1/5., 0.1/25., 0.1/125., 0.1/625., 0.02/625.]; %2.0

% define test cases. works for A-C cases so far.
casenames = {'A1','A2','B1','B2','C1'};
% starting and ending wavelength for each windows for each design, in nm
wmins_array = {[1249; 1597],[1249; 1627],[1249; 2240],[1249; 2240],[1597; 2240]};
wmaxs_array = {[1380; 1701],[1288; 1701],[1380; 2380],[1288; 2340],[1701; 2380]};
% gas species included in each design
gases_array = {{'CH4','O2','O4','H2O','CO2'},...
    {'CH4','O2','O4','H2O','CO2'},...
    {'CH4','O2','O4','H2O','CO2','N2O','CO'},...
    {'CH4','O2','O4','H2O','N2O'},...
    {'CH4','H2O','CO2','N2O','CO'}};
ncase = length(wmins_array);

outp = cell(ncase,1);
for icase = 1:ncase
    disp(['Runing case ',num2str(icase)])
    % define inputs fields that changes between test cases
    inp.wmins = wmins_array{icase};
    inp.wmaxs = wmaxs_array{icase};
    inp.included_gases = gases_array{icase};
    inp.inc_prof = false(length(inp.included_gases),1);
    inp.inc_prof(1) = true;% only retrieve CH4 profiles
    outp{icase} = F_wrapper(inp);
    figure(outp{icase}.fig_overview)
    % you may not run this line without writing permit
%     export_fig([plot_save_dir,casenames{icase},'_jac.pdf'])
%     close
end