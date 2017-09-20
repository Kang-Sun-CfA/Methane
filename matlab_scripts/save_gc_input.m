% Programmatically save multiple inputs to the GC tool. Has to be
% compatible with input_template.gc
% Output a shell script to run all the generated GC tool input files

% Modified from run_GCtool.m by Kang Sun on 2017/09/20
clc
clear
close all
% \ for pc, / for unix
sfs = filesep;
if ispc
    % Kang's old thinkpad
    % where gc output is
    data_dir =  'd:\Research_CfA\CH4sat\';
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
%% inputs
SZA_array = [70 45];% in degree
VZA_array = [45 30];

ReflSpectra_array = {'conifer_ASTER.dat','sand_ASTER.dat','tapwater_ASTER_smooth.dat'};
ReflName_array = {'conifer','sand','tapwater'};

windowlist = [];
windowlist(1).vStart = 670;
windowlist(1).vEnd = 720;
windowlist(1).gas_cell = {'O2','O4','H2O'};

windowlist(2).vStart = 700;
windowlist(2).vEnd = 780;
windowlist(2).gas_cell = {'O2','O4','H2O'};

windowlist(3).vStart = 1240;
windowlist(3).vEnd = 1320;
windowlist(3).gas_cell = {'O2','O4','H2O','CO2'};

windowlist(4).vStart = 1310;
windowlist(4).vEnd = 1400;
windowlist(4).gas_cell = {'O2','O4','H2O','CO2'};

windowlist(5).vStart = 1550;
windowlist(5).vEnd = 1640;
windowlist(5).gas_cell = {'CH4','H2O','CO2'};

windowlist(6).vStart = 1630;
windowlist(6).vEnd = 1720;
windowlist(6).gas_cell = {'CH4','H2O','CO2'};

windowlist(7).vStart = 2230;
windowlist(7).vEnd = 2320;
windowlist(7).gas_cell = {'CH4','H2O','N2O','CO'};

windowlist(8).vStart = 2310;
windowlist(8).vEnd = 2400;
windowlist(8).gas_cell = {'CH4','H2O','N2O','CO'};
%% loops
frun = fopen('run.sh','w');

for iangle = 1:1%length(SZA_array)
    SZA = SZA_array(iangle);
    VZA = VZA_array(iangle);
    for ialb = 1:length(ReflSpectra_array)
        ReflSpectra = ReflSpectra_array{ialb};
        ReflName = ReflName_array{ialb};
        for iwin = 1:length(windowlist)
            inp = [];
            inp.SZA = SZA;
            inp.VZA = VZA;
            inp.vStart = windowlist(iwin).vStart;
            inp.vEnd = windowlist(iwin).vEnd;
            inp.gas_cell = windowlist(iwin).gas_cell;
            inp.dv_calc = 0.01;
            inp.fn_extra = ReflName;
            outp = F_write_GCtool_input(inp);
            run_str = ['./geocape_tool.exe ',outp.fnout];
            fprintf(frun,'%s',run_str);
            fprintf(frun,'\n');
        end
    end
end
fclose(frun);
