% Programmatically save multiple inputs to the GC tool. Has to be
% compatible with input_template.gc
% Output a shell script to run all the generated GC tool input files

% Modified from run_GCtool.m by Kang Sun on 2017/09/20
% Modified from save_gc_input.m on 2017/10/11 to loop over different
% aerosol types
% Modified from save_gc_input_aerosols.m on 2018/03/04 to simulate clear
% sky only and use updated window options
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
SZA_array = [70 45 0];% in degree
VZA_array = [50 25 0];

ReflSpectra_array = {'conifer_ASTER.dat','sand_ASTER.dat','tapwater_ASTER_smooth.dat'};
ReflName_array = {'conifer','sand','tapwater'};

% aername_array = {'SU','BC','OC','SF','SC','DU'};

windowlist = [];

windowlist(1).vStart = 1240;
windowlist(1).vEnd = 1330;
windowlist(1).gas_cell = {'O2','O4','H2O','CO2'};

windowlist(2).vStart = 1600;
windowlist(2).vEnd = 1690;
windowlist(2).gas_cell = {'CH4','H2O','CO2'};

%% loops
frun = fopen('run.sh','w');
fprintf(frun,'#bin!sh\n');
for iangle = 1:length(SZA_array)
    SZA = SZA_array(iangle);
    VZA = VZA_array(iangle);
    for ialb = 1:length(ReflSpectra_array)
        ReflSpectra = ReflSpectra_array{ialb};
        ReflName = ReflName_array{ialb};
        for iaer = 1:1%length(aername_array)
%             input_profile = ['../new_input/input_',aername_array{iaer},'.asc'];
        for iwin = 1:length(windowlist)
            inp = [];
            inp.SZA = SZA;
            inp.VZA = VZA;
%             inp.input_profile = input_profile;
            inp.vStart = windowlist(iwin).vStart;
            inp.vEnd = windowlist(iwin).vEnd;
            inp.gas_cell = windowlist(iwin).gas_cell;
            inp.dv_calc = 0.01;
            inp.fn_extra = ReflName;
            inp.ReflSpectra = ReflSpectra;
            outp = F_write_GCtool_input(inp);
            run_str = ['./geocape_tool.exe ',outp.fnout];
            fprintf(frun,'%s',run_str);
            fprintf(frun,'\n');
        end
        end
    end
end
fclose(frun);
