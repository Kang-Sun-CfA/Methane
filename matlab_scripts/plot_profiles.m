clc
clear
% Plot vertical profiles of gases, aerosol and cloud from input.asc
% Written by Kang Sun on 2017/09/09

% \ for pc, / for unix
sfs = filesep;
if ispc
    % Kang's old think pad
    % where gc output is
    data_dir =  'd:\Research_CfA\CH4sat\';
    % source code containing matlab functions
    git_dir = 'c:\Users\Kang Sun\Documents\GitHub\Methane\';
    % matlab plotting function export_fig
    plot_function_dir = 'c:\Users\Kang Sun\Dropbox\matlab functions\export_fig\';
    % where to save your plot files
    plot_save_dir = [git_dir,'figures\'];
    profile_fn = [data_dir,sfs,'input.asc'];
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
    profile_fn = [data_dir,'../new_input/input.asc'];
end

if ~exist(plot_save_dir,'dir');
    mkdir(plot_save_dir);
end
addpath([git_dir,sfs,'matlab_functions',sfs])
addpath(plot_function_dir)
%%
clc
fid = fopen(profile_fn);
C = cell2mat(textscan(fid,repmat('%f',[1 26]),'delimiter',' ',...
    'multipledelimsasone','1','headerlines',23));
fclose(fid);
%%
P = C(:,2);
H = C(:,3);
T = C(:,4);
H2O = C(:,7);
CO2 = C(:,8);
N2O = C(:,10);
CO = C(:,11);
CH4 = C(:,12);
O2 = C(:,13);
OSU = C(:,18);
OBC = C(:,19);
OOC = C(:,20);
OSF = C(:,21);
OSC = C(:,22);
ODU = C(:,23);
OCW1 = C(:,24);
OCI1 = C(:,25);
%%
clc
close all
figure('unit','inch','color','w','position',[1 1 10 6])
ax1 = subplot(1,2,1);
h1 = plot(CH4,H,CO2,H,H2O,H,N2O,H,CO,H,'linewidth',1);
set(h1,'marker','*','markersize',8)
set(ax1,'xscale','log','xlim',[1e-9 0.01],'xtick',[1e-8 1e-6 1e-4 1e-2],...
    'xticklabel',{'10 ppb','1 ppm','100 ppm','1%'},'linewidth',1,'box','off')
ylabel('Altitude [km]')
xlabel('Volume mixing ratio')
hleg1 = legend(h1,'CH_4','CO_2','H_2O','N_2O','CO');
pos = get(hleg1,'position');
set(hleg1,'box','off','position',[pos(1)+0.03 pos(2:4)])

ax2 = subplot(1,2,2);
h2 = plot(OSU,P,OBC,P,OOC,P,OSC,P,OSF,P,ODU,P,OCW1,P,OCI1,P,'linewidth',1);
set(h2(2),'color','k')
set(h2,'marker','.','markersize',10)
set(h2(end),'marker','d','markersize',8)
set(h2(end-1),'marker','o','markersize',8)
set(ax2,'ydir','rev','yscale','log','ylim',[min(P) max(P)],...
    'yaxislocation','right','box','off','xscale','log','xlim',[1e-5 0.2],...
    'linewidth',1,'xtick',[1e-4 1e-3 1e-2 1e-1])
% set(ax1,'xscale','log','xlim',[1e-9 0.01],'xtick',[1e-8 1e-6 1e-4 1e-2],...
%     'xticklabel',{'10 ppb','1 ppm','100 ppm','1%'},'linewidth',1,'box','off')
ylabel('Pressure [hPa]')
xlabel('Optical depth')
lgstr = {'Sulfate','BC','OC','Sea salt coarse','Sea salt fine','Dust','Water cloud','Ice cloud'};
for i = 1:length(lgstr)
    lgstr{i} = [lgstr{i},', OD = ',num2str(sum(C(:,17+i)),2)];
end
hleg2 = legend(h2,lgstr);
set(hleg2,'box','off')
%%
export_fig([plot_save_dir,'profiles.pdf'],'-q150')
