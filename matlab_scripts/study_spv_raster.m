% code for sensitivty study using spv dated 2018/10/23 for raster study
% written by Kang Sun on 2018/10/24
clc
clear
close all
machine = 'UB';
machine = 'CF';
whichsensor = 'headwall';
whichsensor = 'ball';
dx0 = 1.4; % in km
% define directories
switch machine
    case 'UB'
        % ub workstations
        % where gc output is
        data_dir =  '/mnt/Data2/gctool_data/spv_outp/';
        % source code containing matlab functions
        git_dir = '/home/kangsun/CH4/Methane/';
        % matlab plotting function export_fig
        plot_function_dir = '/home/kangsun/matlab/export_fig/';
        % where to save your plot files
        plot_save_dir = ['/mnt/Data2/gctool_data/figures/'];
        airglowspec_path = '/home/kangsun/CH4/airglowspec.mat';
    case 'CF'
        % SAO's powerful unix workstations
        % where gc output is
        data_dir =  '/data/tempo1/Shared/kangsun/spv/outp/';
        % source code containing matlab functions
        git_dir = '/home/kangsun/CH4/Methane/';
        % matlab plotting function export_fig
        plot_function_dir = '/home/kangsun/matlab functions/export_fig/';
        % where to save your plot files
        plot_save_dir = [git_dir,'figures/'];
        airglowspec_path = '/home/kangsun/CH4/airglowspec.mat';
end

if ~exist(plot_save_dir,'dir')
    mkdir(plot_save_dir);
end
addpath([git_dir,'matlab_functions/'])
addpath(plot_function_dir)
addpath([plot_function_dir,'/..'])

cd(data_dir)

SZA_vec = [35:10:65];
VZA_vec = 5:10:55;
alb_vec_o2 = 0.05:0.1:0.65;
alb_vec_ch4 = 0.05:0.1:0.75;
%%
if_lnR = false;
clc
inp = [];
inp.nwin = 2;
inp.wmins = [1246 1606];
inp.wmaxs = [1293 1689];
inp.nsamp = [3 3];
inp.nalb = [2 2];

inp.included_gases = {'CH4','H2O','CO2'};
inp.inc_prof = logical([1 0 0]);
inp.if_vmr = logical([1 1 1]);
inp.inc_sfcprs = 1;
inp.inc_alb    = 1;
inp.inc_t      = 1;

inp.inc_airglow= 1;

inp.included_aerosols = {};
inp.inc_aod    = 0;
inp.inc_aod_pkh= 0;
inp.inc_aod_hfw= 0;

inp.inc_assa   = 0;
inp.inc_cod    = 0;
inp.inc_cssa   = 0;
inp.inc_cfrac  = 0;

% a priori value relative to gas columns
inp.gascol_aperr_scale     = [10 10 10];
% a priori value relative to gas profiles, LT:0-2; UT:2-17; ST:>17 km
inp.gasprof_aperr_scale_LT = [0.05 0.8];
inp.gasprof_aperr_scale_UT = [0.03 0.5];
inp.gasprof_aperr_scale_ST = [0.01 0.5];

inp.t_aperr       = 5.0;  % in K
inp.aod_aperr     = [0.5 0.5 0.5 0.5 0.5];  % in optical depth
inp.aod_pkh_aperr = [1 1 1 1 1];    % in km
inp.aod_hfw_aperr = [0.1 0.1 0.1 0.1 0.1];    % in km
inp.assa_aperr    = 0.05; %changed from 0.2 to 0.1 then 0.05 on April 2014
inp.cod_aperr     = 2.0;  %changed from 5.0 to 2.0 in April 2014
inp.cssa_aperr    = 0.01;
inp.cfrac_aperr   = 0.1;
inp.sfcprs_aperr  = 4.0;
inp.airglow_aperr = 0.05;
inp.albsnorm = 100; % alb scale factor in percent
inp.albs_aperr   = [0.1, 0.1/5., 0.1/25., 0.1/125., 0.1/625., 0.02/625.]; %2.0

switch whichsensor
    case 'headwall'
        % headwall mct
        dx = 0.092;
        dt = 1/17.5;
        D = 4.62;
        inp.fwhm = [0.225 0.3];
    case 'ball'
        % ball
        dx = 0.126;
        dt = 1/17.5;
        D = 4.37;
        inp.fwhm = [0.15 0.189];
end
inpn_O2 = [];
% readout noise, e per pixel
inpn_O2.Nr = 50;
% dark current, e per s per pixel.
inpn_O2.Nd_per_s = 10000;
% orbit height, km
inpn_O2.H = 617;
% integration time, s. 1/7 means integration through 1 km
inpn_O2.dt = dt;
inpn_O2.eta = 0.4;
% ground fov for single pixel, across track, km
inpn_O2.dx = dx;
% ground fov for single pixel, along track, km
inpn_O2.dy = inpn_O2.dx*inp.nsamp(1);
% aggregated across-track pixel size, km
inpn_O2.dx0 = dx0;
% aggregated along-track pixel size, km
inpn_O2.dy0 = dx0;
% aperture size, cm2
inpn_O2.A = pi*D^2/4;

% initialize the inputs to the noise model
inpn_CH4 = [];
% readout noise, e per pixel
inpn_CH4.Nr = 50;
% dark current, e per s per pixel.
inpn_CH4.Nd_per_s = 10000;
% orbit height, km
inpn_CH4.H = 617;
% integration time, s. 1/7 means integration through 1 km
inpn_CH4.dt = dt;
inpn_CH4.eta = 0.4;
% ground fov for single pixel, across track, km
inpn_CH4.dx = dx;
% ground fov for single pixel, along track, km
inpn_CH4.dy = inpn_CH4.dx*inp.nsamp(1);
% aggregated across-track pixel size, km
inpn_CH4.dx0 = dx0;
% aggregated along-track pixel size, km
inpn_CH4.dy0 = dx0;
% aperture size, cm2
inpn_CH4.A = pi*D^2/4;

inp.inpn = {inpn_O2,inpn_CH4};

xch4er = zeros(length(SZA_vec),length(VZA_vec),length(alb_vec_o2),length(alb_vec_ch4));
xch4em = xch4er;
xch4es = xch4er;
xch4e_airglow = xch4er;
xch4e_sfcprs = xch4er;
for isza = 1:length(SZA_vec)
    SZA = SZA_vec(isza);
    for ivza = 1:length(VZA_vec)
        VZA = VZA_vec(ivza);
        for ialbo = 1:length(alb_vec_o2)
            albo = alb_vec_o2(ialbo);
            for ialbc = 1:length(alb_vec_ch4)
                albc = alb_vec_ch4(ialbc);
                disp(['Working on SZA=',num2str(SZA),', VZA=',num2str(VZA),', albo=',num2str(albo),', albc=',num2str(albc)])
                inp_nc = [];
                inp_nc.fn = ['CH4_1605-1690_0.01_',num2str(SZA),'_',num2str(VZA),'_',num2str(albc),'__GC_upwelling_output.nc'];
                % inp.nz = nz;
                inp_nc.if_lnR = if_lnR;
                outp_CH4 = F_read_spv_output(inp_nc);
                
                inp_nc = [];
                inp_nc.fn = ['O2_1245-1295_0.00_',num2str(SZA),'_',num2str(VZA),'_',num2str(albo),'__GC_upwelling_output.nc'];
                
                % inp.nz = nz;
                inp_nc.O2par_path = '~/CH4/O2.par.html';
                inp_nc.O21D_col = 2e17;
                inp_nc.VZA = VZA;
                inp_nc.if_lnR = if_lnR;
                outp_O2 = F_read_spv_output(inp_nc);
                inp.spv_output = {outp_O2,outp_CH4};
                
                outp_d = F_degrade_spv_output(inp);
                outp = F_gas_sens_spv(inp,outp_d);
                if ~exist('xch4','var')
                    xch4 = outp.h'*(outp_CH4.CH4_gascol*outp_CH4.gasnorm.CH4./outp_CH4.aircol)*1e9;
                end
                xch4er(isza,ivza,ialbo,ialbc) = outp.xch4e_r/xch4;
                xch4em(isza,ivza,ialbo,ialbc) = outp.xch4e_m/xch4;
                xch4es(isza,ivza,ialbo,ialbc) = outp.xch4e_s/xch4;
                xch4e_airglow(isza,ivza,ialbo,ialbc) = outp.xch4e_i_airglow/xch4;
                xch4e_sfcprs(isza,ivza,ialbo,ialbc) = outp.xch4e_i_sfcprs/xch4;
            end
        end
    end
end
dx0_str = num2str(dx0);
if dx0 == 0
dx0_str = 'native';
end
switch machine
    case 'CF'
        %         save([whichsensor,'.mat'],'xch4er','xch4es','xch4em','xch4e_airglow','xch4e_sfcprs')
        ncid = netcdf.create(['/data/wdocs/kangsun/www-docs/transfer/',whichsensor,'_',dx0_str,'.nc'],'CLOBBER');
        dimid_sza = netcdf.defDim(ncid,'sza',length(SZA_vec));
        dimid_vza = netcdf.defDim(ncid,'vza',length(VZA_vec));
        dimid_albo = netcdf.defDim(ncid,'albo',length(alb_vec_o2));
        dimid_albc = netcdf.defDim(ncid,'albc',length(alb_vec_ch4));
        % define variables
        id_xch4er = netcdf.defVar(ncid,'relative_retrieval_error','NC_FLOAT',[dimid_sza,dimid_vza,dimid_albo,dimid_albc]);
        id_xch4em = netcdf.defVar(ncid,'relative_measurement_error','NC_FLOAT',[dimid_sza,dimid_vza,dimid_albo,dimid_albc]);
        id_xch4es = netcdf.defVar(ncid,'relative_smoothing_error','NC_FLOAT',[dimid_sza,dimid_vza,dimid_albo,dimid_albc]);
        id_xch4e_airglow = netcdf.defVar(ncid,'relative_interference_error_airglow','NC_FLOAT',[dimid_sza,dimid_vza,dimid_albo,dimid_albc]);
        id_xch4e_sfcprs = netcdf.defVar(ncid,'relative_interference_error_sfcprs','NC_FLOAT',[dimid_sza,dimid_vza,dimid_albo,dimid_albc]);
        
        id_sza = netcdf.defVar(ncid,'SZA','NC_FLOAT',dimid_sza);
        id_vza = netcdf.defVar(ncid,'VZA','NC_FLOAT',dimid_vza);
        id_albo = netcdf.defVar(ncid,'albedo_o2','NC_FLOAT',dimid_albo);
        id_albc = netcdf.defVar(ncid,'albedo_ch4','NC_FLOAT',dimid_albc);
        % done defining the netcdf
        netcdf.endDef(ncid)
        % store variables
        netcdf.putVar(ncid,id_sza,SZA_vec)
        netcdf.putVar(ncid,id_vza,VZA_vec)
        netcdf.putVar(ncid,id_albo,alb_vec_o2)
        netcdf.putVar(ncid,id_albc,alb_vec_ch4)
        
        netcdf.putVar(ncid,id_xch4er,xch4er)
        netcdf.putVar(ncid,id_xch4em,xch4em)
        netcdf.putVar(ncid,id_xch4es,xch4es)
        netcdf.putVar(ncid,id_xch4e_airglow,xch4e_airglow)
        netcdf.putVar(ncid,id_xch4e_sfcprs,xch4e_sfcprs)
        % close
        netcdf.close(ncid)
end

