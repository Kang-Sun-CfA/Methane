%% define the QE curve
ingaas_wave0 = [1401.83970959775;1407.27295174672;1420.56937029640;1429.62526751405;1441.02252511113;1446.85377965334;1459.39938298194;1481.57512691338;1494.34171242302;1505.73900527982;1520.84728025958;1530.78692990279;1540.06397384682;1553.09583435026;1564.13999528853;1578.40915149941;1589.05578192724;1595.19641622419;1602.04399604605;1609.68676617441;1617.10856587497;1626.34145804510;1641.05197920288;1649.93131633789;1656.99959626305;1668.52957950632;1677.54148949542;1689.91062676728;1695.69771778235;1700.82216783852;1705.15160388675;1710.32034675568;1715.26829549540;1720.74650039174;1724.45778809891;1728.12483000626;1731.35003683716;1733.07335465467;1734.53166780286;1736.56469446580;1738.90706562377;1740.89649291502;1743.19542924665;1745.53877592343;1747.88336844351;1749.43191267066;1750.98084475468;1752.88267369246;1753.90051513991;1755.27140623321;1757.26124488782;1758.80868431052;1760.09061367466;1761.41675357888;1763.22890383993;1765.08533516051;1766.85313384260;1768.88594894724;1771.13967571348;1774.58614680315;1778.07681667967;1782.18589346853;1785.72058583333;1790.18291207310;1794.64513253372;1797.29591968085;1801.00691355705;1805.20395304978;1808.25222369440;1813.55348065122;1821.63774501211;1830.29612347256;1838.60114670294;1845.71342560990;1849.24745979333];
ingaas_qe0 = 0.01*[85.2056549902880;85.8099172611661;86.4591607724769;86.5265514763508;86.4597668303316;86.2361984059808;86.5721819709865;87.0426956195323;87.1773186406700;87.0434116317278;86.4173839156906;86.0149442587079;85.5677367250729;85.2772593015793;84.6287370383850;83.5999502741847;82.8619197462707;82.3474969121150;81.5646055699510;80.7146154265084;80.0436117059593;79.5516546213953;79.2612269391512;79.0601229554118;78.6128499727644;78.1657091971219;77.8751126564258;77.7636085660234;77.5176647117180;77.2940753436832;76.8019729623117;76.3546436935136;75.7506890329937;74.8558865075703;73.7820386551420;72.8424342195797;72.0594355410354;71.1421476313165;70.1353553838187;68.3007376770131;66.2423879233256;63.3114369797963;59.8435162957442;55.9281145011853;49.6410558833448;45.0544068979108;39.7294119203230;35.3665146164746;31.9209300929697;29.1465796705374;25.4325344929057;22.9490528790610;21.3381541293346;19.6601343256655;17.8255100739585;15.7895200424627;14.2905062963908;12.8586227671236;11.3596234198344;9.83828529403765;8.27220023527239;6.79562998612802;5.52040980900963;4.33471360437385;3.35038448850728;2.83585824491198;2.32136341684264;1.58314177781248;1.24762028283424;0.822669061951236;0.464922669746879;0.420430987926730;0.286432350367177;0.174772491315286;0.152503088760753];

% wavelength where QE drops to 75%. 1720 nm in this case
qe0_75_wave = interp1(ingaas_qe0,ingaas_wave0,0.75);

F_ingaas_qe = @(ingaas_wave,qe_75_wave) ...
    interp1(ingaas_wave0-qe0_75_wave+qe_75_wave,ingaas_qe0,ingaas_wave,...
    'linear','extrap');
%% define your directories, you need to change these
% SAO's powerful unix workstations
% where gc output is
data_dir =  '/home/kangsun/GEOCAPE-TOOL/MASTER_MODULE/outp/';
% source code containing matlab functions
git_dir = '/home/kangsun/CH4/Methane/';

% ub workstation
% where gc output is
% data_dir =  '/mnt/Data2/gctool_data/outp/';
% source code containing matlab functions
% git_dir = '/home/kangsun/CH4/Methane/';

addpath([git_dir,'matlab_functions/'])
%%
cd(data_dir)

% viewing geometry scenarios
SZA_array = [50 25 0];% in degree
VZA_array = [45 23 0];
% surface albedo scenarios
ReflSpectra_array = {'conifer_ASTER.dat','sand_ASTER.dat','tapwater_ASTER_smooth.dat'};
ReflName_array = {'conifer','sand','tapwater'};

% which surface/angle to use
iRefl = 1;
iangle = 1;

SZA = SZA_array(iangle);
VZA = VZA_array(iangle);

ReflName = ReflName_array{iRefl};
disp(['run SZA = ',num2str(SZA),', refl = ',ReflName])

inp = [];
inp.fn = ['CH4_1600-1690_0.01_',num2str(SZA),'_',num2str(VZA),...
    '_',ReflName,'GC_upwelling_output.nc'];
outp_CH4 = F_read_gc_output(inp);
%% plot QE curve and methane absorption cross section
ingaas_wave = 1590:1730;
qe_75_wave_vec = 1650:10:1710;

close all;hold on
for qe_75_wave = qe_75_wave_vec
    ingaas_qe = F_ingaas_qe(ingaas_wave,qe_75_wave);
    plot(ingaas_wave,ingaas_qe,'-')
end
plotd = squeeze(outp_CH4.gas_xsecs(:,end,1));
plotd = plotd/max(plotd);
hrad = plot(outp_CH4.wave,plotd,'color',[1 1 1]*0.8);
uistack(hrad,'bottom')
hleg = legend(cat(1,'Methane xsection',cellstr(num2str(qe_75_wave_vec(:)))));
set(hleg,'location','southwest','box','off')
xlabel('Wavelength [nm]')
ylabel('QE')
%% inputs
optical_efficiency = 0.8;
n_samples_per_fwhm = 3;
% initialize the inputs to the noise model
inpn = [];
% readout noise, e per pixel
inpn.Nr = 50;
% dark current, e per s per pixel. place holder here. will loop over it
% later
inpn.Nd_per_s = nan;
% orbit height, km
inpn.H = 600;
% integration time, s. 1/7 means integration through dy0
inpn.dt = 1/14;
% QE curve. place holder here. will loop over it later
inpn.eta_wave = nan;
inpn.eta0 =     nan;
% ground fov for single pixel, across track, km
inpn.dx = 0.1;
% ground fov for single pixel, along track, km
inpn.dy = inpn.dx*n_samples_per_fwhm;
% aggregated across-track pixel size, km
inpn.dx0 = 1;
% aggregated along-track pixel size, km
inpn.dy0 = 1;
% aperture size, cm2
inpn.A = pi*4^2/4;

% initialize input to the main function
inp = [];
inp.inpn = inpn;
% number of windows
inp.nwin  =  1;
% start/end window wavelength, nm
inp.wmins = [1629; nan; nan];
inp.wmaxs = [1683; nan; nan];
% ILS fwhm, nm
inp.fwhm = [0.1; nan; nan];
% number of samples per fwhm
inp.nsamp = [n_samples_per_fwhm;    3;    3];
% albedo polynomial order
inp.nalb  = [1;    1;    1];
% use snre (shot noise only), or use full snr formula
inp.use_snre  = false;
inp.gc_output = {outp_CH4};
inp.gc_fwhm = [0.01];

% input items for F_gas_sens, change them if you know what u r doin
inp.included_gases = {'CH4','H2O','CO2'};
inp.inc_prof = logical([1,0,0]);
inp.inc_alb    = 1;
inp.inc_t      = 0;
inp.inc_aod    = 0;

inp.inc_assa   = 0;
inp.inc_cod    = 0;
inp.inc_cssa   = 0;
inp.inc_cfrac  = 0;
inp.inc_sfcprs = 0;

% a priori value relative to gas columns
inp.gascol_aperr_scale     = 10;
% a priori value relative to gas profiles, LT:0-2; UT:2-17; ST:>17 km
inp.gasprof_aperr_scale_LT = [0.1 0.005];
inp.gasprof_aperr_scale_UT = [0.05 0.005];
inp.gasprof_aperr_scale_ST = [0.02 0.005];

inp.t_aperr       = 5.0;
inp.aod_aperr     = 0.5;  %change from 1.0 to 0.5
inp.assa_aperr    = 0.05; %changed from 0.2 to 0.1 then 0.05 on April 2014
inp.cod_aperr     = 2.0;  %changed from 5.0 to 2.0 in April 2014
inp.cssa_aperr    = 0.01;
inp.cfrac_aperr   = 0.1;
inp.sfcprs_aperr  = 20.0;
inp.airglow_aperr  = .05;
inp.albs_aperr   = [0.1, 0.1/5., 0.1/25., 0.1/125., 0.1/625., 0.02/625.]; %2.0

inp.if_plot_overview = false;
%% loops over dark current and QE75 points
qe_75_wave_vec = 1640:5:1710;
DC_vec = 1000:2000:50000;
ingaas_wave = 1590:1730;
ch4vcde = nan(length(qe_75_wave_vec),length(DC_vec));
for iqe = 1:length(qe_75_wave_vec)
    qe_75_wave = qe_75_wave_vec(iqe);
    ingaas_qe = F_ingaas_qe(ingaas_wave,qe_75_wave);
    inp.inpn.eta_wave = ingaas_wave;
    inp.inpn.eta0 = ingaas_qe*optical_efficiency;
    tmp_ch4vcde = nan(1,length(DC_vec));
    % parallel loop!
    parfor idc = 1:length(DC_vec)
        inplocal = inp;
        inplocal.inpn.Nd_per_s = DC_vec(idc);
        a = F_wrapper(inplocal);
        tmp_ch4vcde(idc) = 100*a.vcd_error(1);
    end
    ch4vcde(iqe,:) = tmp_ch4vcde;
    disp(['QE75 point = ',num2str(qe_75_wave),' nm finished!'])
end
%%
close all
figure('color','w','unit','inch','position',[1 1 10 7])
[C,h] = contourf(qe_75_wave_vec,DC_vec/1000,ch4vcde');
clabel(C,h,'color','w')
set(gca,'linewidth',1,'xgrid','on','ygrid','on')
hc = colorbar;
set(get(hc,'ylabel'),'string','Methane VCD error [%]')
xlabel('QE75 wavelength [nm]')
ylabel('Dark current [1000 electrons/pixel/s]')
title([num2str(inp.wmins(1)),':',num2str(inp.fwhm(1)/n_samples_per_fwhm,2),':',...
    num2str(inp.wmaxs(1)),', fwhm = ',num2str(inp.fwhm(1)),' nm, dx = ',...
    num2str(inp.inpn.dx),' km, dt = ',num2str(inp.inpn.dt,2),' s, H = ',...
    num2str(inp.inpn.H),' km'])
%%
try
    addpath('/home/kangsun/matlab functions/export_fig/')
    export_fig('/data/wdocs/kangsun/www-docs/transfer/ingaas.png')
catch
    disp('Try to come out your own way to save the plot...')
end
