clc;clear
which_linux = 'UB';
% I'm jumping back and forth between three computers
if ispc
    git_dir = 'C:\Users\Kang Sun\Documents\GitHub\Methane\';
    spv_dir = 'C:\data_ks\MethaneSat\';
    O2par_path = 'C:\data_ks\MethaneSat\O2.par.html';
    CIA_xsec_path = 'C:\data_ks\MethaneSat\O2_CIA_296K_all.dat';
else
    switch which_linux
        case 'UB'
            git_dir = '~/CH4/Methane/';
            spv_dir = '/mnt/Data2/gctool_data/spv_outp/';
            O2par_path = '~/CH4/O2.par.html';
            CIA_xsec_path = '~/CH4/O2_CIA_296K_all.dat';
        case 'CF'
            git_dir = '~/CH4/Methane/';
            spv_dir = '/data/tempo1/Shared/kangsun/spv/outp/';
            O2par_path = '~/CH4/O2.par.html';
            CIA_xsec_path = '~/CH4/O2_CIA_296K_all.dat';
    end
end
addpath([git_dir,'matlab_functions'])

% some inputs, from ball
dx = 0.126;
dt = 1/17.5;
D = 4.37;
dx0 = 1.4;

if_lnR = false;
inp_nc = [];
inp_nc.fn = [spv_dir,'proxy_CH4_1603-1690_0.005_0.001_0.25_GC_upwelling_output.nc'];
inp_nc.if_lnR = if_lnR;
outp_CH4 = F_read_spv_output(inp_nc);

inpd = [];
inpd.nwin = 1;
inpd.wmins = 1606;
inpd.wmaxs = 1689;
inpd.nsamp = 3;
inpd.fwhm = 0.189;
inpd.nalb = 2;
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
inpn_CH4.dy = inpn_CH4.dx*inpd.nsamp(1);
% aggregated across-track pixel size, km
inpn_CH4.dx0 = dx0;
% aggregated along-track pixel size, km
inpn_CH4.dy0 = dx0;
% aperture size, cm2
inpn_CH4.A = pi*D^2/4;

inpd.inpn = {inpn_CH4};
inpd.spv_output = {outp_CH4};
outp_CH4_d = F_degrade_spv_output(inpd);
outp_CH4_d.radn = outp_CH4_d.rad+normrnd(0*outp_CH4_d.rad,outp_CH4_d.rad./outp_CH4_d.wsnr);
% plot(outp_CH4_d.wave,outp_CH4_d.rad,outp_CH4_d.wave,outp_CH4_d.radn)

inp_nc = [];
inp_nc.fn = [spv_dir,'proxy_O2_1245-1295_0.003_0.001_0.25_GC_upwelling_output.nc'];
inp_nc.if_lnR = if_lnR;
inp_nc.O2par_path = O2par_path;
inp_nc.O21D_col = 2e17;
inp_nc.VZA = outp_CH4.VZA;
outp_O2 = F_read_spv_output(inp_nc);

inpd = [];
inpd.nwin = 1;
inpd.wmins = 1246;
inpd.wmaxs = 1293;
inpd.nsamp = 3;
inpd.fwhm = 0.15;
inpd.nalb = 2;
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
inpn_O2.dy = inpn_O2.dx*inpd.nsamp(1);
% aggregated across-track pixel size, km
inpn_O2.dx0 = dx0;
% aggregated along-track pixel size, km
inpn_O2.dy0 = dx0;
% aperture size, cm2
inpn_O2.A = pi*D^2/4;

inpd.inpn = {inpn_O2};
inpd.spv_output = {outp_O2};
outp_O2_d = F_degrade_spv_output(inpd);
outp_O2_d.radn = outp_O2_d.rad+normrnd(0*outp_O2_d.rad,outp_O2_d.rad./outp_O2_d.wsnr);
% plot(outp_O2_d.wave,outp_O2_d.rad,outp_O2_d.wave,outp_O2_d.radn)
%%
% prepare input for O2 window
retrieved_molec = {'O2','H2O','O4'};
tmp = outp_O2;
inp = [];
inp.which_band = 'O2';
inp.retrieved_molec = retrieved_molec;
inp.irrad = tmp.irrad;
inp.w1 = tmp.wave;
inp.sza = tmp.sza;
inp.vza = tmp.vza;
inp.fwhm = 0.15;
inp.airglow_scale_factor = 1;
inp.LER_polynomial = 0.25;
inp.agspec = outp_O2.agspec*outp_O2.O21D_col;
inp.w2 = outp_O2_d.wave;
for imol = 1:length(retrieved_molec)
    if ~strcmpi(retrieved_molec{imol},'O4')
        inp.([retrieved_molec{imol},'_od']) = sum(repmat(...
            tmp.([retrieved_molec{imol},'_gascol'])*...
            tmp.gasnorm.(retrieved_molec{imol}),[1,tmp.nw])' ...
            .*tmp.([retrieved_molec{imol},'_gas_xsec']),2);
    else
        ciadata = importdata(CIA_xsec_path);
        O4_xsec = interp1(ciadata(:,1),ciadata(:,2),outp_O2.wave);
        inp.O4_od = sum(repmat(...
            (double(tmp.O2_gascol)*double(tmp.gasnorm.O2)).^2 ...
            ./abs(diff(1e5*double(tmp.zs0))),[1,tmp.nw])' ...
            .*repmat(O4_xsec(:),[1,tmp.nz]),2);
    end
end
coeff = [1 1 1];
% plot(outp_O2.wave,sum(outp_O2.ods,2),outp_O2.wave,inp.H2O_od+inp.O2_od+inp.O4_od)
%%
w1 = inp.w1;
w2 = inp.w2;
irrad = inp.irrad;
sza = inp.sza;
vza = inp.vza;
fwhm = inp.fwhm;
count = length(inp.retrieved_molec)+1;

if strcmpi(inp.which_band,'O2')
if isfield(inp,'airglow_scale_factor')
    airglow_scale_factor = inp.airglow_scale_factor;
else
    airglow_scale_factor = coeff(count);
    count = count+1;
end
end
if isfield(inp,'LER_polynomial')
    LER_polynomial = inp.LER_polynomial;
else
    LER_polynomial = coeff(count:end);
end
optical_path = 0*w1;
for imol = 1:length(inp.retrieved_molec)
    if ~strcmpi(inp.retrieved_molec{imol},'O4')
        optical_path = optical_path+inp.([inp.retrieved_molec{imol},'_od'])*coeff(imol);
    else
        optical_path = optical_path+inp.([inp.retrieved_molec{imol},'_od'])*coeff(imol)^2;
    end
end
s1 = irrad/pi*cos(sza/180*pi) ...
    .*exp(-optical_path*(1/cos(sza/180*pi)+1/cos(vza/180*pi))) ...
    .*polyval(LER_polynomial,w1-mean(w1))+inp.agspec*airglow_scale_factor;
s2 = F_instrument_model(w1,s1,fwhm,w2,[]);
plot(w2,s2,w2,outp_O2_d.radn)