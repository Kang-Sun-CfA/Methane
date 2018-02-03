function outp = F_ro_simulator(inp)
% Simulating the observed radiance spectrum using the VER-based simulator:
% F_VER_airglow.m

% Key inputs:

% air mass factor
% planetary reflectance
% spectral sampling interval in nm
% equivalent SNR at fine at reference dlambda and radiance

% modifed by Kang Sun from F_radiance_simulator.m on 2018/02/02

SZA = inp.SZA;
VZA = inp.VZA;
amf = 1/cos(SZA/180*pi)+1/cos(VZA/180*pi);
lines = inp.lines;
llwaven = inp.llwaven;
lltranswaven = inp.lltranswaven;
window_list = inp.window_list;
wn1 = window_list(1).common_grid;
% vStart = 1235;vEnd = 1300;% in nm
if isfield(inp,'vStep')
    vStep = inp.vStep;
else
    vStep = 0.0005; % in nm
end
if isfield(inp,'vStart')
    vStart = inp.vStart;
else
    vStart = 1235; % in nm
end
if isfield(inp,'vEnd')
    vEnd = inp.vEnd;
else
    vEnd = 1300; % in nm
end
w1 = vStart:vStep:vEnd;
refl = inp.refl;
dlambda = inp.dlambda;
snre = inp.snre;
snrdefine_rad = inp.snrdefine_rad;
snrdefine_dlambda = inp.snrdefine_dlambda;

nsample = 3;
fwhm = dlambda*nsample;
w2 = vStart:dlambda:vEnd;

if ~isfield(inp,'s1') && ~isfield(inp,'airglow_spectrum')
    inpa = [];
    inpa.Z_airglow = inp.Z_airglow;
    inpa.VER_airglow = inp.VER_airglow;
    inpa.T_airglow = inp.T_airglow;
    inpa.P_airglow = inp.P_airglow;
    inpa.if_adjust_S = true;
    inpa.common_grid = 1e7./w1;
    inpa.lines = lines;
    inpa.AMF = 1/cos(VZA/180*pi);
    outpa = F_VER_airglow(inpa);
    
    airglow_spectrum = outpa.airglow_spec;% should already be in per nm grid, on w1 (nm)
    
    lltrans = interp1(1e7./llwaven,lltranswaven,w1);
    
    % light speed in m/s
    c = 2.99792458e8;
    % Planck constant in SI unit
    h = 6.62607004e-34;
    % Bolzmann constant in SI unit
    kB = 1.38064852e-23;
    % spectral radiance of black body on wavelength grid:
    B = @(T,lambda) 2*h*c^2*lambda.^-5./(exp(h*c./lambda/kB/T)-1);
    
    % solar irradiance, in w1 (nm) grid
    irrad = B(5770,w1/1e9)/1e9*pi*0.696^2/149.6^2/h./((1e7./w1)*c*100)/1e4.*lltrans;
%     irrad = 2e14*lltrans;
    total_od = amf*window_list(1).tau_struct.O2.Tau_sum+...
        amf*window_list(1).tau_struct.H2O.Tau_sum+...
        amf*window_list(1).tau_struct.O2.CIA_GFIT;
    
    % total optical depth, in w1 (nm) grid
    total_od = interp1(1e7./wn1,total_od,w1);
    s1 = irrad*cos(SZA/180*pi).*exp(-total_od)*refl/pi;
else
    s1 = inp.s1;
    airglow_spectrum = inp.airglow_spectrum;
end
s1_airglow = s1+airglow_spectrum;
s2 = F_conv_interp(w1,s1_airglow,fwhm,w2);

if ~isinf(snre)
    snr = (snre*sqrt(s2/snrdefine_rad*dlambda/snrdefine_dlambda));
    noise_std = s2./snr;
    
    noise = randn(size(s2)).*noise_std;
    s2 = s2+noise;
else
    noise_std = inf(size(s2));
end
outp.w2 = w2;
outp.s2 = s2;
outp.w1 = w1;
outp.s1 = s1;
outp.noise_std = noise_std;
outp.airglow_spectrum = airglow_spectrum;
outp.amf = amf;