function outp = F_ro_simulator(inp)
% Simulating the observed radiance spectrum using the VER-based simulator:
% F_VER_airglow.m

% Key inputs:

% air mass factor
% planetary reflectance
% spectral sampling interval in nm
% equivalent SNR at fine at reference dlambda and radiance

% modifed by Kang Sun from F_radiance_simulator.m on 2018/02/02 

fn_airglow = inp.fn_airglow;
fn_solar = inp.fn_solar;
SZA = inp.SZA;
VZA = inp.VZA;
amf = 1/cos(SZA/180*pi)+1/cos(VZA/180*pi);
lines = inp.lines;
window_list = inp.window_list;
wn1 = window_list.common_grid;

load(fn_airglow);

inpa = [];
inpa.Z_airglow = Z_airglow;
inpa.VER_airglow = VER_airglow;
inpa.T_airglow = T_airglow;
inpa.P_airglow = P_airglow;
inpa.if_adjust_S = true;
inpa.common_grid = wn1;
inpa.lines = lines;
inpa.AMF = 1/cos(VZA/180*pi);
outpa = F_VER_airglow(inpa);

airglow_spectrum = outpa.airglow_spec;% should already be in per nm grid

refl = inp.refl;
dlambda = inp.dlambda;
snre = inp.snre;
snrdefine_rad = inp.snrdefine_rad;
snrdefine_dlambda = inp.snrdefine_dlambda;

load(fn_solar,'llwaven','lltranswaven')
irrad = 2.7e14*interp1(llwaven,lltranswaven,wn1);
vStart = 1240;vEnd = 1300;% in nm
vStep = 0.0005; % in nm
nsample = 3;

fwhm = dlambda*nsample;
w2 = vStart:dlambda:vEnd;
w1 = vStart:vStep:vEnd;

irrad = interp(w1,interp1(llwaven,lltranswaven,wn1));

% light speed in m/s
c = 2.99792458e8;
% Planck constant in SI unit
h = 6.62607004e-34;
% Bolzmann constant in SI unit
kB = 1.38064852e-23;
% spectral radiance of black body on wavelength grid:
B = @(T,lambda) 2*h*c^2*lambda.^-5./(exp(h*c./lambda/kB/T)-1);

xx = 1220:1320;
plot(xx,B(5770,xx/1e9)/1e9*pi*0.696^2/149.6^2/h./((1e7./xx)*c*100)/1e4)







total_od = amf*window_list.tau_struct.O2.Tau_sum+...
    amf*window_list.tau_struct.H2O.Tau_sum+...
    amf*window_list.tau_struct.O2.CIA_GFIT;
s1 = irrad.*exp(-total_od)*refl/pi;

s1_airglow = interp1(1e7./wn1,s1+airglow_spectrum,w1);
s2 = F_conv_interp(w1,s1_airglow,fwhm,w2);

if ~isinf(snre)
snr = (snre*sqrt(s2/snrdefine_rad*dlambda/snrdefine_dlambda));
noise_std = s2./snr;

noise = randn(size(s2)).*noise_std;
s2 = s2+noise;
end
outp.w2 = w2;
outp.s2 = s2;

