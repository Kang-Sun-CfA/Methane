function outp = F_O21D_hitran(inp)
% calculate line by line parameters and optionally absorption cross
% sections in support of airglow study for the O2 1 delta band

% Written by Kang Sun on 2017/12/22

common_grid = inp.common_grid;
local_T = inp.T;
local_P = inp.P;
if_lbl = inp.if_lbl;
lines = inp.lines;

if ~isfield(inp,'if_adjust_S')
    if_adjust_S = false;
else
    if_adjust_S = inp.if_adjust_S;
end

if ~isfield(inp,'fwhm')
    fwhm = 0;
else
    fwhm = inp.fwhm;
end

if ~if_lbl && fwhm > 0
    warning('Enable if_lbl to convolve spectrum')
    fwhm = 0;
end

if ~isfield(inp,'ints')
    ints = {true(size(lines.transitionWavenumber))};
    scale_ints = 1;
else
    ints = inp.ints;
    scale_ints = inp.scale_ints;
end

v_grid = common_grid;
MW = 31.98983/1000;
% speed of light in SI unit
c = 2.99792458e8;
% Planck constant in SI unit
h = 6.62607004e-34;
% Bolzmann constant in SI unit
kB = 1.38064852e-23;
% Avogadro's Constant in SI unit
Na = 6.02214e23;
% second radiation constant, 1.4388 cm K, note it is cm
c2 = h*c/kB*100;

% HITRAN reference temperature/pressure, 296 K, 1 atm
T0 = 296; P0 = 1013.25;

Filter = true(size(lines.transitionWavenumber));
v0 = lines.transitionWavenumber(Filter);
S0 = lines.lineIntensity(Filter);
for iints = 1:length(scale_ints)
   S0(ints{iints}) = S0(ints{iints})*scale_ints(iints);
end
GammaP0 = lines.airBroadenedWidth(Filter);
GammaPs0 = lines.selfBroadenedWidth(Filter);
E = lines.lowerStateEnergy(Filter);
n = lines.temperatureDependence(Filter);
delta = lines.pressureShift(Filter);
MR = 0.2095;

% number densities in molecules/cm3
N = MR*local_P*100/kB/local_T/1e6;
% Doppler HWHM at this layer
GammaD = (v0/c).*sqrt(2*kB*Na*local_T*log(2)./MW);
% line strength at this layer
if ~if_adjust_S
S = S0*Q(T0,'O2',1)...
    /Q(local_T,'O2',1)...
    .*exp(-c2*E/local_T)./exp(-c2*E/T0)...
    .*(1-exp(-c2*v0/local_T))./(1-exp(-c2*v0/T0));
else
S = S0*Q(T0,'O2',1)...
    /Q(local_T,'O2',1)...
    .*exp(-c2*E/local_T)./exp(-c2*E/T0)...
    .*(1-exp(-c2*v0/local_T))./(1-exp(-c2*v0/T0))...
    .*exp(-c2*(v0-7883)./local_T);
end
% Lorentzian HWHM at this layer
GammaP = (GammaP0.*(1-MR)+GammaPs0.*MR)...
    *(local_P/P0)...
    .*((T0/local_T).^n);
% line position currected by air-broadened pressure shift
v0 = v0+delta*local_P/P0;
outp.w0 = 1e7./v0;
outp.S0 = S0;
outp.S = S;
outp.GammaD = GammaD;
outp.GammaP = GammaP;
outp.N = N;
%%
% the core of line-by-line calculation
xsec = zeros(length(v_grid),1,'single');
localcutoff = 5;
if if_lbl
for iline = 1:length(v0)
    vFilter = v_grid > v0(iline)-localcutoff...
        & v_grid < v0(iline)+localcutoff;
    lineprofile = voigt_fn_fast(v_grid(vFilter),...
        v0(iline),GammaD(iline),GammaP(iline));
    localtau = zeros(length(v_grid),1,'single');
    localtau(vFilter) = S(iline).*lineprofile;
    xsec = xsec+localtau;
end
wgrid = 1e7./inp.common_grid;
if fwhm > 0
    
    wgrid = linspace(min(wgrid),max(wgrid),length(wgrid));
    xsec = interp1(1e7./inp.common_grid,xsec,wgrid);
    xsec = F_conv_interp(wgrid,xsec,fwhm,wgrid);
end
outp.xsec = xsec;
outp.wgrid = wgrid;
end
return

function s1_low = F_conv_interp(w1,s1,fwhm,common_grid)
% This function convolves s1 with a Gaussian fwhm, resample it to
% common_grid

% Made by Kang Sun on 2016/08/02

slit = fwhm/1.66511;% half width at 1e

dw0 = median(diff(w1));
ndx = ceil(slit*2.7/dw0);
xx = (0:ndx*2)*dw0-ndx*dw0;
kernel = exp(-(xx/slit).^2);
kernel = kernel/sum(kernel);
s1_over = conv(s1, kernel, 'same');
s1_low = interp1(w1,s1_over,common_grid);
