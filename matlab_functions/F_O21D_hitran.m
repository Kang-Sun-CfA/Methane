function outp = F_O21D_hitran(inp)
% calculate line by line parameters and optionally absorption cross
% sections in support of airglow study for the O2 1 delta band

% Written by Kang Sun on 2017/12/22
% add patch option when revising grl paper.

if ~isfield(inp,'patch')
    patch = false;
else
    patch = inp.patch;
end
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

if ~isfield(inp,'if_adjust_Q')
    if_adjust_Q = false;
else
    if_adjust_Q = inp.if_adjust_Q;
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
if ~if_adjust_Q
    qqT0 = Q(T0,'O2',1);
    qqT = Q(local_T,'O2',1);
else
   airglowQ = [180,90.620796;181,91.113693;182,91.606583;183,92.099487;184,92.592407;185,93.085342;186,93.578270;187,94.071220;188,94.564171;189,95.057129;190,95.550102;191,96.043068;192,96.536049;193,97.029030;194,97.522018;195,98.015007;196,98.508011;197,99.000999;198,99.494003;199,99.987000;200,100.48000;201,100.97300;202,101.46600;203,101.95900;204,102.45199;205,102.94499;206,103.43798;207,103.93097;208,104.42395;209,104.91693;210,105.40990;211,105.90287;212,106.39583;213,106.88878;214,107.38173;215,107.87466;216,108.36759;217,108.86051;218,109.35342;219,109.84631;220,110.33920;221,110.83207;222,111.32494;223,111.81778;224,112.31062;225,112.80344;226,113.29624;227,113.78903;228,114.28180;229,114.77456;230,115.26730;231,115.76002;232,116.25272;233,116.74541;234,117.23807;235,117.73071;236,118.22333;237,118.71593;238,119.20851;239,119.70107;240,120.19360;241,120.68611;242,121.17859;243,121.67105;244,122.16348;245,122.65589;246,123.14827;247,123.64062;248,124.13294;249,124.62524;250,125.11750;251,125.60973;252,126.10194;253,126.59411;254,127.08625;255,127.57836;256,128.07043;257,128.56248;258,129.05449;259,129.54646;260,130.03841;261,130.53030;262,131.02217;263,131.51401;264,132.00578;265,132.49754;266,132.98924;267,133.48093;268,133.97256;269,134.46416;270,134.95570;271,135.44720;272,135.93867;273,136.43010;274,136.92148;275,137.41281;276,137.90410;277,138.39536;278,138.88654;279,139.37770;280,139.86880;281,140.35986;282,140.85086;283,141.34183;284,141.83273;285,142.32359;286,142.81439;287,143.30515;288,143.79585;289,144.28650;290,144.77710;291,145.26764;292,145.75813;293,146.24857;294,146.73894;295,147.22926;296,147.71953;297,148.20973;298,148.69987;299,149.18997;300,149.67999];
   airglowQ = double(airglowQ);
   qqT0 = interp1(airglowQ(:,1),airglowQ(:,2),T0,'linear','extrap');
   qqT = interp1(airglowQ(:,1),airglowQ(:,2),local_T,'linear','extrap');
end
if ~if_adjust_S
S = S0*qqT0...
    /qqT...
    .*exp(-c2*E/local_T)./exp(-c2*E/T0)...
    .*(1-exp(-c2*v0/local_T))./(1-exp(-c2*v0/T0));
else
S = S0*qqT0...
    /qqT...
    .*exp(-c2*E/local_T)./exp(-c2*E/T0)...
    .*(1-exp(-c2*v0/local_T))./(1-exp(-c2*v0/T0))...
    .*exp(-c2*(v0-7883.7538)./local_T);
end
if patch
    S = S.*v0.^2/mean(v0.^2);
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
