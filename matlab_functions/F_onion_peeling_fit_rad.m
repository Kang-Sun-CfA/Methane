function s2 = F_onion_peeling_fit_rad(coeff,inp)
% onion-peeling for both O2 1D and A band. modified from
% F_fit_absorpbed_airglow.m by Kang Sun on 2018/11/27

count = 1;

if ~isfield(inp,'shift')
    count = count+1;
    shift = coeff(count);
else
    shift = inp.shift;
end
if ~isfield(inp,'FWHM')
    count = count+1;
    FWHM = coeff(count);
else
    FWHM = inp.FWHM;
end
if ~isfield(inp,'whichband')
    whichband = 'O2_1270';
else
    whichband = inp.whichband;
end
if ~isfield(inp,'lineshape')
    lineshape = 'gaussian';
else
    lineshape = inp.lineshape;
end

if strcmp(whichband,'O2_1270')
    if FWHM >= 1.5
        FWHM = 1.5;
    end
    if FWHM <= 1
        FWHM = 1;
    end
elseif strcmp(whichband,'O2_760')
    if FWHM >= 1
        FWHM = 1;
    end
    if FWHM <= 0.2
        FWHM = 0.2;
    end
end
if ~isfield(inp,'T')
    count = count+1;
    if strcmp(inp.Tscale,'log')
        inp.T = 10^coeff(count);
    else
        inp.T = coeff(count);
    end
end
w2 = inp.w2;
bandhead = inp.bandhead;
lines = inp.lines;
v_grid = inp.common_grid;
local_T_airglow = inp.T;
% use climatology temperature for absorption as the fitted one can go wild
local_T_abs = inp.guessT;
local_P = inp.P;
% O2 molecular weight in kg
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
GammaP0 = lines.airBroadenedWidth(Filter);
GammaPs0 = lines.selfBroadenedWidth(Filter);
E = lines.lowerStateEnergy(Filter);
n = lines.temperatureDependence(Filter);
A = lines.einsteinACoefficient(Filter);
delta = lines.pressureShift(Filter);
g = lines.upperStatisticalWeight(Filter);
MR = 0.2095;
qqT0 = Q(T0,'O2',1);
qqT = Q(local_T_abs,'O2',1);
S = S0*qqT0...
    /qqT...
    .*exp(-c2*E/local_T_abs)./exp(-c2*E/T0)...
    .*(1-exp(-c2*v0/local_T_abs))./(1-exp(-c2*v0/T0));

% number densities in molecules/cm3
% N = MR*local_P*100/kB/local_T_airglow/1e6;
N_abs = MR*local_P*100/kB/local_T_abs/1e6;
% Doppler HWHM at this layer
GammaD = (v0/c).*sqrt(2*kB*Na*local_T_airglow*log(2)./MW);
GammaD_abs = (v0/c).*sqrt(2*kB*Na*local_T_abs*log(2)./MW);
EE = E+v0-bandhead;

% Lorentzian HWHM at this layer
GammaP = (GammaP0.*(1-MR)+GammaPs0.*MR)...
    *(local_P/P0)...
    .*((T0/local_T_airglow).^n);
GammaP_abs = (GammaP0.*(1-MR)+GammaPs0.*MR)...
    *(local_P/P0)...
    .*((T0/local_T_abs).^n);
if strcmpi(lineshape,'gaussian')
    GammaP = GammaP*0;
    GammaP_abs = GammaP_abs*0;
end
% line position currected by air-broadened pressure shift
v0 = v0+delta*local_P/P0;

% the core of line-by-line calculation
xsec = zeros(length(v_grid),1,'single');
xsec_abs = xsec;
localcutoff = 5;
for iline = 1:length(v0)
    vFilter = v_grid > v0(iline)-localcutoff...
        & v_grid < v0(iline)+localcutoff;
    lineprofile = voigt_fn_fast(v_grid(vFilter),...
        v0(iline),GammaD(iline),GammaP(iline));
    localtau = zeros(length(v_grid),1,'single');
    localtau(vFilter) = g(iline)*exp(-c2*EE(iline)/local_T_airglow)*A(iline).*lineprofile;
    xsec = xsec+localtau;
    
    lineprofile_abs = voigt_fn_fast(v_grid(vFilter),...
        v0(iline),GammaD_abs(iline),GammaP_abs(iline));
    localtau = localtau*0;
    localtau(vFilter) = S(iline).*lineprofile_abs;
    xsec_abs = xsec_abs+localtau;
end

wgrid = 1e7./inp.common_grid;

w1 = wgrid;
% ver spectrum in photons/s/cm3/nm
ver_spec = xsec/sum(xsec)/mean(diff(w1))*10^coeff(1)*inp.einstein_A;
LL = inp.LL;
% local optical depth
tau_all = 0.5*xsec_abs*N_abs*LL(1);
ver_up_all = zeros(size(tau_all,1),1);

if length(LL) > 1
    absco_up = inp.absco_up;
    ver_up = inp.ver_up;
    
    for i = 1:length(LL)-1
        tau_all = tau_all+absco_up(:,i)*LL(i+1);
        tmptau = zeros(size(ver_up,1),1);
        for j = i:length(LL)-1
            if j == i
                tmptau = tmptau+absco_up(:,j)*LL(j+1)*0.5;
            else
                tmptau = tmptau+absco_up(:,j)*LL(j+1);
            end
        end
        ver_up_all = ver_up_all+ver_up(:,i)*LL(i+1)/4/pi.*exp(-tmptau);
    end
end

s1 = ver_spec*LL(1)/4/pi.*exp(-tau_all)+ver_up_all;

slit = FWHM/1.66511;% half width at 1e

dw0 = median(diff(w1));
ndx = ceil(slit*2.7/dw0);
xx = (0:ndx*2)*dw0-ndx*dw0;
kernel = exp(-(xx/slit).^2);
kernel = kernel/sum(kernel);
s1_over = conv(s1, kernel, 'same');
s2 = interp1(w1,s1_over,w2+shift,'linear','extrap');

% s2 = F_conv_interp(w1,s1,FWHM,w2)*coeff(1);
s2 = double(s2(:));
if sum(isnan(s2)) || sum(isinf(2))
    
    disp('stop')
end
if inp.output_struct
    outp_struct = [];
    outp_struct.nO2 = N_abs;
    outp_struct.local_absco = xsec_abs*N_abs;
    outp_struct.local_ver = ver_spec;
    outp_struct.w1 = w1;
    outp_struct.s2 = s2;
    s2 = outp_struct;
end
end

% calculate the Voigt profile using the complex error function

function lsSA2 = voigt_fn_fast(nu,nu0,gamma_D,gamma_L)

% nu is wavenumber array [cm^-1]
% nu0 is line center [cm^-1]
% gamma_D is Doppler linewidth [cm^-1]
% gamma_L is Lorentz linewidth [cm^-1]
if gamma_L == 0
    lsSA2 = 1/sqrt(2*pi)/(gamma_D/sqrt(2*log(2)))...
        *exp(-(nu-nu0).^2/(2*(gamma_D/sqrt(2*log(2)))^2));
else
% convert to dimensionless units
x = sqrt(log(2)).*(nu-nu0)./(gamma_D);
y = sqrt(log(2)).*(gamma_L/gamma_D);

% call complexErrorFunction
lsSA2 =(y/sqrt(pi)/gamma_L)*real(voigtf(x,y,2));
end
if size(lsSA2) ~= size(nu)
    lsSA2 = lsSA2';
end
end



function VF = voigtf(x,y,opt)

% This function file is a subroutine for computation of the Voigt function.
% The input parameter y is used by absolute value. The parameter opt is
% either 1 for more accurate or 2 for more rapid computation.
%
% NOTE: This program completely covers the domain 0 < x < 40,000 and
% 10^-4 < y < 10^2 required for applications using HITRAN molecular
% spectroscopic database. However, it may be implemented only to cover the
% smaller domain 0 <= x <= 15 and 10^-6 <= y <= 15 that is the most
% difficult for rapid and accurate computation.
%
% The code is written by Sanjar M. Abrarov and Brendan M. Quine, York
% University, Canada, March 2015.

if nargin == 2
    opt = 1; % assign the defaul value opt = 1
end

if opt ~= 1 && opt ~=2
    disp(['opt = ',num2str(opt),' cannot be assigned. Use either 1 or 2.'])
    return
end

% *************************************************************************
% Define array of coefficients as coeff = [alpha;beta;gamma]'
% *************************************************************************
if opt == 1
    
    coeff = [
        1.608290174437121e-001 3.855314219175531e-002  1.366578214428949e+000
        6.885967427017463e-001 3.469782797257978e-001 -5.742919588559361e-002
        2.651151642675390e-001 9.638285547938826e-001 -5.709602545656873e-001
        -2.050008245317253e-001 1.889103967396010e+000 -2.011075414803758e-001
        -1.274551644219086e-001 3.122804517532180e+000  1.069871368716704e-002
        -1.134971805306579e-002 4.664930205202391e+000  1.468639542320982e-002
        4.201921570328543e-003 6.515481030406647e+000  1.816268776500938e-003
        8.084740485193432e-004 8.674456993144942e+000 -6.875907999947567e-005
        1.946391440605860e-005 1.114185809341728e+001 -2.327910355924500e-005
        -4.132639863292073e-006 1.391768433122366e+001 -1.004011418729134e-006
        -2.656262492217795e-007 1.700193570656409e+001  2.304990232059197e-008
        -1.524188131553777e-009 2.039461221943855e+001  2.275276345355270e-009
        2.239681784892829e-010 2.409571386984707e+001  3.383885053101652e-011
        4.939143128687883e-012 2.810524065778962e+001 -4.398940326332977e-013
        4.692078138494072e-015 3.242319258326621e+001 -1.405511706545786e-014
        -2.512454984032184e-016 3.704956964627684e+001 -3.954682293307548e-016
        ];
    mMax = 16; % 16 summation terms
    
elseif opt == 2
    
    coeff = [
        2.307372754308023e-001 4.989787261063716e-002  1.464495070025765e+000
        7.760531995854886e-001 4.490808534957343e-001 -3.230894193031240e-001
        4.235506885098250e-002 1.247446815265929e+000 -5.397724160374686e-001
        -2.340509255269456e-001 2.444995757921221e+000 -6.547649406082363e-002
        -4.557204758971222e-002 4.041727681461610e+000  2.411056013969393e-002
        5.043797125559205e-003 6.037642585887094e+000  4.001198804719684e-003
        1.180179737805654e-003 8.432740471197681e+000 -5.387428751666454e-005
        1.754770213650354e-005 1.122702133739336e+001 -2.451992671326258e-005
        -3.325020499631893e-006 1.442048518447414e+001 -5.400164289522879e-007
        -9.375402319079375e-008 1.801313201244001e+001  1.771556420016014e-008
        8.034651067438904e-010 2.200496182129099e+001  4.940360170163906e-010
        3.355455275373310e-011 2.639597461102705e+001  5.674096644030151e-014
        ];
    mMax = 12; % 12 summation terms
end
% *************************************************************************

varsigma = 2.75; % define the shift constant
y = abs(y) + varsigma/2;

arr1 = y.^2 - x.^2; % define 1st repeating array
arr2 = x.^2 + y.^2; % define 2nd repeating array
arr3 = arr2.^2;  % define 3rd repeating array

VF = 0; % initiate VF
for m = 1:mMax
    VF = VF + (coeff(m,1)*(coeff(m,2) + arr1) + ...
        coeff(m,3)*y.*(coeff(m,2) + arr2))./(coeff(m,2)^2 + ...
        2*coeff(m,2)*arr1 + arr3);
end
end

function Q = Q(T,mol,iiso)

% Reference: Gamache et al.,(2000)
% Total internal partition sums for molecules in the terrestrial atmosphere

if nargin < 2
    mol = 'default';iiso = 1;
elseif nargin < 3
    iiso = 1;
end

switch mol
    case 'NO'
        p = [0 0 0 1];
    case 'N2O'
        p = [0.46310e-4 -0.76213e-2 0.14979e2 0.24892e2];
        
    case 'CH4'
        p = [0.15117e-5 0.26831e-2 0.11557e1 -0.26479e2];
        
    case 'CO'
        p = [0.14896e-7 -0.74669e-5 0.3629 0.27758];
        
    case 'C2H2'
        p = [0.84612e-5 -0.25946e-2 0.14484e1 -0.83088e1];
        
    case 'NH3'
        p = [0.18416e-5 0.94575e-2 0.30915e1 -0.62293e2];
        
    case 'C2H4'
        p = [0 0 0 1];
        
    case 'H2O'
        switch iiso
            case 1
                p = [0.48938e-6 0.12536e-2 0.27678 -0.44405e1];
            case 2
                p = [0.52046e-6 0.12802e-2 0.27647 -0.43624e1];
            case 3
                p = [0.31668e-5 0.76905e-2 0.16458e1 -0.25767e2];
            case 4
                p = [0.21530e-5 0.61246e-2 0.13793e1 -0.23916e2];
            otherwise
                p = [0.48938e-6 0.12536e-2 0.27678 -0.44405e1];
        end
        
    case 'CO2'
        p = [0.25974e-5 -0.69259e-3 0.94899 -0.1317e1];
        
    case 'O3'
        p = [0.26669e-4 0.10396e-1 0.69047e1 -0.16443e3];
        
    case 'SO2'
        p = [0.52334e-4 0.22164e-1 0.11101e2 -0.24056e3];
        
    case 'O2'
        p = [0.13073e-6 -0.64870e-4 0.73534 0.35923];
        
    case 'HF'
        p = [0.46889e-8 0.59154e-5 0.13350 0.15486e1];
        
    case 'default'
        p = [0 0 0 1];
    otherwise
        p = [0 0 0 1];
        
end

Q = polyval(p,T);
end
