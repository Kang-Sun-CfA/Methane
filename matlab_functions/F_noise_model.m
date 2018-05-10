function outpn = F_noise_model(inpn)
% matlab function to calculate channel-by-channel snr. written by Kang Sun
% on 2018/04/30
% updated on 2018/05/09 to include dy, along track ifov

if isfield(inpn,'snrdefine_rad')
    I = inpn.snrdefine_rad;
else
    % reference radiance, in photons/cm2/s/nm/sr
    I = 2e13;
end
if isfield(inpn,'A')
    A = inpn.A;
else
    % area of telescope, in cm2
    A = pi*4^2/4;
end
if isfield(inpn,'dt')
    dt = inpn.dt;
else
    % integration time, in s
    dt = 1/7;
end
if isfield(inpn,'snrdefine_dlambda')
    dlambda = inpn.snredefine_dlambda;
else
    % spectral sampling interval, in nm
    dlambda = 0.05;
end
if isfield(inpn,'dx')
    dx = inpn.dx;
else
    % native pixel size
    dx = 0.2;
end
if isfield(inpn,'dy')
    dy = inpn.dy;
else
    % native pixel size
    dy = dx;
end
if isfield(inpn,'H')
    H = inpn.H;
else
    % orbit height
    H = 460;
end
% footprint solid angle, in steradian
Omega = (dx/H)*(dy/H);

if isfield(inpn,'eta')
    eta = inpn.eta;
end
if isfield(inpn,'eta_wave')
    eta_wave = inpn.eta_wave;
    eta0 = inpn.eta0;
    if isfield(inpn,'wave')
        wave = inpn.wave;
        
        eta = interp1(eta_wave,eta0,wave,'linear','extrap');
    else
        eta = eta0(1);
    end
end
if ~exist('eta','var')
% instrument efficiency
eta = 0.65;
end
if isfield(inpn,'dx0')
    dx0 = inpn.dx0;
else
    % target x track
    dx0 = 1;
end
if isfield(inpn,'dy0')
	dy0 = inpn.dy0;
else
	dy0 = dt/(1/7);
end
% number of along track averaging
n1 = dy0/7/dt;
% number of x-track averaging
n = dx0/dx;
% signal
S = I*A*dt*dlambda*Omega*eta;
if isfield(inpn,'Nr')
    Nr = inpn.Nr;
else
    % readout rms: electron per pixel per readout
    Nr = 30;
end
if isfield(inpn,'Nd_per_s')
    Nd_per_s = inpn.Nd_per_s;
else
    Nd_per_s = 2500;
end
% dark current, electrons per pixel per exposure
Nd = Nd_per_s * dt;
% noise
N = sqrt((Nr^2+Nd+S)/n/n1);
%
SNRe_shot = sqrt(S*n*n1);
SNRe = S./N;
outpn.snre = SNRe;
outpn.snre_shot = SNRe_shot;
if isfield(inpn,'rad')
    rad = inpn.rad;
else
    rad = I;
end
if isfield(inpn,'dl')    
    dl = inpn.dl;
else
    dl = dlambda;
end
tmpS = (rad*A*dt*dl*Omega.*eta);
tmpN = sqrt((Nr^2+Nd+tmpS)/n/n1);
outpn.wsnr = tmpS./tmpN;
outpn.wsnr_single = tmpS./sqrt(Nr^2+Nd+tmpS);
outpn.wsnr_shot = SNRe_shot.*sqrt(rad/I*dl/dlambda);
outpn.A = A;
outpn.dx = dx;
outpn.H = H;

