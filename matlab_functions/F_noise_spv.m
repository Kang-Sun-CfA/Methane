function outpn = F_noise_spv(inpn)
% matlab function to calculate channel-by-channel snr. modified by Kang Sun
% from F_noise_model.m on 2018/08/20, made simpler
% updated on 2018/12/02 to output signal in electrons per exposure

if isfield(inpn,'I')
    I = inpn.I;
else
    % radiance, in photons/cm2/s/nm/sr
    I = 2e13;
end

if isfield(inpn,'A')
    A = inpn.A;
else
    % area of telescope, in cm2
    A = pi*4^2/4;
end

if isfield(inpn,'dl')
    dl = inpn.dl;
else
    % spectral sampling interval, in nm
    dl = 0.05;
end

if isfield(inpn,'dt')
    dt = inpn.dt;
else
    % integration time, in s
    dt = 1/7;
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
    dy = 3*dx;
end
if isfield(inpn,'H')
    H = inpn.H;
else
    % orbit height
    H = 617;
end
% footprint solid angle, in steradian
Omega = (dx/H)*(dy/H);

% instrument efficiency, scalar or vector
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

if dy0 == 0
    n_along_track = 1;
else
% number of along track averaging
n_along_track = dy0/7/dt;
end
if dx0 == 0
    n_across_track = 1;
else
% number of x-track averaging
n_across_track = dx0/dx;
end

% readout and dark noise
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

% if isfield(inpn,'f_number')
%     if isfield(inpn,'A') || isfield(inpn,'dx') || isfield(inpn,'dy') 
%         warning('f_number formula overwrites the aperture size formula. A, dx, dy will not be used!')
%     end
%     S = I*pi/4*(inpn.f_number)^2*(inpn.dp)^2*inpn.nsamp*dt*dl*eta;
% end
%         
% signal
S = I*A*dt*dl*Omega.*eta;
% noise
N = sqrt(Nr^2+Nd+S);

outpn.wsnr = S./N *sqrt(n_across_track*n_along_track);
outpn.wsnr_single = S./N;
outpn.wsnr_shot = sqrt(S*n_across_track*n_along_track);
outpn.A = A;
outpn.dx = dx;
outpn.H = H;
outpn.S = S;
outpn.N = N;

