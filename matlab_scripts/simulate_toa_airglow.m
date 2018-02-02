clear;clc
addpath('C:\Users\Kang Sun\Documents\GitHub\Realtime_FTS\matlab_script\')
addpath('C:\Users\Kang Sun\Documents\GitHub\Methane\matlab_functions\')
lines = F_import_par('C:\data_ks\MethaneSat\O2.par.html');
%%
clc
wStart = 1235; wEnd = 1300;step = 0.001;
w1 = wStart:step:wEnd;
v1 = 1e7./w1;

load('C:\data_ks\MethaneSat\airglow_profile.mat','Z_airglow','VER_airglow','VER_airglowe',...
    'T_airglow','T_airglowe','P_airglow')
inp = [];
inp.Z_airglow = Z_airglow;
inp.VER_airglow = VER_airglow;
inp.T_airglow = T_airglow;
inp.P_airglow = P_airglow;
inp.if_adjust_S = true;
inp.common_grid = v1;
inp.lines = lines;
inp.AMF = 3;
outp = F_VER_airglow(inp);
%%
plot(w1,outp.airglow_spec)
trapz(w1,outp.airglow_spec)*4*pi/1e12