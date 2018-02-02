function outp = F_VER_airglow(inp)
% simulating airglow spectrum attenuated by absorption above, starting from
% local VER in photons/cm3/s, instead of MR
% modified by Kang Sun from F_attenuated_airglow.m on 2018/02/01

Z_airglow = inp.Z_airglow;
P_airglow = inp.P_airglow;
T_airglow = inp.T_airglow;
dZ_airglow = diff(Z_airglow);
dZ_airglow = [dZ_airglow(:);dZ_airglow(end)];
if_adjust_S = inp.if_adjust_S;
lines = inp.lines;
common_grid = inp.common_grid;
nlayer = length(Z_airglow);
AMF = inp.AMF;
VER_airglow = inp.VER_airglow;

tau = zeros(nlayer,length(common_grid),'single');
N_O2 = zeros(size(Z_airglow));
for ilayer = 1:nlayer
    inph = [];
    inph.lines = lines;
    inph.if_adjust_S = false;
    inph.T = T_airglow(ilayer);
    inph.P = P_airglow(ilayer);
    inph.if_lbl = true;
    inph.fwhm = 0;
    inph.common_grid = common_grid;
    outp = F_O21D_hitran(inph);
    tau(ilayer,:) = outp.xsec*outp.N*dZ_airglow(ilayer)*100*1000;
    N_O2(ilayer) = outp.N;
end
outp.N_O2 = N_O2;
outp.tau = tau;

airglow_spec = zeros(1,length(common_grid),'single');
for ilayer = 1:nlayer
    inph = [];
    inph.lines = lines;
    inph.if_adjust_S = if_adjust_S;
    inph.T = T_airglow(ilayer);
    inph.P = P_airglow(ilayer);
    inph.if_lbl = true;
    inph.fwhm = 0;
    inph.common_grid = common_grid;
    outp = F_O21D_hitran(inph);
    airglow_local = outp.xsec/trapz(1e7./common_grid,outp.xsec)*VER_airglow(ilayer);
    airglow_local = airglow_local(:)';
    airglow_spec = airglow_spec+...
        airglow_local*dZ_airglow(ilayer)*AMF*100*1000/4/pi ...
        .*exp(-sum(tau(ilayer:end,:))*AMF);
end
outp = [];
outp.airglow_spec = airglow_spec;