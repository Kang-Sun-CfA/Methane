function outp = F_attenuated_airglow(inp)
% simulating airglow spectrum attenuated by absorption above
% written by Kang Sun on 2017/12/26

if ~isfield(inp,'Z_airglow')
    Z_airglow = [20;21;22;23;24;25;27.5;30;32.5;35;37.5;40;42.5;45;47.5;50;55;60;65;70;75;80;85;90;95;100;105;110;115;120];
    P_airglow = [55.3000000000000;47.3000000000000;40.5000000000000;34.7000000000000;29.7000000000000;25.5000000000000;17.4000000000000;12;8.01000000000000;5.75000000000000;4.15000000000000;2.87114000000000;2.06000000000000;1.49000000000000;1.09000000000000;0.798000000000000;0.425000000000000;0.219000000000000;0.109000000000000;0.0522000000000000;0.0240000000000000;0.0105000000000000;0.00446000000000000;0.00184000000000000;0.000760000000000000;0.000320000000000000;0.000145000000000000;7.10000000000000e-05;4.01000000000000e-05;2.54000000000000e-05];
    T_airglow = [216.700000000000;217.600000000000;218.600000000000;219.600000000000;220.600000000000;221.600000000000;224;226.500000000000;230;236.500000000000;242.900000000000;250.400000000000;257.300000000000;264.200000000000;270.600000000000;270.700000000000;260.800000000000;247;233.300000000000;219.600000000000;208.400000000000;198.600000000000;188.900000000000;186.900000000000;188.400000000000;195.100000000000;208.800000000000;240;300;360];
    dZ_airglow = diff(Z_airglow);
    dZ_airglow = [dZ_airglow(:);dZ_airglow(end)];
else
    Z_airglow = inp.Z_airglow;
    P_airglow = inp.P_airglow;
    T_airglow = inp.T_airglow;
    dZ_airglow = inp.dZ_airglow;
end

lines = inp.lines;
common_grid = inp.common_grid;
nlayer = length(Z_airglow);
AMF = inp.AMF;
airglow_MR = inp.airglow_MR;
airglow_height = inp.airglow_height;
airglow_width_up = inp.airglow_width_up;
airglow_width_lo = inp.airglow_width_lo;
airglow_profile = Z_airglow;
airglow_profile(Z_airglow >= airglow_height)...
    = exp(-(Z_airglow(Z_airglow >= airglow_height)-airglow_height).^2/(airglow_width_up/sqrt(2))^2);
airglow_profile(Z_airglow < airglow_height)...
    = exp(-(Z_airglow(Z_airglow < airglow_height)-airglow_height).^2/(airglow_width_lo/sqrt(2))^2);
outp.airglow_profile = airglow_profile;
outp.Z_airglow = Z_airglow;
outp.P_airglow = P_airglow;
outp.T_airglow = T_airglow;
tau = zeros(nlayer,length(common_grid),'single');
N_O2 = zeros(size(Z_airglow));
for ilayer = 1:nlayer
    inph = [];
    inph.lines = lines;
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
    airglow_spec = airglow_spec+...
        airglow_profile(ilayer)*tau(ilayer,:)/N_O2(ilayer)...
        .*exp(-sum(tau(ilayer:end,:))*AMF);
end
airglow_spec = airglow_spec/sum(airglow_spec)/median(diff(common_grid));

airglow_spec = airglow_spec*airglow_MR/4/pi*1e12;

outp.airglow_spec = airglow_spec;