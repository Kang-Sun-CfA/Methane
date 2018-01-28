function s2 = F_fit_absorbed_airglow(coeff,inp)
% consider self-absorption in onion-peeling, modified from F_fit_airglow.m
% by Kang Sun on 2018/01/27

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
if FWHM >= 1.5
    FWHM = 1.5;
end
if FWHM <= 1
    FWHM = 1;
end
if ~isfield(inp,'T')
    count = count+1;
    if strcmp(inp.Tscale,'log')
    inp.T = 10^coeff(count);
    else
        inp.T = coeff(count);
    end
end

% force the convolution to be off within F_O21D_hitran
inp.fwhm = 0;
w2 = inp.w2;
% simulate emission spectrum
inp.if_adjust_S = true;
inp.if_adjust_Q = false;
outp = F_O21D_hitran(inp);
w1 = outp.wgrid;
s1 = outp.xsec;
% ver spectrum in photons/s/cm3/nm
ver_spec = s1/sum(s1)/mean(diff(w1))*10^coeff(1)*inp.A1D;
% simulate absorption spectrum
inp.if_adjust_S = false;
outp = F_O21D_hitran(inp);

LL = inp.LL;
% local optical depth
tau_all = 0.5*outp.xsec*outp.N*LL(1);
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
    outp_struct.nO2 = outp.N;
    outp_struct.local_absco = outp.xsec*outp.N;
    outp_struct.local_ver = ver_spec;
    outp_struct.w1 = w1;
    outp_struct.s2 = s2;
    s2 = outp_struct;
end