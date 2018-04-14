function s2 = F_fit_airglow(coeff,inp)
% forward model to fit fwhm and temperature using scia limb airglow spectra
% written by Kang Sun on 2018/01/18
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

if ~isfield(inp,'T')
    count = count+1;
    inp.T = coeff(count);
end
% force the convolution to be off within F_O21D_hitran
inp.fwhm = 0;
w2 = inp.w2;
outp = F_O21D_hitran_A(inp);
w1 = outp.wgrid;
s1 = outp.xsec;
s1 = s1/max(s1);

slit = FWHM/1.66511;% half width at 1e

dw0 = median(diff(w1));
ndx = ceil(slit*2.7/dw0);
xx = (0:ndx*2)*dw0-ndx*dw0;
kernel = exp(-(xx/slit).^2);
kernel = kernel/sum(kernel);
s1_over = conv(s1, kernel, 'same');
s2 = interp1(w1,s1_over,w2+shift,'linear','extrap')*coeff(1);

% s2 = F_conv_interp(w1,s1,FWHM,w2)*coeff(1);
s2 = double(s2(:));

