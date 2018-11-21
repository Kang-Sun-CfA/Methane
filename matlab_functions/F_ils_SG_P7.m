function yy = F_ils_SG_P7(inp)
% parameterize ILS using super gaussian and pearson type VII distribution.
% simplified from tropomi: https://www.atmos-meas-tech.net/11/3917/2018/amt-11-3917-2018.pdf
% written by Kang Sun on 2018/11/07
% add asymmetry on 2018/11/10
% clc
% fwhm = 5;
% xx = -10:.1:10;
% r_to_w = 1;
% k = 3;
fwhm = inp.fwhm;
xx = inp.xx;
if ~isfield(inp,'m')
    m = 1;% m should > 0.5, m = 1 means lorentz; r is hwhm
else
    m = inp.m;
end
if ~isfield(inp,'r_to_w')
    r_to_w = 1;
else
    r_to_w = inp.r_to_w;
end
if ~isfield(inp,'k')
    k = 2;
else
    k = inp.k;
end
if ~isfield(inp,'aw')
    aw = 0;
else
    aw = inp.aw;
end
if ~isfield(inp,'eta')
    eta = 0.12;
else
    eta = inp.eta;
end
w = fwhm/2/log(2)^(1/k);
r = w*r_to_w;
% F_SG_P7 = @(w,k,aw,m,r,eta,xx) (1-eta)*F_SG(w,k,xx,aw) + eta*F_P7(m,r,xx);
yy0 = F_SG_P7(w,k,aw,m,r,eta,xx);

yy = interp1(fwhm/F_fwhm(xx,yy0)*xx,yy0,xx,'linear','extrap');

function yy0 = F_SG_P7(w,k,aw,m,r,eta,xx)
F_SG = @(w,k,xx,aw) ...
    k/(2*w*gamma(1/k))*exp(-abs((xx./(aw*w*sign(xx)+w*ones(size(xx))))).^k);
F_P7 = @(m,r,xx) gamma(m)/r/sqrt(pi)/gamma(m-0.5)*(1+(xx.^2)/r^2).^(-m);
y = F_SG(w,k,xx,aw);
if aw ~= 0
    centerofmass = nansum(y.*xx)/nansum(y);
    y = interp1(xx-centerofmass,y,xx,'linear','extrap');
end
yy0 = (1-eta)*y + eta*F_P7(m,r,xx);
% F_fwhm(xx,yy)
% subplot(1,2,1)
% plot(xx,F_SG_P7(w,k,m,r,eta,xx),xx,F_SG(w,k,xx),xx,yy,'.')
% subplot(1,2,2)
% semilogy(xx,F_SG_P7(w,k,m,r,eta,xx))

