function s2 = F_airglow_retrieval(coeff,inp_fit)
% forward function to fit simulated radiance
% written by Kang Sun on 2017/12/01
% modifed on 2017/12/22 to correct CIA AMF from quadratic to linear

w2 = inp_fit.w2;
w1 = inp_fit.w1;
ss = inp_fit.ss;
sa = inp_fit.sa;
s_o2 = inp_fit.s_o2;
s_h2o = inp_fit.s_h2o;
s_cia = inp_fit.s_cia;
scalef = inp_fit.scalef;

if inp_fit.if_fit_airglow
s1 = polyval(coeff(5:end),w1-mean(w1)).*ss.*exp(-(coeff(1)*s_o2+coeff(2)*s_cia+coeff(3)*s_h2o))+sa*coeff(4)/scalef;
s2 = F_conv_interp(w1,s1,inp_fit.fwhm,w2);
else
    s1 = polyval(coeff(4:end),w1-mean(w1)).*ss.*exp(-(coeff(1)*s_o2+coeff(2)*s_cia+coeff(3)*s_h2o));
s2 = F_conv_interp(w1,s1,inp_fit.fwhm,w2);
end

if sum(isnan(s2)) || sum(isinf(s2))
    disp('a');
end
