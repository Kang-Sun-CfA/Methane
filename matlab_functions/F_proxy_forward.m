function s2 = F_proxy_forward(coeff,inp)
% forward function for MethaneSAT proxy retrieval. written by Kang Sun on
% 2019/02/11

w1 = inp.w1;
w2 = inp.w2;
irrad = inp.irrad;
sza = inp.sza;
vza = inp.vza;
fwhm = inp.fwhm;
count = length(inp.retrieved_molec)+1;

if strcmpi(inp.which_band,'O2')
if isfield(inp,'airglow_scale_factor')
    airglow_scale_factor = inp.airglow_scale_factor;
else
    airglow_scale_factor = coeff(count);
    count = count+1;
end
end
if isfield(inp,'LER_polynomial')
    LER_polynomial = inp.LER_polynomial;
else
    LER_polynomial = coeff(count:end);
end
optical_path = 0*w1;
for imol = 1:length(inp.retrieved_molec)
    if ~strcmpi(inp.retrieved_molec{imol},'O4')
        optical_path = optical_path+inp.([inp.retrieved_molec{imol},'_od'])*coeff(imol);
    else
        optical_path = optical_path+inp.([inp.retrieved_molec{imol},'_od'])*coeff(imol)^2;
    end
end
s1 = irrad/pi*cos(sza/180*pi) ...
    .*exp(-optical_path*(1/cos(sza/180*pi)+1/cos(vza/180*pi))) ...
    .*polyval(LER_polynomial,w1-mean(w1));
if strcmpi(inp.which_band,'O2')
    s1 = s1+inp.agspec*airglow_scale_factor;
end
s2 = F_instrument_model(w1,s1,fwhm,w2,[]);
