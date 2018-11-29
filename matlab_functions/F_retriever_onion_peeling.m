function savestruct = F_retriever_onion_peeling(inp)
% onion peeling for A and 1 delta O2 airglow. written by Kang Sun on
% 2018/11/29
whichband = inp.whichband;
scia_scan = inp.scia_scan;
if ~isfield(inp,'hitran_path')
    hitran_path = '/home/kangsun/CH4/airglow/O2.par.html';
else
    hitran_path = inp.hitran_path;
end
switch whichband
    case 'O2_1270'
        % limit of tangent heights, in km
        th_max = 100;
        th_min = 25;
        % list of bad pixel indices
        badpixel_indices = [56 82];
        w1_lim = [1235 1300];
        w2_lim = [1237 1298];
        w1_step = 0.0005;
        guess_slit = 1.46;
        min_photon_flux = 1e11;
        einstein_A = 2.27e-4;
        bandhead = 7883.7538;
    case 'O2_760'
        th_max = 150;
        th_min = 50;
        badpixel_indices = [];
        w1_lim = [750 780];
        w1_step = 0.0002;
        w2_lim = [752 779];       
        guess_slit = 0.5;
        min_photon_flux = 1e10;
        einstein_A = 0.08693;%https://doi.org/10.1016/j.jqsrt.2010.05.011
        bandhead = 13124;
end
w1 = w1_lim(1):w1_step:w1_lim(2);
v1 = 1e7./w1;

hitran_linelist = F_import_par(hitran_path);
% remove out of band and weak (< 1e-5 of max) lines
hitran_fieldnames = fieldnames(hitran_linelist);
int1 = hitran_linelist.transitionWavenumber >= min(v1) &...
    hitran_linelist.transitionWavenumber <= max(v1);
for ifield = 1:length(hitran_fieldnames)
    hitran_linelist.(hitran_fieldnames{ifield}) = ...
        hitran_linelist.(hitran_fieldnames{ifield})(int1,:);
end
int2 = hitran_linelist.lineIntensity > 1e-5*max(hitran_linelist.lineIntensity);
for ifield = 1:length(hitran_fieldnames)
    hitran_linelist.(hitran_fieldnames{ifield}) = ...
        hitran_linelist.(hitran_fieldnames{ifield})(int2,:);
end
% constants
% radius of earth
Re = 6371;
% some funky climatology
Z_airglow = [20;21;22;23;24;25;27.5;30;32.5;35;37.5;40;42.5;45;47.5;50;55;60;65;70;75;80;85;90;95;100;105;110;115;120;160];
P_airglow = [55.3;47.3;40.5;34.7;29.7;25.5;17.4;12;8.01;5.75;4.15;2.87114;2.06;1.49;1.09;0.798;0.425;0.219;0.109;0.0522;0.024;0.0105;0.00446;0.00184;0.00076;0.00032;0.000145;7.1e-05;4.01e-05;2.54e-05;1e-6];
T_airglow = [216.7;217.6;218.6;219.6;220.6;221.6;224;226.5;230;236.5;242.9;250.4;257.3;264.2;270.6;270.7;260.8;247;233.3;219.6;208.4;198.6;188.9;186.9;188.4;195.1;208.8;240;300;360;380];

wv = scia_scan.data(:,1);
spec = scia_scan.data(:,2:end);
spec(badpixel_indices,:) = nan;
useindex_th = find(scia_scan.header(6,:) >= th_min & ...
    scia_scan.header(6,:) <= th_max);
topn = length(useindex_th)-1;

TH_array = scia_scan.header(6,useindex_th);
SZA_array = scia_scan.header(4,useindex_th);
lat_array = scia_scan.header(1,useindex_th);
lon_array = scia_scan.header(2,useindex_th);

if strcmp(whichband,'O2_760')
    scatterbaselineindex = find(TH_array >140);
    spec_scatterbase = spec(:,scatterbaselineindex);
    for iscatter = 1:length(scatterbaselineindex)
        pp = polyfit(wv((wv >= 767 & wv <= 780) | (wv >= 750 & wv <= 759)),...
            spec_scatterbase((wv >= 767 & wv <= 780) | (wv >= 750 & wv <= 759),iscatter),1);
        bsln = polyval(pp,wv);
        spec_scatterbase(:,iscatter) = spec_scatterbase(:,iscatter)-bsln+mean(bsln);
    end
    spec = spec-mean(spec_scatterbase,2);
    
    for j = 1:size(spec,2)
        pp = polyfit(wv((wv >= 767 & wv <= 780) | (wv >= 750 & wv <= 759)),...
            spec((wv >= 767 & wv <= 780) | (wv >= 750 & wv <= 759),j),1);
        bsln = polyval(pp,wv);
        spec(:,j) = spec(:,j)-bsln;
    end
    
elseif strcmp(whichband,'O2_1270')
    for j = 1:size(spec,2)
        pp = polyfit(wv((wv >= 1300 & wv <= 1310) | (wv >= 1220 & wv <= 1235)),...
            spec((wv >= 1300 & wv <= 1310) | (wv >= 1220 & wv <= 1235),j),1);
        bsln = polyval(pp,wv);
        spec(:,j) = spec(:,j)-bsln;
    end
    
end

useindex_wv = ~isnan(spec(:,1));
spec = spec(useindex_wv,useindex_th);
wv = wv(useindex_wv);
% plot(wv,spec)
if TH_array(2)-TH_array(1) < 0
    TH_array = TH_array(end:-1:1);
    SZA_array = SZA_array(end:-1:1);
    lat_array = lat_array(end:-1:1);
    lon_array = lon_array(end:-1:1);
    spec = spec(:,end:-1:1);
end
dz = (diff(TH_array));
z_layer = TH_array(1:end-1)+dz/2;

L = zeros(topn,topn);
for ith = 1:topn
    for jth = ith:topn
        if jth == ith
            L(jth,ith) = sqrt((TH_array(ith)+dz(ith)+Re)^2-(TH_array(ith)+Re)^2);
        else
            L(jth,ith) = sqrt((TH_array(ith)+sum(dz(ith:jth))+Re)^2-(TH_array(ith)+Re)^2)...
                -sqrt((TH_array(ith)+sum(dz(ith:(jth-1)))+Re)^2-(TH_array(ith)+Re)^2);
        end
    end
end
L = L'*1e5;% km to cm;
ver_store = zeros(length(v1),topn);
absco_store = ver_store;
nO2_store = zeros(topn,1);
n1D_store = nO2_store;
n1De_store = nO2_store;
T_store = nO2_store;
Te_store = T_store;
TH_store = T_store;
zlayer_store = T_store;
s2_store = zeros(length(wv(wv > w2_lim(1) & wv < w2_lim(2))),topn);
R_store = s2_store;

% onion peeling

for ith = topn:-1:1
    TH = TH_array(ith);
    tmpspec = spec(:,ith);
    photon_flux = trapz(wv,tmpspec);
    if photon_flux < min_photon_flux
        disp(['TH = ',num2str(TH,3),' km is too dim'])
        continue
    end
    LL = L(ith,ith:topn);
    
    inp = [];
    inp.lines = hitran_linelist;
    inp.common_grid = v1;
    inp.einstein_A = einstein_A;
    inp.bandhead = bandhead;
    inp.whichband = whichband;
    inp.lineshape = 'voigt';
    guessT = interp1(Z_airglow,T_airglow,TH);
    inp.guessT = guessT;
    % inp.T = guessT;
    inp.P = interp1(Z_airglow,P_airglow,TH);
    % inp.FWHM = 1.46;
    inp.if_lbl = true;
    
    inp.w2 = wv(wv > w2_lim(1) & wv < w2_lim(2));
    s2 = double(spec(wv > w2_lim(1) & wv < w2_lim(2),ith));
    s2 = s2(:);
    coeff0 = double([log10(max(s2)/100) 0 guess_slit log10(260)]);
    inp.Tscale = 'log';
    inp.LL = LL;
    if length(LL) >= 2
        inp.absco_up = absco_store(:,end-(length(LL)-2):end);
        inp.ver_up = ver_store(:,end-(length(LL)-2):end);
    end
    inp.output_struct = false;
    try
        [coeff, R, ~,CovB] = nlinfit(inp,s2,@F_onion_peeling_fit_rad,coeff0);
        s2_store(:,ith) = s2;
        R_store(:,ith) = R;
%         figure;plot(inp.w2,s2,inp.w2,F_onion_peeling_fit_rad(coeff,inp))
        disp(['TH = ',num2str(TH,3),' km, guessed T ~ ',num2str(guessT,4),...
            ', retrieved T ~ ',num2str(10^coeff(4),4)])
    catch
        warning(['Fitting failed at Tangent Height = ',num2str(TH,3),' km'])
        break
    end
    inp.output_struct = true;
    inp.LL = LL(1);
    s2_struct = F_onion_peeling_fit_rad(coeff,inp);
    ver_store(:,ith) = s2_struct.local_ver;
    absco_store(:,ith) = s2_struct.local_absco;
    nO2_store(ith) = s2_struct.nO2;
    n1D_store(ith) = 10^coeff(1);
    n1De_store(ith) = sqrt(CovB(1,1));
    if strcmp(inp.Tscale,'log')
        T_store(ith) = 10^coeff(4);
        Te_store(ith) = 10^sqrt(CovB(4,4));
    else
        T_store(ith) = coeff(4);
        Te_store(ith) = sqrt(CovB(4,4));
    end
    %         plot(s2_struct.local_ver)
    %        4*pi*trapz(wv(~isnan(tmpspec)),tmpspec(~isnan(tmpspec)))/LL;
end
savestruct = [];
savestruct.wv = wv;
savestruct.s2 = s2_store;
savestruct.R = R_store;
savestruct.w2 = inp.w2;
savestruct.nO2 = nO2_store;
savestruct.n1D = n1D_store;
savestruct.n1De = n1De_store;
savestruct.T = T_store;
savestruct.Te = Te_store;
savestruct.TH = TH_array(1:topn);
savestruct.zlayer = z_layer(1:topn);
savestruct.einstein_A = inp.einstein_A;
savestruct.lat_scan = lat_array(1:topn);
savestruct.lon_scan = lon_array(1:topn);
savestruct.sza_scan = SZA_array(1:topn);


