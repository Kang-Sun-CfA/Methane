% fit O2 1D number density and temperature using onion peeling. Written by
% Kang Sun on 2018/01/27

% clc
% clear
% fn = 'C:\data_ks\MethaneSat\Ch4orb44801_0000-20100924_211353Y-1c.xlsx';
% sheet = 'Ch4orb44801_0000-20100924_21135';
% 
% wvstart = 7;wvend = 979;
% scia_data = [];
% for i = 1:30
% startl = 2+(i-1)*980;
% endl = 1+i*980;
% xlrange = ['A',num2str(startl),':AG',num2str(endl)];
% [num,txt] = xlsread(fn,sheet,xlrange);
% scia_data(i).header = single(num(1:6,2:end));
% tmpwv = num(7:979,1);
% tmpdata = num(7:979,:);
% scia_data(i).data = double(tmpdata(tmpwv >= 1220 & tmpwv <= 1320,:));
% end
% save('C:\data_ks\MethaneSat\Ch4orb44801_0000-20100924_211353Y-1c.mat','scia_data')
% plot(num(wvstart:wvend,1),num(wvstart:wvend,2:end));xlim([1240 1300])
%%
clear
close;clc
if ispc
addpath('C:\Users\Kang Sun\Documents\GitHub\Methane\matlab_functions\')
addpath('C:\Users\Kang Sun\Documents\GitHub\Realtime_FTS\matlab_script\')
lines = F_import_par('C:\data_ks\MethaneSat\O2.par.html');
load('C:\data_ks\MethaneSat\Ch4orb44801_0000-20100924_211353Y-1c.mat','scia_data')
else
addpath('~/CH4/Methane/matlab_functions/')
addpath('~/FTS/Realtime_FTS/matlab_script/')
lines = F_import_par('~/CH4/airglow/O2.par.html');
load('~/CH4/airglow/Ch4orb44801_0000-20100924_211353Y-1c.mat','scia_data')    
end
wStart = 1235; wEnd = 1300;step = 0.001;
w1 = wStart:step:wEnd;
v1 = 1e7./w1;
Z_airglow = [20;21;22;23;24;25;27.5;30;32.5;35;37.5;40;42.5;45;47.5;50;55;60;65;70;75;80;85;90;95;100;105;110;115;120];
P_airglow = [55.3000000000000;47.3000000000000;40.5000000000000;34.7000000000000;29.7000000000000;25.5000000000000;17.4000000000000;12;8.01000000000000;5.75000000000000;4.15000000000000;2.87114000000000;2.06000000000000;1.49000000000000;1.09000000000000;0.798000000000000;0.425000000000000;0.219000000000000;0.109000000000000;0.0522000000000000;0.0240000000000000;0.0105000000000000;0.00446000000000000;0.00184000000000000;0.000760000000000000;0.000320000000000000;0.000145000000000000;7.10000000000000e-05;4.01000000000000e-05;2.54000000000000e-05];
T_airglow = [216.700000000000;217.600000000000;218.600000000000;219.600000000000;220.600000000000;221.600000000000;224;226.500000000000;230;236.500000000000;242.900000000000;250.400000000000;257.300000000000;264.200000000000;270.600000000000;270.700000000000;260.800000000000;247;233.300000000000;219.600000000000;208.400000000000;198.600000000000;188.900000000000;186.900000000000;188.400000000000;195.100000000000;208.800000000000;240;300;360];

useindex = 11:31;
topn = length(useindex)-1;
% radius of the earth, km
Re = 6371;
nscan = length(scia_data);
a = cell(nscan,1);
parfor i = 1:nscan
    disp(['Working on scan ',num2str(i),'...'])
    wv = scia_data(i).data(:,1);
    spec = scia_data(i).data(:,2:end);
    spec([56 82],:) = nan;
%     f1 = wv >= 1260 & wv < 1263.5;
%     f2 = wv >= 1272 & wv < 1277;
%     f3 = wv >= 1280 & wv < 1285;
    for j = 1:size(spec,2)
%         ref = nanmean(spec(f2,j));
%         spec(f1 & spec(:,j) > 2*ref,j) = nan;
%         spec(f3 & spec(:,j) > 2*ref,j) = nan;
        pp = polyfit(wv((wv >= 1300 & wv <= 1310) | (wv >= 1220 & wv <= 1235)),...
            spec((wv >= 1300 & wv <= 1310) | (wv >= 1220 & wv <= 1235),j),1);
        bsln = polyval(pp,wv);
        spec(:,j) = spec(:,j)-bsln;
    end    
    spec = spec(:,useindex);
    int = ~isnan(spec(:,1));
    spec = spec(int,:);
    wv = wv(int);
    
    TH_array = scia_data(i).header(6,useindex);
    SZA_array = scia_data(i).header(4,useindex);
    lat_array = scia_data(i).header(1,useindex);
    lon_array = scia_data(i).header(2,useindex);
    
%     % adjust top level
%     TH_array(end) = 2*TH_array(end-1)-TH_array(end-2);
    
    % linear inversion, no self absorption
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
    Y = spec(:,1:topn)';
    Syd = var(Y(:,wv < 1240 | wv > 1300),[],2);
    Sy = double(diag(Syd));
    % Sy = zeros(topn:topn);
    % if the error at different TH are correlated, may not be the case
    % for ith = 1:topn
    %     for jth = 1:ith
    %     Sy(ith,jth) = sqrt(Syd(ith)*Syd(jth))*exp(-abs(ith-jth)/10);
    %     Sy(jth,ith) = Sy(ith,jth);
    %     end
    % end
    % Sy = eye(23,23);
    X = 4*pi*inv(L'*inv(Sy)*L)*L'*inv(Sy)*Y;
    
    ver_store = zeros(length(v1),topn);
    absco_store = ver_store;
    nO2_store = zeros(topn,1);
    n1D_store = nO2_store;
    n1De_store = nO2_store;
    T_store = nO2_store;
    Te_store = T_store;
    TH_store = T_store;
    zlayer_store = T_store;
    % onion peeling
    for ith = topn:-1:1
        TH = TH_array(ith);
        TH_store(ith) = TH;
        zlayer_store(ith) = z_layer(ith);
        tmpspec = spec(:,ith);
        LL = L(ith,ith:topn);
        
        inp = [];
        inp.lines = lines;
        inp.common_grid = v1;
        inp.A1D = 2e-4;
        guessT = interp1(Z_airglow,T_airglow,TH);
        % inp.T = guessT;
        inp.P = interp1(Z_airglow,P_airglow,TH);
        % inp.FWHM = 1.46;
        inp.if_lbl = true;
        disp(['Scan ',num2str(i),', TH = ',num2str(TH,3),' km, T ~ ',num2str(guessT,4),', P ~ ',num2str(inp.P,3)])
        inp.w2 = wv(wv > 1237 & wv < 1298);
        s2 = double(spec(wv > 1237 & wv < 1298,ith));
        s2 = s2(:);
        coeff0 = double([log10(max(s2)/100) 0 1.46 log10(260)]);
        inp.LL = LL;
        if length(LL) >= 2
            inp.absco_up = absco_store(:,end-(length(LL)-2):end);
            inp.ver_up = ver_store(:,end-(length(LL)-2):end);
        end
        inp.output_struct = false;
        try
        [coeff, R, ~,CovB] = nlinfit(inp,s2,@F_fit_absorbed_airglow,coeff0);
        catch
            warning(['Fitting failed at Scan ',num2str(i),', TH = ',num2str(TH,3),' km'])
            break
        end
%         ss2 = F_fit_absorbed_airglow(coeff0,inp);
        inp.output_struct = true;
        inp.LL = LL(1);
        s2_struct = F_fit_absorbed_airglow(coeff,inp);
        ver_store(:,ith) = s2_struct.local_ver;
        absco_store(:,ith) = s2_struct.local_absco;
        nO2_store(ith) = s2_struct.nO2;
        n1D_store(ith) = 10^coeff(1);
        n1De_store(ith) = sqrt(CovB(1,1));
        T_store(ith) = 10^coeff(4);
        Te_store(ith) = sqrt(CovB(4,4));
%         plot(s2_struct.local_ver)
%        4*pi*trapz(wv(~isnan(tmpspec)),tmpspec(~isnan(tmpspec)))/LL; 
    end
    
    savestruct = [];
    savestruct.X = single(X);
    savestruct.wv = wv;
    savestruct.nO2 = nO2_store;
    savestruct.n1D = n1D_store;
    savestruct.n1De = n1De_store;
    savestruct.T = T_store;
    savestruct.Te = Te_store;
    savestruct.TH = TH_store;
    savestruct.zlayer = zlayer_store;
    savestruct.A1D = inp.A1D;
    savestruct.lat_scan = lat_array(1:topn);
    savestruct.lon_scan = lon_array(1:topn);
    savestruct.sza_scan = sza_array(1:topn);
    a{i} = savestruct;
end
if ~ispc
save('/data/wdocs/kangsun/www-docs/transfer/onion_peeling.mat','a')
end
