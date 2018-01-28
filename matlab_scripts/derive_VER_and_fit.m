% script to derive volume emission rate at layer above each tangent height,
% fit temperature for each ver spectrum
% summerized from previous code by Kang Sun on 2018/01/23. CfA VPN does not
% work today somehow.
%%
clear
close;clc
if ispc
addpath('C:\Users\Kang Sun\Documents\GitHub\Methane\matlab_functions\')
addpath('C:\Users\Kang Sun\Documents\GitHub\Realtime_FTS\matlab_script\')
lines = F_import_par('C:\data_ks\MethaneSat\O2.par.html');
load('C:\data_ks\MethaneSat\Ch6orb41467_9819-20100203_231737W-1c.mat','scia_data')
else
addpath('~/CH4/Methane/matlab_functions/')
addpath('~/FTS/Realtime_FTS/matlab_script/')
lines = F_import_par('~/CH4/airglow/O2.par.html');
load('~/CH4/airglow/Ch6orb41467_9819-20100203_231737W-1c.mat','scia_data')    
end
wStart = 1235; wEnd = 1300;step = 0.001;
w1 = wStart:step:wEnd;
v1 = 1e7./w1;
Z_airglow = [20;21;22;23;24;25;27.5;30;32.5;35;37.5;40;42.5;45;47.5;50;55;60;65;70;75;80;85;90;95;100;105;110;115;120];
P_airglow = [55.3000000000000;47.3000000000000;40.5000000000000;34.7000000000000;29.7000000000000;25.5000000000000;17.4000000000000;12;8.01000000000000;5.75000000000000;4.15000000000000;2.87114000000000;2.06000000000000;1.49000000000000;1.09000000000000;0.798000000000000;0.425000000000000;0.219000000000000;0.109000000000000;0.0522000000000000;0.0240000000000000;0.0105000000000000;0.00446000000000000;0.00184000000000000;0.000760000000000000;0.000320000000000000;0.000145000000000000;7.10000000000000e-05;4.01000000000000e-05;2.54000000000000e-05];
T_airglow = [216.700000000000;217.600000000000;218.600000000000;219.600000000000;220.600000000000;221.600000000000;224;226.500000000000;230;236.500000000000;242.900000000000;250.400000000000;257.300000000000;264.200000000000;270.600000000000;270.700000000000;260.800000000000;247;233.300000000000;219.600000000000;208.400000000000;198.600000000000;188.900000000000;186.900000000000;188.400000000000;195.100000000000;208.800000000000;240;300;360];

a = cell(30,1);
b = cell(30,1);
% how many TH to use, from bottom
topn = 12;
% radius of the earth, km
Re = 6371;
% loop through scans for this orbit
%parfor i = 1:30
for i = 1
    disp(['Working on scan ',num2str(i),'...'])
    wv = scia_data(i).data(:,1);
    spec = scia_data(i).data(:,2:end);
    f1 = wv >= 1260 & wv < 1263.5;
    f2 = wv >= 1272 & wv < 1277;
    f3 = wv >= 1280 & wv < 1285;
    for j = 1:size(spec,2)
        ref = nanmean(spec(f2,j));
        spec(f1 & spec(:,j) > 2*ref,j) = nan;
        spec(f3 & spec(:,j) > 2*ref,j) = nan;
        pp = polyfit(wv((wv >= 1300 & wv <= 1310) | (wv >= 1220 & wv <= 1235)),...
            spec((wv >= 1300 & wv <= 1310) | (wv >= 1220 & wv <= 1235),j),1);
        bsln = polyval(pp,wv);
        spec(:,j) = spec(:,j)-bsln;
    end
    TH_array = scia_data(i).header(6,:);
    SZA_array = scia_data(i).header(4,:);
    lat_array = scia_data(i).header(1,:);
    lon_array = scia_data(i).header(2,:);
    
    % derive emission spectrum at each height
    TH_array = TH_array(end:-1:1);
    SZA_array = SZA_array(end:-1:1);
    lat_array = lat_array(end:-1:1);
    lon_array = lon_array(end:-1:1);
    spec = spec(:,end:-1:1);
    
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
    L = L';
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
    X = inv(L'*inv(Sy)*L)*L'*inv(Sy)*Y;
    % loop over emission at each TH, from low TH to high TH
    fit_struct = [];
    for ith = 1:topn
        TH = TH_array(ith);
        fit_struct(ith).TH = TH;
        fit_struct(ith).z_layer = z_layer(ith);
        inp = [];
        inp.lines = lines;
        inp.common_grid = v1;
        guessT = interp1(Z_airglow,T_airglow,TH);
        % inp.T = guessT;
        inp.P = interp1(Z_airglow,P_airglow,TH);
        % inp.FWHM = 1.46;
        inp.if_lbl = true;
        disp(['Scan ',num2str(i),', TH = ',num2str(TH,3),' km, T ~ ',num2str(guessT,4),', P ~ ',num2str(inp.P,3)])
        inp.w2 = wv(wv > 1237 & wv < 1298);
        inp.if_adjust_S = true;
        inp.if_adjust_Q = false;
        s2 = double(X(ith,wv > 1237 & wv < 1298));
        s2 = s2(:);
        coeff0 = double([max(s2) 0 1.46 260]);
        try
            [coeff, R, ~,CovB] = nlinfit(inp,s2,@F_fit_airglow,coeff0);
        catch
            warning('Fitting failed!!!')
            coeff = coeff0*nan;
            R = s2*nan;
            CovB = nan(length(coeff),length(coeff));
        end
        if coeff(3) < 1
            disp('Try fixing FWHM...')
            inp.FWHM = 1.2;
            coeff0 = double([max(s2) 0 260]);
            try
                [coeff, R, ~,CovB] = nlinfit(inp,s2,@F_fit_airglow,coeff0);
                coeff = [coeff(1:2) nan coeff(3)];
                disp('Fitting succeeded with fixed fwhm')
            catch
                warning('Fitting failed with fixed fwhm')
                coeff = nan(1,4);
                R = s2*nan;
                CovB = nan(4,4);
            end
        end
        fit_struct(ith).coeff = coeff;
        fit_struct(ith).guessT = guessT;
        fit_struct(ith).P = inp.P;
        fit_struct(ith).R = R;
        fit_struct(ith).CovB = CovB;
        fit_struct(ith).s2 = s2;
    end
    a{i} = fit_struct;
    b{i} = inp;
end
if ~ispc
save('/data/wdocs/kangsun/www-docs/transfer/scia_fit_ver.mat','a','b')
end
% if ispc
% %%
% modelT = [269;268.600000000000;267.900000000000;266.700000000000;265.100000000000;263.100000000000;260.600000000000;257.700000000000;254.500000000000;251.200000000000;247.600000000000;244;240.300000000000;236.700000000000;233.200000000000;229.700000000000;226.400000000000;223.300000000000;220.400000000000;217.700000000000;215.300000000000;213.200000000000;211.300000000000;209.800000000000;208.600000000000;207.700000000000;206.900000000000;206.400000000000;206;205.700000000000;205.500000000000;205.300000000000;205.100000000000;205;204.700000000000;204.400000000000;203.900000000000;203.300000000000;202.500000000000;201.500000000000;200.200000000000;198.700000000000;197.100000000000;195.300000000000;193.500000000000;191.700000000000;189.900000000000;188.300000000000;186.900000000000;185.800000000000;184.900000000000];
% modelH = 50:100;
% %%
% fitshift = nan(length(fit_struct),1);
% fitT = nan(length(fit_struct),1);
% fitfwhm = nan(length(fit_struct),1);
% fitshifte = nan(length(fit_struct),1);
% fitTe = nan(length(fit_struct),1);
% fitfwhme = nan(length(fit_struct),1);
% guessT1 = nan(length(fit_struct),1);
% for k = 1:length(fit_struct)
%     fitshift(k) = fit_struct(k).coeff(2);
%     fitfwhm(k) = fit_struct(k).coeff(3);
%     fitT(k) = fit_struct(k).coeff(4);
%     fitTe(k) = sqrt(fit_struct(k).CovB(4,4));
%     fitshifte(k) = sqrt(fit_struct(k).CovB(2,2));
%     fitfwhme(k) = sqrt(fit_struct(k).CovB(3,3));
%     guessT1(k) = fit_struct(k).guessT;
% end
% close;
% figure('color','w','unit','inch','position',[1 1 12 5])
% subplot(1,3,1)
% h = errorbar(fitT,TH_array(1:topn),fitTe,'horizontal','-ok');
% hold on
% % h1 = plot(guessT1,TH_array(13:24),'linewidth',1);
% h1 = plot(modelT,modelH,'linewidth',1);
% hold off
% % hleg = legend([h(1) h1],'Fitted Temperature','US standard atmosphere');
% hleg = legend([h(1) h1],'Fitted Temperature','MSIS-E-90 model');
% set(hleg,'box','off')
% set(h,'linewidth',1,'markersize',5)
% set(gca,'linewidth',1)
% grid on
% xlabel('Temperature [k]')
% ylabel('Tangent height [km]')
% subplot(1,3,2)
% h = errorbar(fitfwhm,TH_array(1:topn),fitfwhme,'horizontal','-ok');
% set(h,'linewidth',1,'markersize',5)
% set(gca,'linewidth',1)
% grid on
% xlabel('Fitted slit function FWHM [nm]')
% subplot(1,3,3)
% h = errorbar(fitshift,TH_array(1:topn),fitshifte,'horizontal','-ok');
% set(h,'linewidth',1,'markersize',5)
% set(gca,'linewidth',1)
% xlabel('Fitted spectral shift [nm]')
% grid on
% %%
% addpath('C:\Users\Kang Sun\Dropbox\matlab functions\export_fig\')
% export_fig('c:\data_ks\MethaneSat\figures\scia_limb_profile_fit_ver.pdf')
% %%
% close
% Xlim = [1240 1298];
% figure('color','w','unit','inch','position',[1 1 12 5])
% k = 8;
% clear lines
% CC = lines(6);
% subplot(3,2,[1 3])
% inp.P = fit_struct(k).P;
% coeff = fit_struct(k).coeff;
% fitY = F_fit_airglow(coeff,inp);
% inp.if_adjust_S = false;
% fitsY = F_fit_airglow(coeff,inp);
% inp.if_adjust_S = true;
% R = fit_struct(k).R;
% h = plot(inp.w2,fitsY,inp.w2,R+fitY,'ok',inp.w2,fitY,'r','linewidth',1);
% set(h(1),'color',CC(1,:))
% set(h(3),'color',CC(2,:))
% set(h(2),'markersize',3,'markerfacecolor','k')
% Ylim = [0 2.6];
% set(gca,'linewidth',1,'xlim',Xlim,'ylim',Ylim,'xticklabel',[])
% str = ['(a) Tangent height = ',num2str(fit_struct(k).TH,3),' km, fit T = ',num2str(fit_struct(k).coeff(4),'%.1f'),' K'];
% title(str)
% hleg = legend(['Simulation using ',char(10),'standard line intensity'],...
%     'SCIAMACHY',['Simulation using ',char(10),'proposed line intensity']);
% set(hleg,'box','off')
% subplot(3,2,5)
% h = plot(inp.w2,(R+fitY-fitsY)/max(R+fitY),...
%     inp.w2,R/max(R+fitY),'k','linewidth',1);
% set(h(1),'color',CC(1,:))
% set(h(2),'color',CC(2,:))
% 
% set(gca,'linewidth',1,'xlim',Xlim,'ylim',[-0.06 0.06])
% xlabel('Wavelength [nm]')
% str = ['(b) Relative residual'];
% title(str)
% 
% k = 3;
% subplot(3,2,[2 4])
% inp.P = fit_struct(k).P;
% coeff = fit_struct(k).coeff;
% fitY = F_fit_airglow(coeff,inp);
% R = fit_struct(k).R;
% inp.if_adjust_S = false;
% fitsY = F_fit_airglow(coeff,inp);
% inp.if_adjust_S = true;
% R = fit_struct(k).R;
% h = plot(inp.w2,fitsY,inp.w2,R+fitY,'ok',inp.w2,fitY,'r','linewidth',1);
% set(h(1),'color',CC(1,:))
% set(h(3),'color',CC(2,:))
% set(h(2),'markersize',3,'markerfacecolor','k')
% Ylim = [0 22];
% set(gca,'linewidth',1,'xlim',Xlim,'ylim',Ylim,'xticklabel',[])
% str = ['(a) Tangent height = ',num2str(fit_struct(k).TH,3),' km, fit T = ',num2str(fit_struct(k).coeff(4),'%.1f'),' K'];
% title(str)
% hleg = legend(['Simulation using ',char(10),'standard line intensity'],...
%     'SCIAMACHY',['Simulation using ',char(10),'proposed line intensity']);
% set(hleg,'box','off')
% subplot(3,2,6)
% h = plot(inp.w2,(R+fitY-fitsY)/max(R+fitY),...
%     inp.w2,R/max(R+fitY),'k','linewidth',1);
% set(h(1),'color',CC(1,:))
% set(h(2),'color',CC(2,:))
% 
% set(gca,'linewidth',1,'xlim',Xlim,'ylim',[-0.06 0.06])
% xlabel('Wavelength [nm]')
% str = ['(d) Relative residual'];
% title(str)
% %%
% addpath('C:\Users\Kang Sun\Dropbox\matlab functions\export_fig\')
% export_fig('c:\data_ks\MethaneSat\figures\scia_spectra_fit_ver.pdf')
% end