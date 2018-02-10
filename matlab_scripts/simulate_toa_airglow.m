clear;clc
if ispc
    addpath('C:\Users\Kang Sun\Documents\GitHub\Realtime_FTS\matlab_script\')
    addpath('C:\Users\Kang Sun\Documents\GitHub\Methane\matlab_functions\')
    lines = F_import_par('C:\data_ks\MethaneSat\O2.par.html');
    load('C:\data_ks\MethaneSat\airglow_profile.mat','Z_airglow','VER_airglow','VER_airglowe',...
        'T_airglow','T_airglowe','P_airglow')
    load('C:\Users\Kang Sun\Documents\GitHub\Realtime_FTS\spectroscopy\window_list_20160624_0000.mat')
    fn_solar = 'C:\Users\Kang Sun\Documents\GitHub\Realtime_FTS\spectroscopy\FTS_solar.mat';
else
    addpath('~/FTS/Realtime_FTS/matlab_script/')
    addpath('~/CH4/Methane/matlab_functions/')
    lines = F_import_par('/data/tempo1/Shared/kangsun/FTS_data/HITRAN/data/O2_win1_O2.data');
    load('~/CH4/airglow/airglow_profile.mat','Z_airglow','VER_airglow','VER_airglowe',...
        'T_airglow','T_airglowe','P_airglow')
    load('~/CH4/airglow/window_list.mat')
    fn_solar = '~/FTS/Realtime_FTS/spectroscopy/FTS_solar.mat';
    
end
load(fn_solar)
%%
clc
inp = [];
inp.window_list = window_list;
inp.SZA = 45;
inp.VZA = 0;
inp.vStep = 0.0005;
inp.vStart = 1240;
inp.vEnd = 1295;
useairglow_profile = 1:3:length(Z_airglow);
inp.Z_airglow = Z_airglow(useairglow_profile);
inp.VER_airglow = VER_airglow(useairglow_profile)/2;
inp.T_airglow = T_airglow(useairglow_profile);
inp.P_airglow = P_airglow(useairglow_profile);
inp.llwaven = llwaven;
inp.lltranswaven = lltranswaven;
inp.lines = lines;
inp.refl = 0.45;
inp.dlambda = 0.1;
inp.snre = 260;
inp.snrdefine_rad = 6e11;
inp.snrdefine_dlambda = 0.1;
outp = F_ro_simulator(inp);

w1 = outp.w1;
sa = outp.airglow_spectrum;
s1 = outp.s1;
%%
inpa = [];
inpa.Z_airglow = inp.Z_airglow;
inpa.VER_airglow = inp.VER_airglow;
inpa.T_airglow = inp.T_airglow;
inpa.P_airglow = inp.P_airglow;
inpa.if_adjust_S = false;
inpa.common_grid = 1e7./w1;
inpa.lines = lines;
inpa.AMF = 1/cos(inp.VZA/180*pi);
outpa = F_VER_airglow(inpa);
sa_wrong = outpa.airglow_spec;
%%
% plot(w1,s1,w1,sa,w1,sa_wrong)
%% static data for fitting
clc
wn1 = window_list(1).common_grid;
wStart = 1249;wEnd = 1290;
% wStart = 1255;wEnd = 1285;

ss = double(interp1(1e7./llwaven,lltranswaven,w1));
s_o2 = double(interp1(1e7./wn1,window_list(1).tau_struct.O2.Tau_sum,w1));
s_h2o = double(interp1(1e7./wn1,window_list(1).tau_struct.H2O.Tau_sum,w1));
s_cia = double(interp1(1e7./wn1,window_list(1).tau_struct.O2.CIA_GFIT,w1));
% plot(w1,ss,w1,exp(-s_o2),w1,exp(-s_h2o),w1,exp(-s_cia))
inp_fit = [];
inp_fit.w1 = double(w1);
inp_fit.ss = double(ss);
inp_fit.sa = double(sa);
inp_fit.s_o2 = double(s_o2);
inp_fit.s_h2o = double(s_h2o);
inp_fit.s_cia = double(s_cia);

coeff0 = [outp.amf outp.amf outp.amf 1 0 1];
snre_vec = [50:50:500];
nsnre = length(snre_vec);

dlambda_vec = 0.01:0.01:0.1;%[0.01 0.02 0.025 0.035 0.05 0.06 0.075 0.09 0.1];
ndlambda = length(dlambda_vec);

realizeN = 250;
%% know-everything fitting
c1 = nan(length(coeff0),realizeN,nsnre,ndlambda);

for idlambda = 1:ndlambda
    inp.dlambda = dlambda_vec(idlambda);
    
    for isnr = 1:nsnre
        inp.snre = snre_vec(isnr);
        inp.snrdefine_rad = 6e11;
        inp.snrdefine_dlambda = 0.1;
        inp.s1 = s1;
        inp.w1 = w1;
        inp.airglow_spectrum = sa;
        
        tmp_coeff = nan(length(coeff0),realizeN);
        
        parfor irealize = 1:realizeN
            inp_fit_local = inp_fit;
            outp_fit = F_ro_simulator(inp);
            % Weight = outp.noise_std(outp.w2 >= wStart & outp.w2 <= wEnd);
            w2 = double(outp_fit.w2(outp_fit.w2 >= wStart & outp_fit.w2 <= wEnd));
            s2 = double(outp_fit.s2(outp_fit.w2 >= wStart & outp_fit.w2 <= wEnd));
            % Weight = Weight/sum(Weight);
            inp_fit_local.scalef = mean(s2);
            s2 = s2/inp_fit_local.scalef;
            inp_fit_local.fwhm = inp.dlambda*3;
            
            inp_fit_local.if_fit_airglow = true;
            
            inp_fit_local.w2 = w2;
            inp_fit_local.s2 = double(s2);
            coeff = nlinfit(inp_fit_local,s2,@F_airglow_retrieval,coeff0);%,'weight',Weight);
            tmp_coeff(:,irealize) = coeff;
        end
        c1(:,:,isnr,idlambda) = tmp_coeff;
        nowst = datestr(now);
        disp(['SNR = ',num2str(snre_vec(isnr)),', dl = ',num2str(dlambda_vec(idlambda)),' finshed at ',nowst])
    end
end
save('c1.mat','c1','dlambda_vec','snre_vec','ndlambda','nsnre')
%% no-airglow fitting
c2 = nan(length(coeff0),realizeN,nsnre,ndlambda);

for idlambda = 1:ndlambda
    inp.dlambda = dlambda_vec(idlambda);
    
    for isnr = 1:nsnre
        inp.snre = snre_vec(isnr);
        inp.snrdefine_rad = 6e11;
        inp.snrdefine_dlambda = 0.1;
        inp.s1 = s1;
        inp.w1 = w1;
        inp.airglow_spectrum = sa;
        
        tmp_coeff = nan(length(coeff0),realizeN);
        
        parfor irealize = 1:realizeN
            inp_fit_local = inp_fit;
            outp_fit = F_ro_simulator(inp);
            % Weight = outp.noise_std(outp.w2 >= wStart & outp.w2 <= wEnd);
            w2 = double(outp_fit.w2(outp_fit.w2 >= wStart & outp_fit.w2 <= wEnd));
            s2 = double(outp_fit.s2(outp_fit.w2 >= wStart & outp_fit.w2 <= wEnd));
            % Weight = Weight/sum(Weight);
            inp_fit_local.scalef = mean(s2);
            s2 = s2/inp_fit_local.scalef;
            inp_fit_local.fwhm = inp.dlambda*3;
            
            inp_fit_local.if_fit_airglow = false;
            
            inp_fit_local.w2 = w2;
            inp_fit_local.s2 = double(s2);
            coeff = nlinfit(inp_fit_local,s2,@F_airglow_retrieval,coeff0);%,'weight',Weight);
            tmp_coeff(:,irealize) = coeff;
        end
        c2(:,:,isnr,idlambda) = tmp_coeff;
        nowst = datestr(now);
        disp(['SNR = ',num2str(snre_vec(isnr)),', dl = ',num2str(dlambda_vec(idlambda)),' finshed at ',nowst])
    end
end
save('c2.mat','c2','dlambda_vec','snre_vec','ndlambda','nsnre')
%% gravity wave fitting
c4 = nan(length(coeff0),realizeN,nsnre,ndlambda);

for idlambda = 1:ndlambda
    inp.dlambda = dlambda_vec(idlambda);
    
    for isnr = 1:nsnre
        inp.snre = snre_vec(isnr);
        inp.snrdefine_rad = 6e11;
        inp.snrdefine_dlambda = 0.1;
        if isfield(inp,'s1')
            inp = rmfield(inp,'s1');
        end
        if isfield(inp,'w1')
            inp = rmfield(inp,'w1');
        end
        if isfield(inp,'airglow_spectrum')
            inp = rmfield(inp,'airglow_spectrum');
        end
        
        tmp_coeff = nan(length(coeff0),realizeN);
        
        parfor irealize = 1:realizeN
            inpsimlocal = inp;
            inpsimlocal.T_airglow = ...
                inpsimlocal.T_airglow+...
                randn(size(inpsimlocal.T_airglow)).*T_airglowe(useairglow_profile);
            inp_fit_local = inp_fit;
            outp_fit = F_ro_simulator(inpsimlocal);
            % Weight = outp.noise_std(outp.w2 >= wStart & outp.w2 <= wEnd);
            w2 = double(outp_fit.w2(outp_fit.w2 >= wStart & outp_fit.w2 <= wEnd));
            s2 = double(outp_fit.s2(outp_fit.w2 >= wStart & outp_fit.w2 <= wEnd));
            % Weight = Weight/sum(Weight);
            inp_fit_local.scalef = mean(s2);
            s2 = s2/inp_fit_local.scalef;
            inp_fit_local.fwhm = inp.dlambda*3;
            
            inp_fit_local.if_fit_airglow = true;
            
            inp_fit_local.w2 = w2;
            inp_fit_local.s2 = double(s2);
            coeff = nlinfit(inp_fit_local,s2,@F_airglow_retrieval,coeff0);%,'weight',Weight);
            tmp_coeff(:,irealize) = coeff;
        end
        c4(:,:,isnr,idlambda) = tmp_coeff;
        nowst = datestr(now);
        disp(['SNR = ',num2str(snre_vec(isnr)),', dl = ',num2str(dlambda_vec(idlambda)),' finshed at ',nowst])
    end
end
save('c4.mat','c4','dlambda_vec','snre_vec','ndlambda','nsnre')

%% wrong-airglow fitting
c3 = nan(length(coeff0),realizeN,nsnre,ndlambda);
inp_fit.sa = sa_wrong;
for idlambda = 1:ndlambda
    inp.dlambda = dlambda_vec(idlambda);
    
    for isnr = 1:nsnre
        inp.snre = snre_vec(isnr);
        inp.snrdefine_rad = 6e11;
        inp.snrdefine_dlambda = 0.1;
        inp.s1 = s1;
        inp.w1 = w1;
        inp.airglow_spectrum = sa;
        
        tmp_coeff = nan(length(coeff0),realizeN);
        
        parfor irealize = 1:realizeN
            inp_fit_local = inp_fit;
            outp_fit = F_ro_simulator(inp);
            % Weight = outp.noise_std(outp.w2 >= wStart & outp.w2 <= wEnd);
            w2 = double(outp_fit.w2(outp_fit.w2 >= wStart & outp_fit.w2 <= wEnd));
            s2 = double(outp_fit.s2(outp_fit.w2 >= wStart & outp_fit.w2 <= wEnd));
            % Weight = Weight/sum(Weight);
            inp_fit_local.scalef = mean(s2);
            s2 = s2/inp_fit_local.scalef;
            inp_fit_local.fwhm = inp.dlambda*3;
            
            inp_fit_local.if_fit_airglow = true;
            
            inp_fit_local.w2 = w2;
            inp_fit_local.s2 = double(s2);
            coeff = nlinfit(inp_fit_local,s2,@F_airglow_retrieval,coeff0);%,'weight',Weight);
            tmp_coeff(:,irealize) = coeff;
        end
        c3(:,:,isnr,idlambda) = tmp_coeff;
        nowst = datestr(now);
        disp(['SNR = ',num2str(snre_vec(isnr)),', dl = ',num2str(dlambda_vec(idlambda)),' finshed at ',nowst])
    end
end
save('c3.mat','c3','dlambda_vec','snre_vec','ndlambda','nsnre')
%%
% plot(T_airglow(useairglow_profile),Z_airglow(useairglow_profile),...
%     T_airglow(useairglow_profile)+...
%     randn(size(T_airglow(useairglow_profile))).*T_airglowe(useairglow_profile),...
%     Z_airglow(useairglow_profile),'o')
%%
% o2_std = nan(nsnre,ndlambda,1);
% for idlambda = 1:ndlambda
%     for isnr = 1:nsnre
%         for i = 1
%             c = eval(['c',num2str(i)]);
%             tmpd = squeeze(c(:,:,isnr,idlambda));
%             meand = nanmean(tmpd,2);
%             stdd = nanstd(tmpd,0,2);
%             o2_std(isnr,idlambda,i) = stdd(1);
%             
%         end
%     end
% end
% %%
% h = contourf(dlambda_vec,snre_vec,squeeze(o2_std)/outp.amf);
% colorbar
% %%
% close
% subplot(2,1,1)
% plot(w2,s2,'k',w2,F_airglow_retrieval(coeff,inp_fit),'r')
% subplot(2,1,2)
% plot(w2,s2-F_airglow_retrieval(coeff,inp_fit))
% %%
% (coeff(1)-outp.amf)/outp.amf*100