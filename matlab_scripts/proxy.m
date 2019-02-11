clc;clear
which_linux = 'CF';
% I'm jumping back and forth between three computers
if ispc
    git_dir = 'C:\Users\Kang Sun\Documents\GitHub\Methane\';
    spv_dir = 'C:\data_ks\MethaneSat\';
    O2par_path = 'C:\data_ks\MethaneSat\O2.par.html';
    CIA_xsec_path = 'C:\data_ks\MethaneSat\O2_CIA_296K_all.dat';
    aod_vec = 0.001;% 0.01 0.05 0.1 0.2 0.5 1];
    peakz_vec = 0.25;% 0.5 1 2 5 10];
    save_dir = spv_dir;
else
    switch which_linux
        case 'UB'
            git_dir = '~/CH4/Methane/';
            spv_dir = '/mnt/Data2/gctool_data/spv_outp/';
            O2par_path = '~/CH4/O2.par.html';
            CIA_xsec_path = '~/CH4/O2_CIA_296K_all.dat';
            aod_vec = 0.001;% 0.01 0.05 0.1 0.2 0.5 1];
            peakz_vec = 0.25;% 0.5 1 2 5 10];
            save_dir = spv_dir;
        case 'CF'
            git_dir = '~/CH4/Methane/';
            spv_dir = '/data/tempo1/Shared/kangsun/spv/outp/';
            O2par_path = '~/CH4/O2.par.html';
            CIA_xsec_path = '~/CH4/O2_CIA_296K_all.dat';
            aod_vec = [0.001 0.01 0.05 0.1 0.2 0.5 1];
            peakz_vec = [0.25 0.5 1 2 5 10];
            save_dir = '/data/wdocs/kangsun/www-docs/transfer/';
    end
end
addpath([git_dir,'matlab_functions'])

% some inputs, from ball
dx = 0.126;
dt = 1/17.5;
D = 4.37;
dx0 = 0;

nmc = 10;
naod = length(aod_vec);
npeakz = length(peakz_vec);
coeff_mat_O2 = nan(10,naod,npeakz,nmc,'single');
coeff_mat_CH4 = nan(10,naod,npeakz,nmc,'single');

for iaod = 1:naod
    for ipeakz = 1:npeakz
        
        if_lnR = false;
        inp_nc = [];
        inp_nc.fn = [spv_dir,'proxy_CH4_1603-1690_0.005_',num2str(aod_vec(iaod),'%.3f'),...
            '_',num2str(peakz_vec(ipeakz),'%.2f'),'_GC_upwelling_output.nc'];
        inp_nc.if_lnR = if_lnR;
        outp_CH4 = F_read_spv_output(inp_nc);
        
        inpd = [];
        inpd.nwin = 1;
        inpd.wmins = 1606;
        inpd.wmaxs = 1689;
        inpd.nsamp = 3;
        inpd.fwhm = 0.189;
        inpd.nalb = 2;
        inpn_CH4 = [];
        % readout noise, e per pixel
        inpn_CH4.Nr = 50;
        % dark current, e per s per pixel.
        inpn_CH4.Nd_per_s = 10000;
        % orbit height, km
        inpn_CH4.H = 617;
        % integration time, s. 1/7 means integration through 1 km
        inpn_CH4.dt = dt;
        inpn_CH4.eta = 0.4;
        % ground fov for single pixel, across track, km
        inpn_CH4.dx = dx;
        % ground fov for single pixel, along track, km
        inpn_CH4.dy = inpn_CH4.dx*inpd.nsamp(1);
        % aggregated across-track pixel size, km
        inpn_CH4.dx0 = dx0;
        % aggregated along-track pixel size, km
        inpn_CH4.dy0 = dx0;
        % aperture size, cm2
        inpn_CH4.A = pi*D^2/4;
        
        inpd.inpn = {inpn_CH4};
        inpd.spv_output = {outp_CH4};
        outp_CH4_d = F_degrade_spv_output(inpd);
        outp_CH4_d.radn = outp_CH4_d.rad+normrnd(0*outp_CH4_d.rad,outp_CH4_d.rad./outp_CH4_d.wsnr);
        % plot(outp_CH4_d.wave,outp_CH4_d.rad,outp_CH4_d.wave,outp_CH4_d.radn)
        
        inp_nc = [];
        inp_nc.fn = [spv_dir,'proxy_O2_1245-1295_0.003_',num2str(aod_vec(iaod),'%.3f'),...
            '_',num2str(peakz_vec(ipeakz),'%.2f'),'_GC_upwelling_output.nc'];
        inp_nc.if_lnR = if_lnR;
        inp_nc.O2par_path = O2par_path;
        inp_nc.O21D_col = 2e17;
        inp_nc.VZA = outp_CH4.VZA;
        outp_O2 = F_read_spv_output(inp_nc);
        
        inpd = [];
        inpd.nwin = 1;
        inpd.wmins = 1246;
        inpd.wmaxs = 1293;
        inpd.nsamp = 3;
        inpd.fwhm = 0.15;
        inpd.nalb = 2;
        inpn_O2 = [];
        % readout noise, e per pixel
        inpn_O2.Nr = 50;
        % dark current, e per s per pixel.
        inpn_O2.Nd_per_s = 10000;
        % orbit height, km
        inpn_O2.H = 617;
        % integration time, s. 1/7 means integration through 1 km
        inpn_O2.dt = dt;
        inpn_O2.eta = 0.4;
        % ground fov for single pixel, across track, km
        inpn_O2.dx = dx;
        % ground fov for single pixel, along track, km
        inpn_O2.dy = inpn_O2.dx*inpd.nsamp(1);
        % aggregated across-track pixel size, km
        inpn_O2.dx0 = dx0;
        % aggregated along-track pixel size, km
        inpn_O2.dy0 = dx0;
        % aperture size, cm2
        inpn_O2.A = pi*D^2/4;
        
        inpd.inpn = {inpn_O2};
        inpd.spv_output = {outp_O2};
        outp_O2_d = F_degrade_spv_output(inpd);
        % plot(outp_O2_d.wave,outp_O2_d.rad,outp_O2_d.wave,outp_O2_d.radn)
        %%
        for imc = 1:nmc
            % prepare input for O2 window
            w_start = 1249;
            w_end = 1289;
            retrieved_molec = {'O2','O4','H2O'};
            tmp = outp_O2;
            inp = [];
            inp.which_band = 'O2';
            inp.retrieved_molec = retrieved_molec;
            inp.irrad = tmp.irrad;
            inp.w1 = tmp.wave;
            inp.sza = tmp.sza;
            inp.vza = tmp.vza;
            inp.fwhm = 0.15;
            % inp.airglow_scale_factor = 1;
            % inp.LER_polynomial = 0.25;
            inp.agspec = outp_O2.agspec*outp_O2.O21D_col;
            wint = outp_O2_d.wave >= w_start & outp_O2_d.wave <= w_end;
            inp.w2 = outp_O2_d.wave(wint);
            s2 = double(outp_O2_d.rad+normrnd(0*outp_O2_d.rad,outp_O2_d.rad./outp_O2_d.wsnr));
            s2 = s2(wint);
            for imol = 1:length(retrieved_molec)
                if ~strcmpi(retrieved_molec{imol},'O4')
                    inp.([retrieved_molec{imol},'_od']) = sum(repmat(...
                        tmp.([retrieved_molec{imol},'_gascol'])*...
                        tmp.gasnorm.(retrieved_molec{imol}),[1,tmp.nw])' ...
                        .*tmp.([retrieved_molec{imol},'_gas_xsec']),2);
                else
                    ciadata = importdata(CIA_xsec_path);
                    O4_xsec = interp1(ciadata(:,1),ciadata(:,2),outp_O2.wave);
                    inp.O4_od = sum(repmat(...
                        (double(tmp.O2_gascol)*double(tmp.gasnorm.O2)).^2 ...
                        ./abs(diff(1e5*double(tmp.zs0))),[1,tmp.nw])' ...
                        .*repmat(O4_xsec(:),[1,tmp.nz]),2);
                end
            end
            coeff0 = [1 1 1 1 0 .25];
            % s_prior = F_proxy_forward(coeff0,inp);
            [coeff,~] = nlinfit(inp,s2,@F_proxy_forward,coeff0);
            % s_posterior = F_proxy_forward(coeff,inp);
            % plot(inp.w2,F_proxy_forward(coeff0,inp),'b',inp.w2,s2,'k',inp.w2,s2-R,'r')
            % coeff
            % plot(outp_O2.wave,sum(outp_O2.ods,2),outp_O2.wave,inp.H2O_od+inp.O2_od+inp.O4_od)
            coeff_mat_O2(1:length(coeff),iaod,ipeakz,imc) = coeff;
            %%
            % prepare input for CH4 window
            w_start = 1606;
            w_end = 1689;
            retrieved_molec = {'CH4','CO2','H2O'};
            tmp = outp_CH4;
            inp = [];
            inp.which_band = 'CH4';
            inp.retrieved_molec = retrieved_molec;
            inp.irrad = tmp.irrad;
            inp.w1 = tmp.wave;
            inp.sza = tmp.sza;
            inp.vza = tmp.vza;
            inp.fwhm = 0.189;
            % inp.LER_polynomial = 0.25;
            wint = outp_CH4_d.wave >= w_start & outp_CH4_d.wave <= w_end;
            inp.w2 = outp_CH4_d.wave(wint);
            s2 = double(outp_CH4_d.rad+normrnd(0*outp_CH4_d.rad,outp_CH4_d.rad./outp_CH4_d.wsnr));
            s2 = s2(wint);
            for imol = 1:length(retrieved_molec)
                if ~strcmpi(retrieved_molec{imol},'O4')
                    inp.([retrieved_molec{imol},'_od']) = sum(repmat(...
                        tmp.([retrieved_molec{imol},'_gascol'])*...
                        tmp.gasnorm.(retrieved_molec{imol}),[1,tmp.nw])' ...
                        .*tmp.([retrieved_molec{imol},'_gas_xsec']),2);
                else
                    ciadata = importdata(CIA_xsec_path);
                    O4_xsec = interp1(ciadata(:,1),ciadata(:,2),outp_O2.wave);
                    inp.O4_od = sum(repmat(...
                        (double(tmp.O2_gascol)*double(tmp.gasnorm.O2)).^2 ...
                        ./abs(diff(1e5*double(tmp.zs0))),[1,tmp.nw])' ...
                        .*repmat(O4_xsec(:),[1,tmp.nz]),2);
                end
            end
            coeff0 = [1 1 1  0 .25];
            % s_prior = F_proxy_forward(coeff0,inp);
            [coeff,~] = nlinfit(inp,s2,@F_proxy_forward,coeff0);
%             plot(inp.w2,F_proxy_forward(coeff0,inp),'b',inp.w2,s2,'k',inp.w2,s2-R,'r')
            coeff_mat_CH4(1:length(coeff),iaod,ipeakz,imc) = coeff;
        end
    end
end
save([save_dir,'proxy_coeff.mat'],'coeff_mat_CH4','coeff_mat_O2')