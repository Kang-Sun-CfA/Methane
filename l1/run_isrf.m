clc;close all;clear
% Kang's pc: 'pc'; UB work station: 'UB'; add new options by expanding the
% switch structure below
whichMachine = 'UB';
% 'CH4' or 'O2'
whichBand = 'CH4';
% true if want to plot detailed information, false for running all cases
ifPlotDiagnose = false;
% do stray light correction or not
doStrayLight = true;
% number of iterations in stray light correction
niter = 3;
switch whichMachine
    case 'pc'
        addpath('c:\Users\kangsun\Dropbox\matlab functions\export_fig\')
        ch4_bad_pix_fn = 'C:\research\CH4\stray_light\CH4_bad_pix.csv';
        ch4_rad_cal_fn = 'C:\research\CH4\stray_light\rad_coef_ch4_50ms_ord4_20200209T211211.mat';
        ch4_median_straylight_fn = 'C:\research\CH4\stray_light\K_stable_CH4.mat';
        o2_median_straylight_fn = 'C:\research\CH4\stray_light\K_stable_O2.mat';
        % data directory containing each wavelength as a directory
        ils_dir = 'C:\research\CH4\ils_20200127\';
        code_dir = ils_dir;
        output_dir = ils_dir;
    case 'UB'
        ch4_bad_pix_fn = '/home/kangsun/CH4/MethaneAIR/CH4_bad_pix.csv';
        o2_bad_pix_fn = '/home/kangsun/CH4/MethaneAIR/O2_bad_pix.csv';
        ch4_rad_cal_fn = '/home/kangsun/CH4/MethaneAIR/rad_coef_50ms/rad_coef_ch4_50ms_ord4_20200209T211211.mat';
        o2_rad_cal_fn = '/home/kangsun/CH4/MethaneAIR/rad_coef_50ms/rad_coef_o2_50ms_ord4_20200205T230528.mat';
        ch4_median_straylight_fn = '/home/kangsun/CH4/MethaneAIR/K_stable_CH4.mat';
        o2_median_straylight_fn = '/home/kangsun/CH4/MethaneAIR/K_stable_O2.mat';
        % data directory containing each wavelength as a directory
        ils_dir = '/home/kangsun/CH4/ISRF/CH4_20200125/';
        code_dir = '/home/kangsun/CH4/ISRF/';
        output_dir = '/home/kangsun/CH4/ISRF/output/';
end
% need nanconv.m at code_dir for stray light correction and fit_2D_data.m
% for orthogonal linear fitting
addpath(code_dir)
F_shift_scale = @(coeff,inp) interp1(inp.ref_x+coeff(2),inp.ref_y*coeff(1),inp.xx,'nearest','extrap');
nrow = 1280;
ncol = 1024;
% intial guess of wavelength registration to roughly locate the slit image
ch4_slope_guess = 0.1035;%nm per pix
ch4_intercept_guess = 1591.6;
o2_slope_guess = 0.0809345;
o2_intercept_guess = 1236.1;
ch4_w_range = 10;
o2_w_range = 10;
switch whichBand
    case 'CH4'
        % laser wavelength steps
        center_w_vec = [1593 1600:10:1670 1679];
        ch4_bad_pix = logical(load(ch4_bad_pix_fn));
        straylight_data = load(ch4_median_straylight_fn);
        % binning scheme of rows, 0.5:1:1280.5 means no binning
        %rowBinning = [0.5:6:1278.5];
		%[~,~,binSubs] = histcounts(1:1278,rowBinning);
		rowBinning = 0.5:1:1280.5;
        [~,~,binSubs] = histcounts(1:1280,rowBinning);
        uniqueBin = unique(binSubs);
    case 'O2'
        center_w_vec = 1247:7:1317;
        o2_bad_pix = logical(load(o2_bad_pix_fn));
        straylight_data = load(o2_median_straylight_fn);
        % binning scheme of rows, 0.5:1:1280.5 means no binning
        rowBinning = [0.5:6:1278.5];%rowBinning = 0.5:1:1280.5;
        [~,~,binSubs] = histcounts(1:1278,rowBinning);
        uniqueBin = unique(binSubs);
end
% load/process stray light kernel
K_far = straylight_data.medianStrayLight;
K_far(isnan(K_far)|K_far<0) = 0;
% if ifPlotDiagnose
%     figure;imagesc(K_far);set(gca,'colorscale','log')
% end
K_far = K_far(straylight_data.rowAllGrid > -350 & straylight_data.rowAllGrid < 350,...
    straylight_data.colAllGrid > -350 & straylight_data.colAllGrid < 350);
K_far = K_far/sum(K_far(:));
reducedRow = straylight_data.rowAllGrid(straylight_data.rowAllGrid > -350 & straylight_data.rowAllGrid < 350);
reducedCol = straylight_data.colAllGrid(straylight_data.colAllGrid > -350 & straylight_data.colAllGrid < 350);
K_far(reducedRow > -6 & reducedRow < 6,...
    reducedCol > -7 & reducedCol < 7) = 0;
if ifPlotDiagnose
    figure;imagesc(K_far);set(gca,'colorscale','log')
end
sum_K_far = sum(K_far(:));
ncenter = length(center_w_vec);
%%
% icenter loops over laser wavelength steps
for icenter = 1:ncenter
    %if pix_ext_left/right are 7.5, the ISRF extends across a 15-pix window
    pix_ext_left = 8.5;
    pix_ext_right = 8.5;
    % 0.05 nm (0.1 nm window) is enough to fill the gap
    wav_ext = 0.095;
    
    center_w = center_w_vec(icenter);
    if center_w > 1500
        x_w_range = ch4_w_range;
        x_slope_guess = ch4_slope_guess;
        x_intercept_guess = ch4_intercept_guess;
        x_bad_pix = ch4_bad_pix;
        x_camera = 'ch4';
        min_isrf = 1e-5;
        if ~exist('ch4_rad_coef','var')
            load(ch4_rad_cal_fn)
            ch4_rad_coef = coef;
            clear coef
        end
        x_rad_coef = ch4_rad_coef;
    else
        x_w_range = o2_w_range;
        x_slope_guess = o2_slope_guess;
        x_intercept_guess = o2_intercept_guess;
        x_bad_pix = o2_bad_pix;
        x_camera = 'o2';
        min_isrf = 1e-5;
        if ~exist('o2_rad_coef','var')
            load(o2_rad_cal_fn)
            o2_rad_coef = coef;
            clear coef
        end
        x_rad_coef = o2_rad_coef;
    end
    % pad x_rad_coef with zero intercept
    x_rad_coef = cat(3,zeros(nrow,ncol,1),x_rad_coef);
    % start/end of spectral pixel indices of interest
    pix_start = floor((center_w-x_w_range/2-x_intercept_guess)/x_slope_guess);
    pix_end = floor((center_w+x_w_range/2-x_intercept_guess)/x_slope_guess);
    pix_start = max(pix_start,1);
    pix_end = min(pix_end,ncol);
    col0_vec = pix_start:pix_end;
    ncol0 = length(col0_vec);
    cd([ils_dir,num2str(round(center_w))])
    flist = dir('SANTEC_*.csv');
    infoT = readtable(flist(1).name);
    nstep = size(infoT,1);
    dark_steps = [1,nstep];
    step_w_vec = zeros(nstep,1);
    issf_data = nan(nrow,ncol,nstep);
    for istep = 1:nstep
        step_w = str2double(infoT.Var8{istep}(9:16));
        step_w_vec(istep) = step_w;
        fn = [x_camera,'_camera_',num2str(infoT.Var3(istep)),'_',num2str(infoT.Var4(istep),'%02d'),...
            '_',num2str(infoT.Var5(istep),'%02d'),'_',num2str(infoT.Var6(istep),'%02d'),'_',...
            num2str(infoT.Var7(istep),'%02d'),'_',infoT.Var8{istep}(1:2),'.mat'];
        if ~exist(fn,'file')
            warning([fn,' does not exist!'])
            continue
        end
        ils_data = load(fn);
        data = ils_data.data;
        data(x_bad_pix) = nan;
        issf_data(:,:,istep) = data;
        if ~exist('frameTime','var')
            frameTime = ils_data.meta.frameTime(1);
        end
        
    end
    dark_mean = squeeze(nanmean(issf_data(:,:,dark_steps),3));
    use_step_int = step_w_vec >= center_w-wav_ext & step_w_vec <= center_w+wav_ext;
    use_step_int([1,nstep]) = false;
    tmp_idx = 1:nstep;
    use_steps = tmp_idx(use_step_int);
    for istep = use_steps
        %% stray light correction
        if doStrayLight
            step_image0 = squeeze(issf_data(:,:,istep))-dark_mean;
            step_image = step_image0;
            
            for iter = 1:niter
                step_image = (step_image0-nanconv(step_image,K_far))/(1-sum_K_far);
            end
            issf_data(:,:,istep) = step_image;
        else
            issf_data(:,:,istep) = squeeze(issf_data(:,:,istep))-dark_mean;
        end
        if ifPlotDiagnose% && ~doStrayLight
            clf
            subplot(2,1,1)
            imagesc(squeeze(issf_data(:,:,istep)))
            set(gca,'colorscale','log','ydir','norm')
            xlim([ncol-pix_end ncol-pix_start])
            title([num2str(step_w_vec(istep),'%.3f'),' nm'])
            xlabel('Column index')
            ylabel('Row index')
            hc = colorbar;
            set(get(hc,'ylabel'),'string','Digital Number')
            subplot(2,1,2)
            imagesc(squeeze(issf_data(:,:,istep)))
            set(gca,'colorscale','lin','ydir','norm')
            xlim([ncol-pix_end ncol-pix_start])
            title([num2str(step_w_vec(istep),'%.3f'),' nm'])
            xlabel('Column index')
            ylabel('Row index')
            hc = colorbar;
            set(get(hc,'ylabel'),'string','Digital Number')
            
            %         export_fig(['C:\research\CH4\issf_data',num2str(istep),'.png'])
            pause
        end
    end
    % flip column order to match level 1b convention
    issf_data = issf_data(:,end:-1:1,:);
    % flip row order for O2 camera
    if strcmpi(whichBand,'O2')
        issf_data = issf_data(end:-1:1,:,:);
    end
    % extract only useful spectral pixels
    issf_data = issf_data(:,pix_start:pix_end,:);
    %%
    % rad cal, remember to flip the columns of the coef, and the coef it self
    for irow = 1:size(issf_data,1)
        row_rad_coef = squeeze(x_rad_coef(irow,end:-1:1,end:-1:1));
        for icol = 1:ncol0
            local_poly = row_rad_coef(col0_vec(icol),:);
            issf_data(irow,icol,:) = polyval(local_poly,issf_data(irow,icol,:)./frameTime);
        end
    end
    
    %% bin the rows
    issf_reduced_data = nan(length(uniqueBin),size(issf_data,2),size(issf_data,3));
    for ibin = 1:length(uniqueBin)
        issf_reduced_data(ibin,:,:) = nanmean(issf_data(binSubs==uniqueBin(ibin),:,:),1);
    end
    % number of footprints, binned (reduced) from number of rows
    nft = size(issf_reduced_data,1);
    if ifPlotDiagnose
        for istep = use_steps
            clf
            subplot(2,1,1)
            imagesc(pix_start:pix_end,1:length(uniqueBin),squeeze(issf_reduced_data(:,:,istep)))
            set(gca,'colorscale','log','ydir','norm')
            title([num2str(step_w_vec(istep),'%.3f'),' nm'])
            xlabel('Spectral pixel')
            ylabel('Aggregated rows/footprints')
            hc = colorbar;
            set(get(hc,'ylabel'),'string','Radiance')
            %         xlim([pix_start pix_end])
            subplot(2,1,2)
            imagesc(pix_start:pix_end,1:length(uniqueBin),squeeze(issf_reduced_data(:,:,istep)))
            set(gca,'colorscale','lin','ydir','norm')
            hc = colorbar;
            set(get(hc,'ylabel'),'string','Radiance')
            xlabel('Spectral pixel')
            ylabel('Aggregated rows/footprints')
            title([num2str(step_w_vec(istep),'%.3f'),' nm'])
            %         xlim([pix_start pix_end])
            pause
        end
    end
    %%
    final_center_pix_vec = nan(1,nft);
    xx_all_row = nan(nft,700);
    yy_all_row = nan(nft,700);
    %% (parallel) loop over the rows, change for to parfor to run parallel
    parfor ift = 1:nft
        row_data = squeeze(issf_reduced_data(ift,:,:));
        if sum(isnan(nanmean(row_data,2))) > 500
            disp(['Footprint ',num2str(ift),' seems empty'])
            continue
        end
        %         semilogy(col0_vec,tmp)
        try
            %% issf1
            if ifPlotDiagnose
                close all;figure('color','w','unit','inch','position',[0 1,15 5])
                cc = jet(length(use_steps));
                hold on
                legstr = cell(0);
                count = 0;
            end
            scales = nan(nstep,1);
            scales0 = scales;
            centers_of_mass = nan(nstep,1);
            for istep = use_steps
                yy = row_data(:,istep);
                int = ~isnan(yy);
                yy = yy(int);
                xx = col0_vec(int);
                xx = xx(:);
                yy = yy(:);
                centers_of_mass(istep) = trapz(xx,xx.*yy)/trapz(xx,yy);
                scales(istep) = trapz(xx,yy)/nanmedian(abs(diff(xx)));
                scales0(istep) = sum(yy);
                if ifPlotDiagnose
                    count = count+1;
                    h = plot(xx,yy,'-','linewidth',1,'color',cc(count,:));
                    set(h,'marker','.','markersize',16)
                    legstr = cat(1,legstr,{[num2str(step_w_vec(istep)-center_w,'%.3f'),'nm, COM=',...
                    num2str(centers_of_mass(istep),'%.2f')]});
                end
            end
            if ifPlotDiagnose
                hleg = legend(legstr);
                set(hleg,'Box','off','NumColumns',2,'fontsize',10)
                xlim([min(centers_of_mass)-7,max(centers_of_mass)+9])
                set(gca,'ycolor','none','linewidth',1)
                xlabel(['Columns of footprint ',num2str(ift),', laser central wavelength = ',num2str(center_w),' nm'])
            end
            [~,pp] = fit_2D_data(step_w_vec(~isnan(centers_of_mass)),centers_of_mass(~isnan(centers_of_mass)),false);
            center_pix = polyval(pp,center_w);
            %% look at the difference between OLS and orthogonal fit. They're very close, but not identical
            if ifPlotDiagnose
                pp_ols = polyfit(step_w_vec(~isnan(centers_of_mass)),centers_of_mass(~isnan(centers_of_mass)),1);
                clf
                plot(step_w_vec(~isnan(centers_of_mass)),centers_of_mass(~isnan(centers_of_mass)),'ok',...
                    step_w_vec,polyval(pp_ols,step_w_vec),'-b',...
                    step_w_vec,polyval(pp,step_w_vec),'-r')
                legend('Wavelength vs. spectral pixel','OLS fit','Orthogonal fit')
            end
            %% isrf1
            pix_mat = nan(ncol0,nstep);
            for istep = use_steps
                pix_mat(:,istep) = col0_vec-(centers_of_mass(istep)-center_pix);
            end
            wint = step_w_vec >= center_w-wav_ext & step_w_vec <= center_w+wav_ext;
            xx = pix_mat(:,wint);
            yy = squeeze(row_data(:,wint))./repmat(scales(wint),[1,ncol0])';
            xx = xx(:);
            yy = yy(:);
            yy = yy(xx >= center_pix-pix_ext_left & xx <= center_pix+pix_ext_right);
            xx = xx(xx >= center_pix-pix_ext_left & xx <= center_pix+pix_ext_right);
            [xx,I] = sort(xx);yy = yy(I);
            int = xx > 0 & yy > 0;
            xx = xx(int);
            yy = yy(int);
            isrf1_x = xx-center_pix;
            isrf1_y = yy;
            center_pix1 = center_pix;
            if ifPlotDiagnose
                clf
                count = 0;
                hold on
                for istep = use_steps
                    count = count+1;
                    yy = row_data(:,istep);
                    int = ~isnan(yy);
                    yy = yy(int);
                    xx = col0_vec(int);
                    xx = xx(:)-centers_of_mass(istep)+center_pix1;
                    yy = yy(:)/scales(istep);
                    
                    h = plot(xx,yy,'.','linewidth',1,'color',cc(count,:));
                    set(h,'marker','.','markersize',16)
                end
                xlim([center_pix1-7,center_pix1+9])
                hleg = legend(legstr);
                set(hleg,'Box','off','NumColumns',2,'fontsize',10)
                set(gca,'ycolor','none','linewidth',1)
                set(gca,'yscale','log')
%                 set(gca,'yscale','lin')
                xlabel(['Columns of footprint ',num2str(ift),', laser central wavelength = ',...
                    num2str(center_w),' nm, central column = ',num2str(center_pix1,'%.3f')])
            end
            %% issf2
            if ifPlotDiagnose
                close all;figure('color','w','unit','inch','position',[0 1,15 5])
                cc = jet(length(use_steps));
                hold on
                legstr = cell(0);
                count = 0;
            end
            scales2 = nan(nstep,1);
            centers_of_mass2 = nan(nstep,1);
            warning off
            for istep = use_steps
                yy = row_data(:,istep);
                [~,max_y_id] = max(yy);
                xx = col0_vec(max(max_y_id-6,1):min(max_y_id+6,ncol0));
                yy = yy(max(max_y_id-6,1):min(max_y_id+6,ncol0));
                xx = xx(~isnan(yy));
                yy = yy(~isnan(yy));
                xx = xx(:);yy = yy(:);
                inp_issf2 = [];
                inp_issf2.xx = xx;
                inp_issf2.ref_x = isrf1_x;
                inp_issf2.ref_y = isrf1_y;
                prior_issf2 = [scales0(istep) trapz(xx,xx.*yy)/trapz(xx,yy)];
                posterior_issf2 = nlinfit(inp_issf2,yy,F_shift_scale,prior_issf2);
                
                scales2(istep) = posterior_issf2(1);
                centers_of_mass2(istep) = posterior_issf2(2);
                if ifPlotDiagnose
                    count = count+1;
                    h = plot(xx,yy,'-','linewidth',1,'color',cc(count,:));
                    set(h,'marker','.','markersize',16)
                    legstr = cat(1,legstr,{[num2str(step_w_vec(istep)-center_w,'%.3f'),'nm, COM=',...
                    num2str(centers_of_mass(istep),'%.2f')]});
                end
            end
            if ifPlotDiagnose
                hleg = legend(legstr);
                set(hleg,'Box','off','NumColumns',2,'fontsize',10)
                xlim([min(centers_of_mass)-7,max(centers_of_mass)+9])
                set(gca,'ycolor','none','linewidth',1)
                xlabel(['Columns of footprint ',num2str(ift),', laser central wavelength = ',num2str(center_w),' nm'])
            end
            scales2(scales2 < 0.9 *nanmedian(scales2) | scales2 > 1.1 *nanmedian(scales2)) = nan;
            warning on
%             pp = polyfit(step_w_vec(~isnan(centers_of_mass2)),centers_of_mass2(~isnan(centers_of_mass2)),1);
            [~,pp] = fit_2D_data(step_w_vec(~isnan(centers_of_mass2)),centers_of_mass2(~isnan(centers_of_mass2)),false);
            center_pix = polyval(pp,center_w);
            %% isrf2
            pix_mat = nan(ncol0,nstep);
            for istep = use_steps
                pix_mat(:,istep) = col0_vec-(centers_of_mass2(istep)-center_pix);
            end
            wint = step_w_vec >= center_w-wav_ext & step_w_vec <= center_w+wav_ext;
            xx = pix_mat(:,wint);
            yy = squeeze(row_data(:,wint))./repmat(scales2(wint),[1,ncol0])';
            xx = xx(:);
            yy = yy(:);
            yy = yy(xx >= center_pix-pix_ext_left & xx <= center_pix+pix_ext_right);
            xx = xx(xx >= center_pix-pix_ext_left & xx <= center_pix+pix_ext_right);
            [xx,I] = sort(xx);yy = yy(I);
            int = xx > 0 & yy > 0;
            xx = xx(int);
            yy = yy(int);
            
            isrf2_x = xx-center_pix;
            isrf2_y = yy;
            center_pix2 = center_pix;
            if ifPlotDiagnose
                clf
                count = 0;
                hold on
                for istep = use_steps
                    count = count+1;
                    yy = row_data(:,istep);
                    int = ~isnan(yy);
                    yy = yy(int);
                    xx = col0_vec(int);
                    xx = xx(:)-centers_of_mass(istep)+center_pix1;
                    yy = yy(:)/scales(istep);
                    
                    h = plot(xx,yy,'.','linewidth',1,'color',cc(count,:));
                    set(h,'marker','.','markersize',16)
                end
                xlim([center_pix1-7,center_pix1+9])
                hleg = legend(legstr);
                set(hleg,'Box','off','NumColumns',2,'fontsize',10)
                set(gca,'ycolor','none','linewidth',1)
                set(gca,'yscale','log')
                set(gca,'yscale','lin')
                xlabel(['Columns of footprint ',num2str(ift),', laser central wavelength = ',...
                    num2str(center_w),' nm, central column = ',num2str(center_pix1,'%.3f')])
            end
            %% issf3
            if ifPlotDiagnose
                close all;figure('color','w','unit','inch','position',[0 1,15 5])
                cc = jet(length(use_steps));
                hold on
                legstr = cell(0);
                count = 0;
            end
            scales3 = nan(nstep,1);
            centers_of_mass3 = nan(nstep,1);
            warning off
            for istep = use_steps
                yy = row_data(:,istep);
                [~,max_y_id] = max(yy);
                xx = col0_vec(max(max_y_id-6,1):min(max_y_id+6,ncol0));
                yy = yy(max(max_y_id-6,1):min(max_y_id+6,ncol0));
                xx = xx(~isnan(yy));
                yy = yy(~isnan(yy));
                xx = xx(:);yy = yy(:);
                inp_issf2 = [];
                inp_issf2.xx = xx;
                inp_issf2.ref_x = isrf2_x;
                inp_issf2.ref_y = isrf2_y;
                prior_issf2 = [scales0(istep) trapz(xx,xx.*yy)/trapz(xx,yy)];
                posterior_issf2 = nlinfit(inp_issf2,yy,F_shift_scale,prior_issf2);
                
                scales3(istep) = posterior_issf2(1);
                centers_of_mass3(istep) = posterior_issf2(2);
                if ifPlotDiagnose
                    count = count+1;
                    h = plot(xx,yy,'-','linewidth',1,'color',cc(count,:));
                    set(h,'marker','.','markersize',16)
                    legstr = cat(1,legstr,{[num2str(step_w_vec(istep)-center_w,'%.3f'),'nm, COM=',...
                    num2str(centers_of_mass(istep),'%.2f')]});
                end
            end
            if ifPlotDiagnose
                hleg = legend(legstr);
                set(hleg,'Box','off','NumColumns',2,'fontsize',10)
                xlim([min(centers_of_mass)-7,max(centers_of_mass)+9])
                set(gca,'ycolor','none','linewidth',1)
                xlabel(['Columns of footprint ',num2str(ift),', laser central wavelength = ',num2str(center_w),' nm'])
            end
            warning on
            scales3(scales3 < 0.9 *nanmedian(scales3) | scales3 > 1.1 *nanmedian(scales3)) = nan;
%             pp = polyfit(step_w_vec(~isnan(centers_of_mass3)),centers_of_mass3(~isnan(centers_of_mass3)),1);
            [~,pp] = fit_2D_data(step_w_vec(~isnan(centers_of_mass3)),centers_of_mass3(~isnan(centers_of_mass3)),false);
            center_pix = polyval(pp,center_w);
            %% isrf3
            pix_mat = nan(ncol0,nstep);
            for istep = use_steps
                pix_mat(:,istep) = col0_vec-(centers_of_mass3(istep)-center_pix);
            end
            wint = step_w_vec >= center_w-wav_ext & step_w_vec <= center_w+wav_ext;
            xx = pix_mat(:,wint);
            yy = squeeze(row_data(:,wint))./repmat(scales3(wint),[1,ncol0])';
            xx = xx(:);
            yy = yy(:);
            yy = yy(xx >= center_pix-pix_ext_left & xx <= center_pix+pix_ext_right);
            xx = xx(xx >= center_pix-pix_ext_left & xx <= center_pix+pix_ext_right);
            [xx,I] = sort(xx);yy = yy(I);
            int = xx > 0 & yy > 0;
            xx = xx(int);
            yy = yy(int);
            isrf3_x = xx-center_pix;
            isrf3_y = yy;
            center_pix3 = center_pix;
            if ifPlotDiagnose
                clf
                count = 0;
                hold on
                for istep = use_steps
                    count = count+1;
                    yy = row_data(:,istep);
                    int = ~isnan(yy);
                    yy = yy(int);
                    xx = col0_vec(int);
                    xx = xx(:)-centers_of_mass(istep)+center_pix1;
                    yy = yy(:)/scales(istep);
                    
                    h = plot(xx,yy,'.','linewidth',1,'color',cc(count,:));
                    set(h,'marker','.','markersize',16)
                end
                xlim([center_pix1-7,center_pix1+9])
                hleg = legend(legstr);
                set(hleg,'Box','off','NumColumns',2,'fontsize',10)
                set(gca,'ycolor','none','linewidth',1)
                set(gca,'yscale','log')
                set(gca,'yscale','lin')
                xlabel(['Columns of footprint ',num2str(ift),', laser central wavelength = ',...
                    num2str(center_w),' nm, central column = ',num2str(center_pix1,'%.3f')])
            end
            %% issf4
            if ifPlotDiagnose
                close all;figure('color','w','unit','inch','position',[0 1,15 5])
                cc = jet(length(use_steps));
                hold on
                legstr = cell(0);
                count = 0;
            end
            scales4 = nan(nstep,1);
            centers_of_mass4 = nan(nstep,1);
            warning off
            for istep = use_steps
                yy = row_data(:,istep);
                [~,max_y_id] = max(yy);
                xx = col0_vec(max(max_y_id-6,1):min(max_y_id+6,ncol0));
                yy = yy(max(max_y_id-6,1):min(max_y_id+6,ncol0));
                xx = xx(~isnan(yy));
                yy = yy(~isnan(yy));
                xx = xx(:);yy = yy(:);
                inp_issf2 = [];
                inp_issf2.xx = xx;
                inp_issf2.ref_x = isrf3_x;
                inp_issf2.ref_y = isrf3_y;
                prior_issf2 = [scales0(istep) trapz(xx,xx.*yy)/trapz(xx,yy)];
                posterior_issf2 = nlinfit(inp_issf2,yy,F_shift_scale,prior_issf2);
                
                scales4(istep) = posterior_issf2(1);
                centers_of_mass4(istep) = posterior_issf2(2);
                if ifPlotDiagnose
                    count = count+1;
                    h = plot(xx,yy,'-','linewidth',1,'color',cc(count,:));
                    set(h,'marker','.','markersize',16)
                    legstr = cat(1,legstr,{[num2str(step_w_vec(istep)-center_w,'%.3f'),'nm, COM=',...
                    num2str(centers_of_mass(istep),'%.2f')]});
                end
            end
            if ifPlotDiagnose
                hleg = legend(legstr);
                set(hleg,'Box','off','NumColumns',2,'fontsize',10)
                xlim([min(centers_of_mass)-7,max(centers_of_mass)+9])
                set(gca,'ycolor','none','linewidth',1)
                xlabel(['Columns of footprint ',num2str(ift),', laser central wavelength = ',num2str(center_w),' nm'])
            end
            warning on
            scales4(scales4 < 0.9 *nanmedian(scales4) | scales4 > 1.1 *nanmedian(scales4)) = nan;
%             pp = polyfit(step_w_vec(~isnan(centers_of_mass4)),centers_of_mass4(~isnan(centers_of_mass4)),1);
            [~,pp] = fit_2D_data(step_w_vec(~isnan(centers_of_mass4)),centers_of_mass4(~isnan(centers_of_mass4)),false);
            center_pix = polyval(pp,center_w);
            %% isrf4
            pix_mat = nan(ncol0,nstep);
            for istep = use_steps
                pix_mat(:,istep) = col0_vec-(centers_of_mass4(istep)-center_pix);
            end
            wint = step_w_vec >= center_w-wav_ext & step_w_vec <= center_w+wav_ext;
            xx = pix_mat(:,wint);
            yy = squeeze(row_data(:,wint))./repmat(scales4(wint),[1,ncol0])';
            xx = xx(:);
            yy = yy(:);
            yy = yy(xx >= center_pix-pix_ext_left & xx <= center_pix+pix_ext_right);
            xx = xx(xx >= center_pix-pix_ext_left & xx <= center_pix+pix_ext_right);
            [xx,I] = sort(xx);yy = yy(I);
            int = xx > 0 & yy > 0;
            xx = xx(int);
            yy = yy(int);
            isrf4_x = xx-center_pix;
            isrf4_y = yy;
            center_pix4 = center_pix;
            xx = isrf4_x+center_pix4;
            yy = isrf4_y;
            int = yy >= min_isrf;
            xx = xx(int);
            yy = yy(int);
            final_center_pix_vec(ift) = center_pix4;
            if ifPlotDiagnose
                clf
                count = 0;
                hold on
                for istep = use_steps
                    count = count+1;
                    yy = row_data(:,istep);
                    int = ~isnan(yy);
                    yy = yy(int);
                    xx = col0_vec(int);
                    xx = xx(:)-centers_of_mass(istep)+center_pix1;
                    yy = yy(:)/scales(istep);
                    
                    h = plot(xx,yy,'.','linewidth',1,'color',cc(count,:));
                    set(h,'marker','.','markersize',16)
                end
                xlim([center_pix1-7,center_pix1+9])
                hleg = legend(legstr);
                set(hleg,'Box','off','NumColumns',2,'fontsize',10)
                set(gca,'ycolor','none','linewidth',1)
                set(gca,'yscale','log')
                set(gca,'yscale','lin')
                xlabel(['Columns of footprint ',num2str(ift),', laser central wavelength = ',...
                    num2str(center_w),' nm, central column = ',num2str(center_pix1,'%.3f')])
            end
            %% additional diagnostics
            if ifPlotDiagnose
                clf
                subplot(2,1,1)
                plot(step_w_vec,scales0,'k',step_w_vec,scales2,'-o',...
                    step_w_vec,scales3,'-*',step_w_vec,scales4,'-s')
                subplot(2,1,2)
                plot(step_w_vec,centers_of_mass-center_pix1,'k',...
                    step_w_vec,centers_of_mass2-center_pix2,'-o',...
                    step_w_vec,centers_of_mass3-center_pix3,'-*',...
                    step_w_vec,centers_of_mass4-center_pix4,'-s')
            end
        catch
            warning(['wavelength ',num2str(center_w),', footprint ',num2str(ift),' failed oversampling!']);
            continue
        end
        xx_all = nan(700,1);
        xx_all(1:length(xx)) = xx;
        xx_all_row(ift,:) = xx_all;
        xx_all(1:length(xx)) = yy;
        yy_all_row(ift,:) = xx_all;
    end
    save([output_dir,whichBand,'_ISRF_',num2str(center_w,'%.1f'),'.mat'],...
        'xx_all_row','yy_all_row','nrow','ncol','nft','rowBinning',...
        'final_center_pix_vec',...
        'step_w_vec','nstep','whichMachine','whichBand','doStrayLight')
end
